from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from statistics import mean, median

import torch

from cuda_kernel_lab.backends import cuda_cpp
from cuda_kernel_lab.tasks import get_task


class FDCapture:
    def __enter__(self):
        sys.stdout.flush()
        sys.stderr.flush()
        self._old_stdout = os.dup(1)
        self._old_stderr = os.dup(2)
        self._stdout = tempfile.TemporaryFile(mode="w+")
        self._stderr = tempfile.TemporaryFile(mode="w+")
        os.dup2(self._stdout.fileno(), 1)
        os.dup2(self._stderr.fileno(), 2)
        self.stdout = ""
        self.stderr = ""
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self._old_stdout, 1)
        os.dup2(self._old_stderr, 2)
        os.close(self._old_stdout)
        os.close(self._old_stderr)
        self._stdout.seek(0)
        self._stderr.seek(0)
        self.stdout = self._stdout.read()
        self.stderr = self._stderr.read()
        self._stdout.close()
        self._stderr.close()
        return False


BANNED_SNIPPETS = (
    "subprocess",
    "os.system",
    "socket",
    "requests",
    "urllib",
    "open(",
    "exec(",
    "eval(",
    "sys.modules",
    "torch.cuda.synchronize =",
    "torch.allclose =",
    "time_call =",
)

MAX_SOURCE_BYTES = 64_000


def static_check(source: str, backend: str) -> tuple[bool, str | None]:
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        return False, f"source exceeds {MAX_SOURCE_BYTES} bytes"
    for snippet in BANNED_SNIPPETS:
        if snippet in source:
            return False, f"banned snippet {snippet!r}"
    if "class ModelNew" not in source:
        return False, "missing class ModelNew"
    if backend == "cuda_cpp":
        required = (
            "load_inline",
            "CUDA_SOURCE",
            "CPP_SOURCE",
            "PYBIND11_MODULE",
        )
        for snippet in required:
            if snippet not in source:
                return False, f"missing native CUDA requirement {snippet!r}"
        compact_source = "".join(source.split())
        if "functions=None" not in compact_source:
            return False, "missing native CUDA requirement 'functions=None'"
    if backend == "triton":
        required = ("triton", "@triton.jit")
        for snippet in required:
            if snippet not in source:
                return False, f"missing Triton requirement {snippet!r}"
    return True, None


def load_model(source_path: Path):
    spec = importlib.util.spec_from_file_location("candidate_kernel", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create import spec for {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop("candidate_kernel", None)
    spec.loader.exec_module(module)
    return module.ModelNew


def time_call(fn, args: tuple, warmup: int, iters: int) -> dict:
    for _ in range(warmup):
        out = fn(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return {
        "median_us": median(times),
        "mean_us": mean(times),
        "min_us": min(times),
        "max_us": max(times),
        "iters": iters,
    }


def evaluate(
    task_id: str,
    source_path: Path,
    seeds: list[int],
    warmup: int,
    iters: int,
    backend: str,
) -> dict:
    started = time.perf_counter()
    result = {
        "task_id": task_id,
        "backend": backend,
        "source_path": str(source_path),
        "compiled": False,
        "correct": False,
        "speedup": None,
        "failure_type": None,
        "error": None,
        "per_seed": [],
        "candidate_timing": None,
        "reference_timing": None,
        "elapsed_s": None,
        "compile_stdout": "",
        "compile_stderr": "",
        "runtime_stdout": "",
        "runtime_stderr": "",
        "backend_env": None,
    }

    source = source_path.read_text()
    ok, reason = static_check(source, backend)
    if not ok:
        result.update({"failure_type": "static_reject", "error": reason})
        result["elapsed_s"] = time.perf_counter() - started
        return result

    device = torch.device("cuda")
    task = get_task(task_id)
    build_dir = source_path.parent / f"{source_path.stem}_build"
    old_env = None
    if backend == "cuda_cpp":
        old_env = cuda_cpp.configure_environment(build_dir)
        result["backend_env"] = cuda_cpp.environment_summary()

    try:
        with FDCapture() as compile_capture:
            ModelNew = load_model(source_path)
            model = ModelNew().to(device)
        if hasattr(model, "eval"):
            model.eval()
        result["compiled"] = True
        result["compile_stdout"] = compile_capture.stdout
        result["compile_stderr"] = compile_capture.stderr
    except Exception:
        capture = locals().get("compile_capture")
        result["compile_stdout"] = getattr(capture, "stdout", "")
        result["compile_stderr"] = getattr(capture, "stderr", "")
        result.update(
            {
                "failure_type": "compile_error",
                "error": traceback.format_exc(limit=8),
                "elapsed_s": time.perf_counter() - started,
            }
        )
        if old_env is not None:
            cuda_cpp.restore_environment(old_env)
        return result

    try:
        with torch.no_grad():
            runtime_stdout = []
            runtime_stderr = []
            for seed in seeds:
                args = task.input_factory(seed, device)
                expected = task.reference(*args)
                with FDCapture() as runtime_capture:
                    actual = model(*args)
                runtime_stdout.append(runtime_capture.stdout)
                runtime_stderr.append(runtime_capture.stderr)
                torch.cuda.synchronize()
                max_abs = (actual - expected).abs().max().item()
                passed = torch.allclose(actual, expected, atol=task.atol, rtol=task.rtol)
                result["runtime_stdout"] = "".join(runtime_stdout)
                result["runtime_stderr"] = "".join(runtime_stderr)
                result["per_seed"].append(
                    {"seed": seed, "passed": bool(passed), "max_abs_error": max_abs}
                )
                if not passed:
                    result.update(
                        {
                            "failure_type": "correctness_failure",
                            "error": f"seed={seed} max_abs_error={max_abs}",
                            "elapsed_s": time.perf_counter() - started,
                        }
                    )
                    if old_env is not None:
                        cuda_cpp.restore_environment(old_env)
                    return result

            timing_args = task.input_factory(seeds[0], device)
            reference_timing = time_call(task.reference, timing_args, warmup, iters)
            with FDCapture() as runtime_capture:
                candidate_timing = time_call(model, timing_args, warmup, iters)
            runtime_stdout.append(runtime_capture.stdout)
            runtime_stderr.append(runtime_capture.stderr)
            result["runtime_stdout"] = "".join(runtime_stdout)
            result["runtime_stderr"] = "".join(runtime_stderr)
            ref_us = reference_timing["median_us"]
            cand_us = candidate_timing["median_us"]
            speedup = ref_us / cand_us if cand_us > 0 else None
            result.update(
                {
                    "correct": True,
                    "failure_type": None,
                    "error": None,
                    "reference_timing": reference_timing,
                    "candidate_timing": candidate_timing,
                    "speedup": speedup,
                    "elapsed_s": time.perf_counter() - started,
                }
            )
            if old_env is not None:
                cuda_cpp.restore_environment(old_env)
            return result
    except Exception:
        result["runtime_stdout"] = "".join(locals().get("runtime_stdout", []))
        result["runtime_stderr"] = "".join(locals().get("runtime_stderr", []))
        result.update(
            {
                "failure_type": "runtime_error",
                "error": traceback.format_exc(limit=8),
                "elapsed_s": time.perf_counter() - started,
            }
        )
        if old_env is not None:
            cuda_cpp.restore_environment(old_env)
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--backend", choices=("auto", "cuda_cpp", "triton"), default="auto")
    args = parser.parse_args()

    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]
    task = get_task(args.task_id)
    backend = task.backend if args.backend == "auto" else args.backend
    result = evaluate(args.task_id, args.source, seeds, args.warmup, args.iters, backend)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
