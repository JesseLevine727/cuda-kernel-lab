from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path
from statistics import mean, median

import torch

from cuda_kernel_lab.tasks import get_task


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
)


def static_check(source: str) -> tuple[bool, str | None]:
    for snippet in BANNED_SNIPPETS:
        if snippet in source:
            return False, f"banned snippet {snippet!r}"
    if "class ModelNew" not in source:
        return False, "missing class ModelNew"
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


def evaluate(task_id: str, source_path: Path, seeds: list[int], warmup: int, iters: int) -> dict:
    started = time.perf_counter()
    result = {
        "task_id": task_id,
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
    }

    source = source_path.read_text()
    ok, reason = static_check(source)
    if not ok:
        result.update({"failure_type": "static_reject", "error": reason})
        result["elapsed_s"] = time.perf_counter() - started
        return result

    device = torch.device("cuda")
    task = get_task(task_id)

    try:
        ModelNew = load_model(source_path)
        model = ModelNew().to(device)
        if hasattr(model, "eval"):
            model.eval()
        result["compiled"] = True
    except Exception:
        result.update(
            {
                "failure_type": "compile_error",
                "error": traceback.format_exc(limit=8),
                "elapsed_s": time.perf_counter() - started,
            }
        )
        return result

    try:
        with torch.no_grad():
            for seed in seeds:
                args = task.input_factory(seed, device)
                expected = task.reference(*args)
                actual = model(*args)
                torch.cuda.synchronize()
                max_abs = (actual - expected).abs().max().item()
                passed = torch.allclose(actual, expected, atol=task.atol, rtol=task.rtol)
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
                    return result

            timing_args = task.input_factory(seeds[0], device)
            reference_timing = time_call(task.reference, timing_args, warmup, iters)
            candidate_timing = time_call(model, timing_args, warmup, iters)
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
            return result
    except Exception:
        result.update(
            {
                "failure_type": "runtime_error",
                "error": traceback.format_exc(limit=8),
                "elapsed_s": time.perf_counter() - started,
            }
        )
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]
    result = evaluate(args.task_id, args.source, seeds, args.warmup, args.iters)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

