from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from cuda_kernel_lab.run_gemma_eval import summarize


def repair_source(source: str) -> tuple[str, list[str]]:
    actions: list[str] = []
    repaired = source

    replacements = {
        "super().___init__()": "super().__init__()",
        "super()._init__()": "super().__init__()",
        "torcheres::Tensor": "torch::Tensor",
        "FLINF": "3.402823466e38f",
    }
    for before, after in replacements.items():
        if before in repaired:
            repaired = repaired.replace(before, after)
            actions.append(f"replace {before!r} with {after!r}")

    new_lines = []
    in_raw_string = False
    for line in repaired.splitlines():
        if 'r"""' in line or '"""' in line:
            in_raw_string = not in_raw_string
        stripped = line.lstrip()
        if (
            not in_raw_string
            and line != stripped
            and (
                stripped.startswith("ext = load_inline(")
                or stripped.startswith("ext=load_inline(")
                or stripped.startswith("class ModelNew(")
            )
        ):
            new_lines.append(stripped)
            actions.append("dedent top-level Python statement")
        else:
            new_lines.append(line)
    repaired = "\n".join(new_lines) + ("\n" if source.endswith("\n") else "")

    repaired2 = re.sub(
        r"\b(?:const\s+)?float\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\.data_ptr",
        r"\1.data_ptr",
        repaired,
    )
    if repaired2 != repaired:
        repaired = repaired2
        actions.append("remove C++ type specifier from data_ptr kernel argument")

    repaired2 = re.sub(
        r"\b(?:const\s+)?int\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\.data_ptr",
        r"\1.data_ptr",
        repaired,
    )
    if repaired2 != repaired:
        repaired = repaired2
        actions.append("remove C++ int pointer specifier from data_ptr kernel argument")

    # Some generations leave one top-level assignment indented but keep valid
    # continuation indentation inside the open parenthesis. Normalize only the
    # top-level statement, not C++ source strings.
    repaired2 = repaired.replace("\n ext = load_inline(", "\next = load_inline(")
    if repaired2 != repaired:
        repaired = repaired2
        actions.append("dedent load_inline assignment by literal replacement")

    repaired2 = repaired.replace("\n class ModelNew", "\nclass ModelNew")
    if repaired2 != repaired:
        repaired = repaired2
        actions.append("dedent ModelNew class by literal replacement")

    repaired2 = re.sub(r"(dim3\s+dimGrid\([^;\n]+)\)\);", r"\1);", repaired)
    if repaired2 != repaired:
        repaired = repaired2
        actions.append("remove extra closing parenthesis from dimGrid constructor")

    repaired_lines = []
    changed_return = False
    for line in repaired.splitlines():
        stripped = line.strip()
        if stripped.startswith("return ext.") and stripped.endswith("))"):
            line = line.rstrip()[:-1]
            changed_return = True
        repaired_lines.append(line)
    if changed_return:
        repaired = "\n".join(repaired_lines) + ("\n" if repaired.endswith("\n") else "")
        actions.append("remove extra closing parenthesis from ext return")

    return repaired, sorted(set(actions))


def evaluate_candidate(task_id: str, source_path: Path, timeout: int) -> dict[str, Any]:
    cmd = [
        "python3",
        "-m",
        "cuda_kernel_lab.kernel_eval",
        "--task-id",
        task_id,
        "--source",
        str(source_path),
        "--backend",
        "cuda_cpp",
    ]
    started = time.perf_counter()
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        return {
            "compiled": False,
            "correct": False,
            "failure_type": "evaluator_error",
            "error": proc.stderr or proc.stdout,
            "subprocess_elapsed_s": elapsed,
        }
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {
            "compiled": False,
            "correct": False,
            "failure_type": "evaluator_json_error",
            "error": proc.stdout + proc.stderr,
            "subprocess_elapsed_s": elapsed,
        }
    result["subprocess_elapsed_s"] = elapsed
    return result


def load_text(path: str | None) -> str:
    if not path:
        return ""
    return Path(path).read_text()


def repair_results(results_paths: list[Path], run_dir: Path, eval_timeout: int) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "run_name": run_dir.name,
        "kind": "mechanical_repair",
        "source_results": [str(path) for path in results_paths],
        "backend": "cuda_cpp",
        "eval_timeout": eval_timeout,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    grouped: dict[str, dict[str, Any]] = {}
    per_task_attempts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    repair_index = 0

    for results_path in results_paths:
        data = json.loads(results_path.read_text())
        source_run = data.get("metadata", {}).get("run_name", results_path.parent.name)
        for task in data.get("results", []):
            if task.get("backend") != "cuda_cpp":
                continue
            grouped.setdefault(
                task["task_id"],
                {
                    "task_id": task["task_id"],
                    "task_name": task["task_name"],
                    "level": task["level"],
                    "backend": "cuda_cpp",
                    "attempts": [],
                },
            )
            for attempt in task.get("attempts", []):
                ev = attempt.get("eval", {})
                if ev.get("correct"):
                    continue
                source_path = Path(attempt["source_path"])
                if not source_path.exists():
                    continue
                original = source_path.read_text()
                repaired, actions = repair_source(original)
                if not actions or repaired == original:
                    continue

                repair_index += 1
                task_dir = run_dir / task["task_id"]
                task_dir.mkdir(parents=True, exist_ok=True)
                repaired_path = task_dir / f"repair_{repair_index:03d}_{source_run}_attempt_{attempt['attempt']:02d}.py"
                repaired_path.write_text(repaired)
                eval_result = evaluate_candidate(task["task_id"], repaired_path, eval_timeout)
                repair_attempt = {
                    "attempt": repair_index,
                    "prompt_path": attempt.get("prompt_path"),
                    "raw_response_path": attempt.get("raw_response_path"),
                    "source_path": str(repaired_path),
                    "eval": eval_result,
                    "repair": {
                        "source_run": source_run,
                        "source_attempt": attempt.get("attempt"),
                        "original_source_path": str(source_path),
                        "actions": actions,
                    },
                }
                per_task_attempts[task["task_id"]].append(repair_attempt)
                grouped[task["task_id"]]["attempts"].append(repair_attempt)
                partial = list(grouped.values())
                (run_dir / "results.partial.json").write_text(json.dumps(partial, indent=2))
                print(
                    json.dumps(
                        {
                            "task_id": task["task_id"],
                            "source_run": source_run,
                            "source_attempt": attempt.get("attempt"),
                            "correct": eval_result.get("correct"),
                            "failure_type": eval_result.get("failure_type"),
                            "speedup": eval_result.get("speedup"),
                            "actions": actions,
                        }
                    )
                )

    results = []
    for task in grouped.values():
        attempts = task["attempts"]
        correct_attempts = [attempt for attempt in attempts if attempt["eval"].get("correct")]
        best = None
        if correct_attempts:
            best = max(correct_attempts, key=lambda attempt: attempt["eval"].get("speedup") or 0)
        task.update(
            {
                "solved": bool(correct_attempts),
                "best_attempt": best["attempt"] if best else None,
                "best_speedup": best["eval"].get("speedup") if best else None,
            }
        )
        results.append(task)

    output = {"metadata": metadata, "summary": summarize(results), "results": results}
    (run_dir / "results.json").write_text(json.dumps(output, indent=2))
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--eval-timeout", type=int, default=180)
    args = parser.parse_args()

    output = repair_results(
        results_paths=args.results,
        run_dir=Path("cuda_kernel_lab/runs") / args.run_name,
        eval_timeout=args.eval_timeout,
    )
    print(json.dumps(output["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
