from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_text(path: str | None) -> str:
    if not path:
        return ""
    return Path(path).read_text()


def classify(
    eval_result: dict,
    repair: dict | None = None,
    normalization: dict | None = None,
) -> str:
    if repair:
        prefix = "mechanical_repair_"
    elif normalization:
        prefix = "extraction_normalized_"
    else:
        prefix = ""
    if eval_result.get("correct") and (eval_result.get("speedup") or 0) > 1.0:
        return f"{prefix}correct_and_faster"
    if eval_result.get("correct"):
        return f"{prefix}correct_but_slower"
    failure = eval_result.get("failure_type") or "unknown_failure"
    if failure == "compile_error":
        return f"{prefix}compile_fix_needed"
    if failure == "correctness_failure":
        return f"{prefix}correctness_fix_needed"
    if failure == "runtime_error":
        return f"{prefix}runtime_fix_needed"
    return f"{prefix}{failure}"


def build_records(results_path: Path) -> list[dict]:
    data = json.loads(results_path.read_text())
    records = []
    for task in data["results"]:
        for attempt in task["attempts"]:
            ev = attempt["eval"]
            prompt = load_text(attempt.get("prompt_path"))
            source = load_text(attempt.get("source_path"))
            raw = load_text(attempt.get("raw_response_path"))
            records.append(
                {
                    "run_name": data["metadata"]["run_name"],
                    "task_id": task["task_id"],
                    "task_name": task["task_name"],
                    "level": task["level"],
                    "backend": task["backend"],
                    "attempt": attempt["attempt"],
                    "label": classify(ev, attempt.get("repair"), attempt.get("normalization")),
                    "compiled": bool(ev.get("compiled")),
                    "correct": bool(ev.get("correct")),
                    "speedup": ev.get("speedup"),
                    "failure_type": ev.get("failure_type"),
                    "error": ev.get("error"),
                    "repair": attempt.get("repair"),
                    "normalization": attempt.get("normalization"),
                    "prompt": prompt,
                    "raw_response": raw,
                    "completion": source,
                }
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--successful-only", action="store_true")
    args = parser.parse_args()

    records = build_records(args.results)
    if args.successful_only:
        records = [record for record in records if record["correct"]]
    out = args.out or Path("cuda_kernel_lab/datasets") / f"{args.results.parent.name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    print(json.dumps({"out": str(out), "records": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
