from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cuda_kernel_lab.prompts import SYSTEM_PROMPT, build_prompt
from cuda_kernel_lab.tasks import TASKS


DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = "gemma-4-12b-it-q4_k_m"


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def chat_completion(
    base_url: str,
    model: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "top_p": 0.7,
        "max_tokens": max_tokens,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"API request failed: {exc}") from exc
    elapsed = time.perf_counter() - started
    parsed = json.loads(raw)
    content = parsed["choices"][0]["message"]["content"]
    meta = {
        "api_seconds": elapsed,
        "usage": parsed.get("usage", {}),
        "finish_reason": parsed["choices"][0].get("finish_reason"),
    }
    usage = meta["usage"]
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens:
        meta["completion_tokens_per_second"] = completion_tokens / elapsed
    return content, meta


def extract_code(text: str) -> str:
    matches = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return max(matches, key=len).strip()
    return text.strip()


def normalize_extracted_cuda_cpp(source: str) -> tuple[str, list[str]]:
    """Normalize common formatting damage in extracted CUDA modules.

    This is intentionally limited to syntax-level cleanup around code-block
    extraction and repeated one-character parenthesis mistakes. It records
    actions so normalized generations can be kept distinct from strict raw
    pass@N and from post-hoc mechanical repair datasets.
    """
    actions: list[str] = []
    normalized = source

    lines = []
    in_triple_string = False
    for line in normalized.splitlines():
        stripped = line.lstrip()
        if '"""' in line or "'''" in line:
            in_triple_string = not in_triple_string
        if (
            not in_triple_string
            and line != stripped
            and (
                stripped.startswith("ext = load_inline(")
                or stripped.startswith("ext=load_inline(")
                or stripped.startswith("class ModelNew(")
            )
        ):
            lines.append(stripped)
            actions.append("dedent top-level Python statement")
        else:
            lines.append(line)
    normalized = "\n".join(lines) + ("\n" if source.endswith("\n") else "")

    normalized2 = normalized.replace("\n ext = load_inline(", "\next = load_inline(")
    normalized2 = normalized2.replace("\n class ModelNew", "\nclass ModelNew")
    if normalized2 != normalized:
        normalized = normalized2
        actions.append("dedent literal top-level Python statement")

    normalized2 = re.sub(r"(dim3\s+dimGrid\([^;\n]+)\)\);", r"\1);", normalized)
    if normalized2 != normalized:
        normalized = normalized2
        actions.append("remove extra closing parenthesis from dimGrid constructor")

    new_lines = []
    changed_return = False
    for line in normalized.splitlines():
        stripped = line.strip()
        if stripped.startswith("return ext.") and stripped.endswith("))"):
            line = line.rstrip()[:-1]
            changed_return = True
            stripped = line.strip()
        if stripped.startswith("return ext."):
            missing_parens = stripped.count("(") - stripped.count(")")
            if 0 < missing_parens <= 3:
                line = line.rstrip() + (")" * missing_parens)
                changed_return = True
        new_lines.append(line)
    if changed_return:
        normalized = "\n".join(new_lines) + ("\n" if normalized.endswith("\n") else "")
        actions.append("normalize ext return parentheses")

    return normalized, sorted(set(actions))


def evaluate_candidate(task_id: str, source_path: Path, timeout: int, backend: str) -> dict[str, Any]:
    cmd = [
        "python3",
        "-m",
        "cuda_kernel_lab.kernel_eval",
        "--task-id",
        task_id,
        "--source",
        str(source_path),
        "--backend",
        backend,
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
            "elapsed_s": elapsed,
        }
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {
            "compiled": False,
            "correct": False,
            "failure_type": "evaluator_json_error",
            "error": proc.stdout + proc.stderr,
            "elapsed_s": elapsed,
        }
    result["subprocess_elapsed_s"] = elapsed
    return result


def run_task(
    task_id: str,
    run_dir: Path,
    base_url: str,
    model: str,
    max_attempts: int,
    max_tokens: int,
    temperature: float,
    api_timeout: int,
    eval_timeout: int,
    backend: str,
    normalize_extracted_source: bool,
) -> dict[str, Any]:
    task = TASKS[task_id]
    task_dir = run_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    feedback = None
    attempts = []

    for attempt_num in range(1, max_attempts + 1):
        prompt = build_prompt(task, feedback, backend=backend)
        prompt_path = task_dir / f"attempt_{attempt_num:02d}_prompt.txt"
        raw_path = task_dir / f"attempt_{attempt_num:02d}_raw.txt"
        source_path = task_dir / f"attempt_{attempt_num:02d}_candidate.py"
        prompt_path.write_text(prompt)

        attempt: dict[str, Any] = {
            "attempt": attempt_num,
            "prompt_path": str(prompt_path),
            "raw_response_path": str(raw_path),
            "source_path": str(source_path),
        }
        try:
            raw, api_meta = chat_completion(
                base_url=base_url,
                model=model,
                user_prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=api_timeout,
            )
            raw_path.write_text(raw)
            source = extract_code(raw)
            normalization_actions: list[str] = []
            if normalize_extracted_source and backend == "cuda_cpp":
                source, normalization_actions = normalize_extracted_cuda_cpp(source)
            source_path.write_text(source)
            attempt["api"] = api_meta
            if normalization_actions:
                attempt["normalization"] = {
                    "kind": "extraction_normalization",
                    "actions": normalization_actions,
                }
            eval_result = evaluate_candidate(task_id, source_path, eval_timeout, backend)
            attempt["eval"] = eval_result
        except Exception as exc:
            attempt["eval"] = {
                "compiled": False,
                "correct": False,
                "failure_type": "harness_error",
                "error": repr(exc),
            }
        attempts.append(attempt)

        eval_result = attempt["eval"]
        if eval_result.get("correct"):
            break
        feedback = (
            f"failure_type={eval_result.get('failure_type')}\n"
            f"error={str(eval_result.get('error'))[:3000]}"
        )

    best = None
    correct_attempts = [a for a in attempts if a["eval"].get("correct")]
    if correct_attempts:
        best = max(correct_attempts, key=lambda a: a["eval"].get("speedup") or 0)

    return {
        "task_id": task_id,
        "task_name": task.name,
        "level": task.level,
        "backend": backend,
        "attempts": attempts,
        "solved": bool(correct_attempts),
        "best_attempt": best["attempt"] if best else None,
        "best_speedup": best["eval"].get("speedup") if best else None,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    solved = sum(1 for r in results if r["solved"])
    compiled_attempts = 0
    total_attempts = 0
    failure_types: dict[str, int] = {}
    speedups = []
    for result in results:
        for attempt in result["attempts"]:
            total_attempts += 1
            ev = attempt["eval"]
            if ev.get("compiled"):
                compiled_attempts += 1
            failure = ev.get("failure_type")
            if failure:
                failure_types[failure] = failure_types.get(failure, 0) + 1
        if result.get("best_speedup") is not None:
            speedups.append(result["best_speedup"])
    return {
        "tasks": total,
        "solved": solved,
        "correctness_rate": solved / total if total else 0,
        "fast_1": sum(1 for s in speedups if s > 1.0) / total if total else 0,
        "total_attempts": total_attempts,
        "compiled_attempt_rate": compiled_attempts / total_attempts if total_attempts else 0,
        "failure_types": failure_types,
        "speedups": speedups,
        "average_speedup_correct": sum(speedups) / len(speedups) if speedups else None,
        "best_speedup": max(speedups) if speedups else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="affine_1d,leaky_relu_1d,fused_square_relu_1d,row_mean_2d")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--api-timeout", type=int, default=180)
    parser.add_argument("--eval-timeout", type=int, default=120)
    parser.add_argument("--backend", choices=("triton", "cuda_cpp"), default="triton")
    parser.add_argument("--normalize-extracted-source", action="store_true")
    args = parser.parse_args()

    task_ids = [task.strip() for task in args.tasks.split(",") if task.strip()]
    for task_id in task_ids:
        if task_id not in TASKS:
            raise SystemExit(f"Unknown task {task_id!r}")

    run_name = args.run_name or f"gemma_q4_triton_{now_stamp()}"
    run_dir = Path("cuda_kernel_lab/runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "run_name": run_name,
        "created_utc": now_stamp(),
        "base_url": args.base_url,
        "model": args.model,
        "max_attempts": args.max_attempts,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "tasks": task_ids,
        "backend": args.backend,
        "normalize_extracted_source": args.normalize_extracted_source,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    results = []
    for task_id in task_ids:
        result = run_task(
            task_id=task_id,
            run_dir=run_dir,
            base_url=args.base_url,
            model=args.model,
            max_attempts=args.max_attempts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            api_timeout=args.api_timeout,
            eval_timeout=args.eval_timeout,
            backend=args.backend,
            normalize_extracted_source=args.normalize_extracted_source,
        )
        results.append(result)
        (run_dir / "results.partial.json").write_text(json.dumps(results, indent=2))
        print(json.dumps({"task_id": task_id, "solved": result["solved"], "best_speedup": result["best_speedup"]}))

    summary = summarize(results)
    output = {"metadata": metadata, "summary": summary, "results": results}
    (run_dir / "results.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
