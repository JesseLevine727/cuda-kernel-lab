from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_baseline(results_path: Path, out: Path) -> None:
    data = read_json(results_path)
    summary = data["summary"]
    lines = [
        "# Gemma 4 12B Q4 Kernel Baseline",
        "",
        f"Run: `{data['metadata']['run_name']}`",
        f"Model: `{data['metadata']['model']}`",
        "Benchmark scope: local reduced KernelBench-style Level 1/2 tasks using Triton backend.",
        "",
        "Q8 comparison: attempted, but not feasible in this live run because the temporary Q8 server exited during load. See `cuda_kernel_lab/reports/q8_attempt.json`.",
        "",
        "## Summary",
        "",
        f"- Tasks: {summary['tasks']}",
        f"- Solved: {summary['solved']}",
        f"- Correctness rate (`fast_0`): {summary['correctness_rate']:.3f}",
        f"- Faster-than-PyTorch rate (`fast_1`): {summary['fast_1']:.3f}",
        f"- Attempt compile rate: {summary['compiled_attempt_rate']:.3f}",
        f"- Average speedup among correct attempts: {summary['average_speedup_correct']}",
        f"- Best speedup: {summary['best_speedup']}",
        f"- Failure types: `{summary['failure_types']}`",
        "",
        "## Per Task",
        "",
    ]
    for result in data["results"]:
        lines.extend(
            [
                f"### {result['task_id']}",
                "",
                f"- Name: {result['task_name']}",
                f"- Backend: {result['backend']}",
                f"- Solved: {result['solved']}",
                f"- Best attempt: {result['best_attempt']}",
                f"- Best speedup: {result['best_speedup']}",
                f"- Attempts: {len(result['attempts'])}",
                "",
            ]
        )
        for attempt in result["attempts"]:
            ev = attempt["eval"]
            lines.extend(
                [
                    f"Attempt {attempt['attempt']}: compiled={ev.get('compiled')} correct={ev.get('correct')} "
                    f"failure={ev.get('failure_type')} speedup={ev.get('speedup')}",
                    "",
                ]
            )
    out.write_text("\n".join(lines))


def dataset_stats(dataset_path: Path) -> dict:
    stats = {"records": 0, "correct": 0, "labels": {}}
    if not dataset_path.exists():
        return stats
    with dataset_path.open() as fh:
        for line in fh:
            record = json.loads(line)
            stats["records"] += 1
            if record.get("correct"):
                stats["correct"] += 1
            label = record.get("label", "unknown")
            stats["labels"][label] = stats["labels"].get(label, 0) + 1
    return stats


def write_environment(env_path: Path, out: Path) -> None:
    env = read_json(env_path)
    torch = env.get("torch", {})
    nvidia = env.get("nvidia_smi", {})
    nvcc = env.get("nvcc", {})
    llama = env.get("llama_server_models", {})
    opencode = env.get("opencode_version", {})
    kernelbench = env.get("kernelbench_commit", {})
    lines = [
        "# Environment Report",
        "",
        "## Local GPU Runtime",
        "",
        f"- Python/platform: `{torch.get('python', '').splitlines()[0]}` / `{torch.get('platform')}`",
        f"- PyTorch: `{torch.get('torch')}`",
        f"- PyTorch CUDA: `{torch.get('torch_cuda')}`",
        f"- CUDA available: `{torch.get('cuda_available')}`",
        f"- GPU: `{torch.get('gpu_name')}`",
        f"- GPU capability: `{torch.get('gpu_capability')}`",
        f"- Torch arch list: `{torch.get('torch_arch_list')}`",
        f"- Triton: `{torch.get('triton')}`",
        "",
        "## Tooling",
        "",
        f"- nvcc: `{nvcc.get('stdout', '').strip().replace(chr(10), ' ')}`",
        f"- opencode: `{opencode.get('stdout', '').strip()}`",
        f"- KernelBench commit: `{kernelbench.get('stdout', '').strip()}`",
        f"- llama.cpp model endpoint: `{llama.get('stdout', '').strip()[:500]}`",
        "",
        "## Native CUDA Extension Status",
        "",
        "Native PyTorch CUDA extension evaluation is not the active backend for this run: "
        "the installed `nvcc` is CUDA 12.0 and fails on the RTX 5080 `compute_120` target. "
        "The local baseline therefore uses KernelBench-supported Triton kernels.",
        "",
        "Raw environment JSON: `cuda_kernel_lab/reports/environment.json`",
        "",
    ]
    out.write_text("\n".join(lines))


def write_feasibility(results_path: Path, env_path: Path, dataset_path: Path, out: Path) -> None:
    data = read_json(results_path)
    env = read_json(env_path)
    stats = dataset_stats(dataset_path)
    dataset_count = stats["records"]
    summary = data["summary"]
    nvcc_stdout = env.get("nvcc", {}).get("stdout", "").strip().replace("\n", " ")
    torch = env.get("torch", {})
    solved = summary["solved"]
    tasks = summary["tasks"]
    decision = (
        "Do not fine-tune yet. The immediate blocker is evaluation/tooling maturity, not enough curated data."
    )
    if solved >= max(2, tasks // 2) and dataset_count >= 50:
        decision = "Fine-tuning is justified as a next experiment, provided held-out tasks remain clean."

    lines = [
        "# Fine-Tuning Feasibility",
        "",
        f"Decision: **{decision}**",
        "",
        "## Evidence",
        "",
        f"- GPU: {torch.get('gpu_name')} capability {torch.get('gpu_capability')}",
        f"- PyTorch: {torch.get('torch')} CUDA {torch.get('torch_cuda')}",
        f"- Triton: {torch.get('triton')}",
        f"- nvcc: {nvcc_stdout}",
        f"- Baseline solved {solved}/{tasks} tasks.",
        f"- Dataset records created from eval loop: {dataset_count}",
        f"- Correct dataset records: {stats['correct']}",
        f"- Dataset labels: `{stats['labels']}`",
        "- Q8 comparison was attempted but not feasible in the live run; the temporary Q8 server exited during load and Q4 was restored.",
        "",
        "## Implications",
        "",
        "- The original HF safetensors checkpoint is required for LoRA/QLoRA training; the current GGUF is inference-only.",
        "- The local CUDA extension path is not ready because the installed nvcc cannot compile `compute_120` for the RTX 5080.",
        "- Triton gives a working local GPU-kernel backend and should be used to collect traces first.",
        "- A useful first fine-tune should wait until there are at least 50-200 clean CUDA/Triton examples with held-out eval tasks excluded.",
        "",
        "## Next Gate",
        "",
        "Collect more successful traces or install a CUDA 13 toolkit for native CUDA extension evaluation, then rerun the baseline before QLoRA.",
        "",
    ]
    out.write_text("\n".join(lines))


def write_final(results_path: Path, dataset_path: Path, out: Path) -> None:
    data = read_json(results_path)
    summary = data["summary"]
    stats = dataset_stats(dataset_path)
    lines = [
        "# Final Kernel Workflow Report",
        "",
        "This is the completed first local gate for the Gemma 4 12B CUDA-kernel fine-tuning project.",
        "",
        "## What Was Built",
        "",
        "- Official KernelBench cloned under `external/KernelBench`.",
        "- Local reduced KernelBench-style Triton harness under `cuda_kernel_lab/`.",
        "- Bounded Gemma generation loop with retry feedback.",
        "- Subprocess evaluator for compile/import, multi-seed correctness, timing, speedup, and failure labels.",
        "- JSONL trace dataset builder.",
        "- Environment, baseline, and fine-tuning feasibility reports.",
        "",
        "## Baseline Result",
        "",
        f"- Model: `{data['metadata']['model']}`",
        f"- Tasks: {summary['tasks']}",
        f"- Correctness rate (`fast_0`): {summary['correctness_rate']:.3f}",
        f"- Faster-than-PyTorch rate (`fast_1`): {summary['fast_1']:.3f}",
        f"- Correct traces available: {stats['correct']}",
        "- Q8 comparison: attempted but not feasible in this live run; Q4 was restored.",
        "",
        "## Fine-Tuning Decision",
        "",
        "Do not QLoRA fine-tune yet. The current run proves the local eval loop works, "
        "but it only produced a tiny number of correct examples and the native CUDA-extension backend is blocked by the CUDA 12.0 `nvcc` toolchain on compute capability 12.0.",
        "",
        "The next useful work is trace collection and/or a CUDA 13 toolkit install, not immediate training.",
        "",
        "## Before/After Status",
        "",
        "No tuned adapter was produced because the fine-tuning gate failed. The before/after comparison is therefore intentionally deferred; "
        "the baseline artifacts here are the reference point for any future tuned model.",
        "",
    ]
    out.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--env", type=Path, default=Path("cuda_kernel_lab/reports/environment.json"))
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()

    reports = Path("cuda_kernel_lab/reports")
    reports.mkdir(parents=True, exist_ok=True)
    write_environment(args.env, reports / "environment_report.md")
    write_baseline(args.results, reports / "baseline_report.md")
    write_feasibility(args.results, args.env, args.dataset, reports / "finetune_feasibility.md")
    write_final(args.results, args.dataset, reports / "final_report.md")
    print(
        json.dumps(
            {
                "reports": [
                    "environment_report.md",
                    "baseline_report.md",
                    "finetune_feasibility.md",
                    "final_report.md",
                ]
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
