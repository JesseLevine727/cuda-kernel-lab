from __future__ import annotations

import json
import platform
import subprocess
import sys
import traceback
from pathlib import Path


def run(cmd: list[str]) -> dict:
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=30)
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except Exception as exc:
        return {"error": repr(exc)}


def probe_torch() -> dict:
    info = {"python": sys.version, "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_capability"] = torch.cuda.get_device_capability(0)
            info["torch_arch_list"] = torch.cuda.get_arch_list()
    except Exception:
        info["torch_error"] = traceback.format_exc(limit=5)
    try:
        import triton

        info["triton"] = triton.__version__
    except Exception:
        info["triton_error"] = traceback.format_exc(limit=5)
    return info


def main() -> int:
    info = {
        "torch": probe_torch(),
        "nvidia_smi": run(["nvidia-smi"]),
        "nvcc": run(["nvcc", "--version"]),
        "llama_server_models": run(["curl", "-sS", "http://127.0.0.1:8080/v1/models"]),
        "opencode_version": run(["opencode", "--version"]),
        "kernelbench_commit": run(["git", "-C", "external/KernelBench", "rev-parse", "HEAD"]),
    }
    out = Path("cuda_kernel_lab/reports/environment.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2))
    print(json.dumps({"out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

