from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def cuda_home() -> Path:
    return repo_root() / "cuda_kernel_lab/toolchains/cuda-13.0-local/usr/local/cuda-13.0"


def configure_environment(build_dir: Path | None = None) -> dict[str, str | None]:
    """Configure this process for repo-local CUDA 13 native extension builds."""
    cuda = cuda_home()
    old = {
        "CUDA_HOME": os.environ.get("CUDA_HOME"),
        "CUDA_PATH": os.environ.get("CUDA_PATH"),
        "PATH": os.environ.get("PATH"),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "TORCH_EXTENSIONS_DIR": os.environ.get("TORCH_EXTENSIONS_DIR"),
        "MAX_JOBS": os.environ.get("MAX_JOBS"),
    }

    bin_dir = cuda / "bin"
    lib_dirs = [cuda / "targets/x86_64-linux/lib", cuda / "nvvm/lib64"]
    old_path = os.environ.get("PATH", "")
    old_ld = os.environ.get("LD_LIBRARY_PATH", "")

    os.environ["CUDA_HOME"] = str(cuda)
    os.environ["CUDA_PATH"] = str(cuda)
    os.environ["PATH"] = f"{bin_dir}:{old_path}" if old_path else str(bin_dir)
    os.environ["LD_LIBRARY_PATH"] = (
        ":".join(str(path) for path in lib_dirs) + (f":{old_ld}" if old_ld else "")
    )
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
    os.environ.setdefault("MAX_JOBS", "1")
    if build_dir is not None:
        build_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_EXTENSIONS_DIR"] = str(build_dir)
    return old


def restore_environment(old: dict[str, str | None]) -> None:
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def environment_summary() -> dict[str, str]:
    cuda = cuda_home()
    return {
        "CUDA_HOME": str(cuda),
        "nvcc": str(cuda / "bin/nvcc"),
        "TORCH_CUDA_ARCH_LIST": "12.0",
    }

