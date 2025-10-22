from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class EnvReport:
    cuda_available: bool
    cuda_device_count: int
    nvidia_smi: bool
    docker_installed: bool
    docker_running: bool


def _check_cuda():
    try:
        import torch  # type: ignore
    except Exception:
        return False, 0
    available = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    count = torch.cuda.device_count() if available else 0
    return available, count


def _check_cmd(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _check_docker_running() -> bool:
    if not _check_cmd("docker"):
        return False
    try:
        subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3, check=True)
        return True
    except Exception:
        return False


def check_env() -> EnvReport:
    cuda_ok, cuda_count = _check_cuda()
    smi = _check_cmd("nvidia-smi")
    docker_ok = _check_cmd("docker")
    docker_run = _check_docker_running()
    return EnvReport(
        cuda_available=cuda_ok,
        cuda_device_count=cuda_count,
        nvidia_smi=smi,
        docker_installed=docker_ok,
        docker_running=docker_run,
    )

