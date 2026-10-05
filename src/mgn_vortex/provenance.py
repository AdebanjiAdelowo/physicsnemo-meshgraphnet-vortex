"""Run metadata: code version, environment and device."""

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import physicsnemo
import torch

from .config import REPO_ROOT

UPSTREAM_EXAMPLE = "examples/cfd/vortex_shedding_mgn"


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def physicsnemo_source() -> dict:
    """Version of the installed PhysicsNeMo and, for a git install, its commit."""
    info = {"version": physicsnemo.__version__, "commit": None, "url": None, "example": UPSTREAM_EXAMPLE}
    try:
        text = metadata.distribution("nvidia-physicsnemo").read_text("direct_url.json")
        if text:
            direct = json.loads(text)
            info["url"] = direct.get("url")
            info["commit"] = direct.get("vcs_info", {}).get("commit_id")
    except metadata.PackageNotFoundError:
        pass
    return info


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(name)


def collect_metadata(device: torch.device) -> dict:
    """Environment record stored with every study."""
    status = _git("status", "--porcelain")
    cuda = device.type == "cuda"
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if cuda else None,
        "gpu_memory_gb": round(torch.cuda.get_device_properties(device).total_memory / 2**30, 2) if cuda else None,
        "cuda_version": torch.version.cuda if cuda else None,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "physicsnemo": physicsnemo_source(),
        "torch_geometric_version": _version("torch_geometric"),
        "torch_scatter_version": _version("torch_scatter"),
        "tfrecord_version": _version("tfrecord"),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "torch_threads": torch.get_num_threads(),
    }


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")
