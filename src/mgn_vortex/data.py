"""Access to the cylinder-flow data through the upstream ``VortexSheddingDataset``.

The dataset class, graph construction, noise and normalisation are PhysicsNeMo's and
are used unchanged. This module only locates the files, checks that enough
trajectories were downloaded and runs the constructor in the run directory, where
the class writes and reads its ``edge_stats.json`` and ``node_stats.json``.
"""

import contextlib
import json
import os
import struct
from pathlib import Path

from omegaconf import DictConfig
from physicsnemo.datapipes.gnn.vortex_shedding_dataset import VortexSheddingDataset

from .config import REPO_ROOT

SPLIT_FILES = {"train": "train", "valid": "valid", "test": "test"}


def data_dir(cfg: DictConfig) -> Path:
    path = Path(cfg.data_dir)
    return path if path.is_absolute() else REPO_ROOT / path


def count_records(path: Path) -> int:
    """Number of complete TFRecord records (trajectories) in a file, from the record headers."""
    size, offset, records = path.stat().st_size, 0, 0
    with open(path, "rb") as f:
        while offset + 12 <= size:
            f.seek(offset)
            (length,) = struct.unpack("<Q", f.read(8))
            offset += 12 + length + 4
            if offset <= size:
                records += 1
    return records


def required_trajectories(cfg: DictConfig) -> dict:
    return {"train": cfg.num_training_samples, "valid": cfg.validation.num_samples, "test": cfg.num_test_samples}


def check_data(cfg: DictConfig) -> dict:
    """Verify that the files hold the trajectories the config asks for. Returns the dataset record."""
    root = data_dir(cfg)
    need = required_trajectories(cfg)
    hint = "python scripts/download_data.py " + " ".join(f"--{s} {n}" for s, n in need.items())
    if not (root / "meta.json").exists():
        raise FileNotFoundError(f"{root}/meta.json not found. Download the data first: {hint}")
    meta = json.loads((root / "meta.json").read_text())
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())["splits"] if manifest_path.exists() else {}
    record = {
        "name": "DeepMind MeshGraphNets cylinder_flow",
        "source": "https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/",
        "simulator": meta.get("simulator"),
        "dt": meta.get("dt"),
        "trajectory_length": meta.get("trajectory_length"),
        "splits": {},
    }
    for split, n in need.items():
        path = root / f"{SPLIT_FILES[split]}.tfrecord"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Download the data first: {hint}")
        available = count_records(path)
        if available < n:
            raise ValueError(f"{path} holds {available} trajectories, the config needs {n}. Run: {hint}")
        entry = manifest.get(split, {})
        same_file = entry.get("bytes") == path.stat().st_size
        record["splits"][split] = {
            "file": path.name,
            "trajectories_in_file": available,
            "trajectories_used": n,
            "selection": f"first {n} records of the file",
            "file_bytes": path.stat().st_size,
            "file_sha256": entry.get("sha256") if same_file else None,
            "complete_file": entry.get("complete_file") if same_file else None,
        }
    record["splits"]["train"]["time_steps_used"] = cfg.num_training_time_steps
    record["splits"]["valid"]["time_steps_used"] = cfg.num_training_time_steps
    record["splits"]["test"]["time_steps_used"] = cfg.num_test_time_steps
    return record


@contextlib.contextmanager
def working_directory(path: Path):
    """Run a block with ``path`` as the current directory (upstream uses Hydra's chdir for this)."""
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def build_dataset(cfg: DictConfig, split: str, run_dir: Path) -> VortexSheddingDataset:
    """Construct the upstream dataset for a split.

    The ``train`` split adds noise with the global torch generator and writes the
    normalisation statistics to ``run_dir``; the other splits read them from there,
    so the training set of a run has to be built first.
    """
    samples = required_trajectories(cfg)[split]
    steps = cfg.num_test_time_steps if split == "test" else cfg.num_training_time_steps
    with working_directory(run_dir):
        return VortexSheddingDataset(
            name=f"vortex_shedding_{split}",
            data_dir=str(data_dir(cfg)),
            split=SPLIT_FILES[split],
            num_samples=samples,
            num_steps=steps,
        )
