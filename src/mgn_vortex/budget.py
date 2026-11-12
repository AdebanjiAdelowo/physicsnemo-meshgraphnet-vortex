"""Step-based checkpoints for a matched-data MeshGraphNet training trajectory."""

import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from itertools import islice
from pathlib import Path

import torch
from omegaconf import OmegaConf

from . import engine
from .config import steps_per_run
from .data import build_dataset, check_data, data_dir
from .provenance import collect_metadata, write_json


def matched_data(cfg):
    record = check_data(cfg)
    root = data_dir(cfg)

    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    record["meta_sha256"] = digest(root / "meta.json")
    for entry in record["splits"].values():
        actual = digest(root / entry["file"])
        if entry["file_sha256"] not in (None, actual):
            raise ValueError(f"Dataset hash mismatch for {entry['file']}")
        entry["file_sha256"] = actual
    return record


def atomic_save(path, state):
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as stream:
        temporary = stream.name
        try:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def publish_snapshot(root, cfg, seed, state, source):
    step = state["step"]
    name = engine.run_name(cfg.processor_size, seed)
    target = root / f"steps_{step:06d}" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = json.loads((target / "train_metrics.json").read_text())
        if existing != state["metrics"]:
            raise ValueError(f"Conflicting snapshot at {target}")
        return
    with tempfile.TemporaryDirectory(dir=target.parent) as scratch:
        staged = Path(scratch) / name
        staged.mkdir()
        for filename in ("edge_stats.json", "node_stats.json"):
            shutil.copyfile(source / filename, staged / filename)
        atomic_save(staged / "budget_weights.pt", state["model"])
        write_json(staged / "train_metrics.json", state["metrics"])
        write_json(staged / "training_provenance.json", {
            **state["identity"], "sessions": state["sessions"], "checkpoint": "budget_weights.pt",
        })
        with (staged / "history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=engine.HISTORY_FIELDS)
            writer.writeheader()
            writer.writerows(state["rows"])
        staged.rename(target)


def train_budget(cfg, root, seed, device, *, dataset, stop_at=None, deadline=None):
    cutoffs = list(cfg.budget.cutoffs)
    total = steps_per_run(cfg)
    target = total if stop_at is None else stop_at
    if not cutoffs or cutoffs != sorted(set(cutoffs)) or cutoffs[0] <= 0 or cutoffs[-1] != total:
        raise ValueError("Budget cutoffs must be strictly increasing and end at the configured step count")
    if not 0 < target <= total or cfg.budget.checkpoint_every <= 0 or cfg.log_every <= 0:
        raise ValueError("Invalid target step or checkpoint/logging interval")

    source = root / "training_state" / engine.run_name(cfg.processor_size, seed)
    source.mkdir(parents=True, exist_ok=True)
    with (source / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _train_locked(cfg, root, seed, device, dataset, target, deadline, source)


def _train_locked(cfg, root, seed, device, dataset, target, deadline, source):
    started = time.perf_counter()
    environment = collect_metadata(device)
    environment["notebook_version"] = os.environ.get("MGN_NOTEBOOK_VERSION")
    identity = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "dataset": dataset,
        "seed": seed,
        "environment": {key: environment[key] for key in (
            "git_commit", "device", "gpu_name", "cuda_version", "torch_version", "physicsnemo",
            "torch_geometric_version", "torch_scatter_version", "numpy_version", "torch_threads",
            "python_version", "tfrecord_version",
        )},
    }
    checkpoints = sorted(source.glob("step_??????.pt"))
    saved = torch.load(checkpoints[-1], map_location="cpu", weights_only=True) if checkpoints else None
    if saved is not None:
        if saved["identity"] != identity:
            raise ValueError("Resume configuration, dataset or software environment differs")
        if saved["step"] in cfg.budget.cutoffs:
            publish_snapshot(root, cfg, seed, saved, source)
        if saved["step"] >= target:
            return saved["step"]

    engine.reset_peak_memory(device)
    trainer = engine.MGNTrainer(cfg, cfg.processor_size, seed, source, device)
    valid = build_dataset(cfg, "valid", source)
    data_seconds = time.perf_counter() - started
    loader = trainer.dataloader
    per_epoch = len(loader)
    if per_epoch * cfg.epochs != steps_per_run(cfg):
        raise ValueError("Data loader step count differs from the configured budget")
    valid_indices = range(0, len(valid), cfg.validation.stride)
    valid_steps = {
        epoch * per_epoch + round(per_epoch * (i + 1) / cfg.validation.per_epoch)
        for epoch in range(cfg.epochs)
        for i in range(cfg.validation.per_epoch)
    } | set(cfg.budget.cutoffs)

    step = saved["step"] if saved else 0
    rows = saved["rows"] if saved else []
    sessions = [*(saved["sessions"] if saved else []), environment]
    prior_seconds = saved["metrics"]["train_seconds"] if saved else 0.0
    prior_data_seconds = saved["metrics"]["data_seconds"] if saved else 0.0
    optimisation_seconds = saved["metrics"]["optimisation_seconds"] if saved else 0.0
    last_valid = saved["metrics"]["final_valid_loss"] if saved else None
    loss_sum = torch.zeros((), device=device)
    window_steps = 0
    if saved:
        for key in ("model", "optimizer", "scheduler", "scaler"):
            getattr(trainer, key).load_state_dict(saved[key])
        torch.set_rng_state(saved["rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
        loader.generator.set_state(saved["loader_rng"])

    for epoch in range(step // per_epoch, cfg.epochs):
        epoch_rng = loader.generator.get_state()
        # Recreate the exact shuffled epoch and consume its already completed prefix.
        batches = islice(iter(loader), step % per_epoch, None)
        engine.synchronize(device)
        tick = time.perf_counter()
        for graph in batches:
            loss_sum += trainer.train(graph).detach()
            step += 1
            if step % per_epoch == 0:
                # RandomSampler can advance its generator again on exhaustion.
                next(batches, None)
            window_steps += 1
            do_valid = step in valid_steps
            due = step % cfg.budget.checkpoint_every == 0 or step in cfg.budget.cutoffs or step == target
            expired = deadline is not None and time.time() >= deadline
            if not (due or expired or do_valid or step % cfg.log_every == 0 or step % per_epoch == 0):
                continue
            engine.synchronize(device)
            optimisation_seconds += time.perf_counter() - tick
            train_loss = float(loss_sum) / window_steps
            if not math.isfinite(train_loss):
                raise RuntimeError(f"Nonfinite training loss at step {step}")
            if do_valid:
                last_valid = engine.validation_loss(trainer.model, valid, valid_indices, device)
                if not math.isfinite(last_valid):
                    raise RuntimeError(f"Nonfinite validation loss at step {step}")
            rows.append({
                "step": step, "epoch": epoch, "lr": trainer.scheduler.get_last_lr()[0],
                "train_loss": train_loss, "valid_loss": last_valid if do_valid else None,
                "optimisation_seconds": optimisation_seconds,
            })
            print(f"[seed {seed}] step {step}/{target} train {train_loss:.4e} valid {last_valid}", flush=True)
            loss_sum.zero_()
            window_steps = 0
            if due or expired:
                peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
                metrics = {
                    "run": source.name, "processor_size": cfg.processor_size, "seed": seed,
                    "parameters": engine.count_parameters(trainer.model), "steps": step,
                    "final_train_loss": train_loss, "final_valid_loss": last_valid,
                    "train_seconds": prior_seconds + time.perf_counter() - started,
                    "optimisation_seconds": optimisation_seconds, "steps_per_second": step / optimisation_seconds,
                    "data_seconds": prior_data_seconds + data_seconds,
                    "peak_train_memory_mb": max(peak, saved["metrics"]["peak_train_memory_mb"]) if saved and peak is not None else peak,
                    "device": str(device),
                }
                state = {
                    **{key: getattr(trainer, key).state_dict() for key in ("model", "optimizer", "scheduler", "scaler")},
                    "identity": identity, "sessions": sessions, "step": step, "rows": rows, "metrics": metrics,
                    "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                    "loader_rng": loader.generator.get_state() if step % per_epoch == 0 else epoch_rng,
                }
                atomic_save(source / f"step_{step:06d}.pt", state)
                if step in cfg.budget.cutoffs:
                    publish_snapshot(root, cfg, seed, state, source)
                if step == target or expired:
                    return step
            tick = time.perf_counter()
    return step
