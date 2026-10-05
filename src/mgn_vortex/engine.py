# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from examples/cfd/vortex_shedding_mgn/train.py of NVIDIA PhysicsNeMo, commit
# b45a5c810c741e6b41f8515be24c51121f8fc21f (MGNTrainer and the epoch loop of main).
# Modified by Adebanji Adelowo (2026): single device instead of DistributedManager and
# DistributedDataParallel (devices cuda, mps or cpu), a seeded shuffling DataLoader in
# place of the DistributedSampler, no Weights & Biases logging, processor_size passed to
# the model, a fresh start for every run instead of resuming from ckpt_path, and added
# bookkeeping: loss history, validation loss, timings, memory and run metadata. The
# loss, optimiser, learning-rate schedule, AMP handling and update step are unchanged.

"""Training of one MeshGraphNet and the timing probe used for budget estimates."""

import csv
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from physicsnemo.utils import load_checkpoint, save_checkpoint
from torch.amp import GradScaler, autocast
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader as PyGDataLoader

from .config import steps_per_run
from .data import build_dataset, working_directory
from .model import build_model, count_parameters
from .provenance import write_json

HISTORY_FIELDS = ["step", "epoch", "lr", "train_loss", "valid_loss", "optimisation_seconds"]


def run_name(processor_size: int, seed: int) -> str:
    return f"mp{processor_size:02d}_seed{seed}"


def study_runs(cfg: DictConfig) -> list[tuple[int, int]]:
    """(processor size, seed) pairs, seed-major: every size is trained for a seed before the next seed."""
    return [(int(k), int(s)) for s in cfg.study.seeds for k in cfg.study.processor_sizes]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class MGNTrainer:
    def __init__(self, cfg: DictConfig, processor_size: int, seed: int, run_dir: Path, device: torch.device):
        self.device = device
        self.amp = cfg.amp

        # the dataset draws its training noise from the global generator
        torch.manual_seed(seed)
        np.random.seed(seed)

        # instantiate dataset
        self.dataset = build_dataset(cfg, "train", run_dir)

        # instantiate dataloader
        self.dataloader = PyGDataLoader(
            self.dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(seed),
            pin_memory=device.type == "cuda",
            num_workers=cfg.num_dataloader_workers,
        )

        # instantiate the model
        torch.manual_seed(seed)
        self.model = build_model(cfg, processor_size)
        if cfg.jit:
            if not self.model.meta.jit:
                raise ValueError("MeshGraphNet is not yet compatible with torch.compile.")
            self.model = torch.compile(self.model).to(device)
        else:
            self.model = self.model.to(device)

        # enable train mode
        self.model.train()

        # instantiate loss, optimizer, and scheduler
        self.criterion = torch.nn.MSELoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.lr,
            fused=device.type == "cuda",
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: cfg.lr_decay_rate**epoch
        )
        self.scaler = GradScaler(device.type, enabled=self.amp)

    def train(self, graph):
        graph = graph.to(self.device)
        self.optimizer.zero_grad()
        loss = self.forward(graph)
        self.backward(loss)
        self.scheduler.step()
        return loss

    def forward(self, graph):
        # forward pass
        with autocast(device_type=self.device.type, enabled=self.amp):
            pred = self.model(graph.x, graph.edge_attr, graph)
            loss = self.criterion(pred, graph.y)
            return loss

    def backward(self, loss):
        # backward pass
        if self.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()


@torch.no_grad()
def validation_loss(model, dataset, indices, device: torch.device) -> float:
    """Mean squared error of the normalised targets on noise-free samples (one-step, reference inputs)."""
    model.eval()
    criterion = torch.nn.MSELoss()
    total = torch.zeros((), device=device)
    loader = PyGDataLoader(Subset(dataset, indices), batch_size=1, shuffle=False)
    for graph, _cells, _mask in loader:
        graph = graph.to(device)
        total += criterion(model(graph.x, graph.edge_attr, graph), graph.y)
    model.train()
    return float(total) / len(loader)


def train_run(cfg: DictConfig, processor_size: int, seed: int, run_dir: Path, device: torch.device) -> dict:
    """Train one model for ``cfg.epochs`` epochs and write history, checkpoints and metrics to ``run_dir``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / cfg.ckpt_path
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    trainer = MGNTrainer(cfg, processor_size, seed, run_dir, device)
    valid = build_dataset(cfg, "valid", run_dir)
    valid_indices = range(0, len(valid), cfg.validation.stride)
    data_seconds = time.perf_counter() - start

    steps_per_epoch = len(trainer.dataloader)
    total_steps = cfg.epochs * steps_per_epoch
    assert total_steps == steps_per_run(cfg)
    valid_steps = {
        e * steps_per_epoch + round(steps_per_epoch * (j + 1) / cfg.validation.per_epoch)
        for e in range(cfg.epochs)
        for j in range(cfg.validation.per_epoch)
    }

    rows, step, optimisation_seconds = [], 0, 0.0
    window = torch.zeros((), device=device)
    window_steps = 0
    last_valid = None
    synchronize(device)
    tick = time.perf_counter()
    for epoch in range(cfg.epochs):
        for graph in trainer.dataloader:
            loss = trainer.train(graph)
            window += loss.detach()
            window_steps += 1
            step += 1

            do_valid = step in valid_steps
            if do_valid or step % cfg.log_every == 0 or step == total_steps:
                synchronize(device)
                optimisation_seconds += time.perf_counter() - tick
                row = {
                    "step": step,
                    "epoch": epoch,
                    "lr": trainer.scheduler.get_last_lr()[0],
                    "train_loss": float(window) / window_steps,
                    "valid_loss": None,
                    "optimisation_seconds": optimisation_seconds,
                }
                if do_valid:
                    row["valid_loss"] = last_valid = validation_loss(trainer.model, valid, valid_indices, device)
                rows.append(row)
                print(
                    f"  [{run_dir.name}] step {step}/{total_steps} train {row['train_loss']:.4e}"
                    + (f" valid {last_valid:.4e}" if do_valid else "")
                    + f" ({step / optimisation_seconds:.1f} steps/s)",
                    flush=True,
                )
                window.zero_()
                window_steps = 0
                synchronize(device)
                tick = time.perf_counter()

        # save checkpoint
        with working_directory(run_dir):
            save_checkpoint(
                cfg.ckpt_path,
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch,
            )
        synchronize(device)
        tick = time.perf_counter()

    with open(run_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    metrics = {
        "run": run_dir.name,
        "processor_size": processor_size,
        "seed": seed,
        "parameters": count_parameters(trainer.model),
        "epochs": cfg.epochs,
        "steps": total_steps,
        "training_trajectories": cfg.num_training_samples,
        "training_time_steps": cfg.num_training_time_steps,
        "final_train_loss": rows[-1]["train_loss"],
        "final_valid_loss": last_valid,
        "train_seconds": time.perf_counter() - start,
        "optimisation_seconds": optimisation_seconds,
        "steps_per_second": total_steps / optimisation_seconds,
        "data_seconds": data_seconds,
        "peak_train_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        "device": str(device),
    }
    write_json(run_dir / "train_metrics.json", metrics)
    return metrics


def load_trained_model(cfg: DictConfig, processor_size: int, run_dir: Path, device: torch.device):
    """The model of a finished run, from its last checkpoint, in evaluation mode."""
    model = build_model(cfg, processor_size).to(device)
    model.eval()
    with working_directory(run_dir):
        load_checkpoint(cfg.ckpt_path, models=model, device=device)
    return model


def probe_speed(cfg: DictConfig, processor_size: int, run_dir: Path, device: torch.device, steps: int, warmup: int) -> dict:
    """Training speed and memory of one model size, measured over ``steps`` gradient steps after ``warmup``."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    trainer = MGNTrainer(cfg, processor_size, 0, run_dir, device)
    count, tick = 0, None
    while count < warmup + steps:
        for graph in trainer.dataloader:
            if count == warmup:
                synchronize(device)
                tick = time.perf_counter()
            trainer.train(graph)
            count += 1
            if count == warmup + steps:
                break
    synchronize(device)
    seconds = time.perf_counter() - tick
    return {
        "processor_size": processor_size,
        "parameters": count_parameters(trainer.model),
        "measured_steps": steps,
        "steps_per_second": steps / seconds,
        "peak_train_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
    }
