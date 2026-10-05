"""Evaluation of a trained model: one-step errors, rollout errors and physical diagnostics."""

import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

from . import mesh_ops, metrics
from .data import build_dataset
from .engine import load_trained_model, synchronize
from .rollout import rollout

CURVES = [
    "rel_l2_velocity",
    "rel_l2_pressure",
    "rel_l2_vorticity",
    "rmse_velocity",
    "rmse_pressure",
    "lagmin_rel_l2_velocity",
    "best_lag",
    "kinetic_energy_pred",
    "kinetic_energy_true",
    "persistence_rel_l2_velocity",
]


def trajectory_metrics(result: dict, max_lag: int) -> dict:
    """Time series of every rollout metric for one trajectory. ``result`` comes from ``rollout``."""
    pos, cells = result["mesh_pos"], result["cells"]
    weights = mesh_ops.nodal_areas(pos, cells)
    dx, dy = mesh_ops.gradient_operators(pos, cells)
    pred, true = result["pred"], result["exact"]
    pred_u, true_u = pred[..., 0:2], true[..., 0:2]
    pred_p, true_p = pred[..., 2:3], true[..., 2:3]
    initial = np.broadcast_to(result["input_velocity"][0], true_u.shape)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        lagmin, best_lag = metrics.lag_minimised_error(pred_u, true_u, weights, max_lag)
        omega_pred = mesh_ops.vorticity(pred_u, dx, dy)[..., None]
        omega_true = mesh_ops.vorticity(true_u, dx, dy)[..., None]
        return {
            "rel_l2_velocity": metrics.relative_l2(pred_u, true_u, weights),
            "rel_l2_pressure": metrics.relative_l2(pred_p, true_p, weights),
            "rel_l2_vorticity": metrics.relative_l2(omega_pred, omega_true, weights),
            "rmse_velocity": metrics.nodal_rmse(pred_u, true_u),
            "rmse_pressure": metrics.nodal_rmse(pred_p, true_p),
            "lagmin_rel_l2_velocity": lagmin,
            "best_lag": best_lag.astype(np.float64),
            "kinetic_energy_pred": metrics.kinetic_energy(pred_u, weights),
            "kinetic_energy_true": metrics.kinetic_energy(true_u, weights),
            # error of holding the initial velocity for all times
            "persistence_rel_l2_velocity": metrics.relative_l2(initial, true_u, weights),
        }


def one_step_metrics(result: dict) -> dict:
    """Sums over the evaluated steps of one trajectory; pooled by the caller."""
    weights = mesh_ops.nodal_areas(result["mesh_pos"], result["cells"])
    pred, true = result["pred"], result["exact"]
    err_u = pred[..., 0:2] - true[..., 0:2]
    err_p = pred[..., 2:3] - true[..., 2:3]
    increment = true[..., 0:2] - result["input_velocity"]
    return {
        "steps": result["steps"],
        "rel_l2_velocity": metrics.relative_l2(pred[..., 0:2], true[..., 0:2], weights),
        "rel_l2_pressure": metrics.relative_l2(pred[..., 2:3], true[..., 2:3], weights),
        "rmse_velocity": metrics.nodal_rmse(pred[..., 0:2], true[..., 0:2]),
        "rmse_pressure": metrics.nodal_rmse(pred[..., 2:3], true[..., 2:3]),
        "error_sq": metrics.weighted_norm(err_u, weights) ** 2,
        "increment_sq": metrics.weighted_norm(increment, weights) ** 2,
        "pressure_error_sq": metrics.weighted_norm(err_p, weights) ** 2,
    }


def _stats(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": None, "median": None, "max": None, "n_finite": 0}
    return {"mean": float(finite.mean()), "median": float(np.median(finite)), "max": float(finite.max()), "n_finite": int(finite.size)}


def evaluate_run(cfg: DictConfig, processor_size: int, run_dir: Path, device: torch.device, field_trajectory: int | None = None):
    """Evaluate one trained run on the test trajectories.

    Returns ``(metrics, curves, fields)``: scalar metrics for ``eval_metrics.json``, per-trajectory
    time series of shape (trajectories, steps), and the fields of ``field_trajectory`` at
    ``cfg.rollout.field_times`` (``None`` if no trajectory was requested).
    """
    dataset = build_dataset(cfg, "test", run_dir)
    model = load_trained_model(cfg, processor_size, run_dir, device)
    n_traj = cfg.num_test_samples
    window = cfg.num_training_time_steps - 1  # last time level inside the training window

    curves = {name: [] for name in CURVES}
    one_step = []
    fields = None
    rollout_seconds = 0.0
    for traj in range(n_traj):
        synchronize(device)
        tick = time.perf_counter()
        result = rollout(model, dataset, traj, device, autoregressive=True)
        synchronize(device)
        rollout_seconds += time.perf_counter() - tick
        series = trajectory_metrics(result, cfg.rollout.max_lag)
        for name in CURVES:
            curves[name].append(series[name])
        one_step.append(one_step_metrics(rollout(model, dataset, traj, device, autoregressive=False, stride=cfg.rollout.one_step_stride)))
        if traj == field_trajectory:
            idx = [t - 1 for t in cfg.rollout.field_times]
            fields = {
                "mesh_pos": result["mesh_pos"],
                "cells": result["cells"],
                "times": np.asarray(list(cfg.rollout.field_times)),
                "truth": result["exact"][idx],
                "prediction": result["pred"][idx],
            }
    curves = {name: np.stack(values) for name, values in curves.items()}
    steps = curves["rel_l2_velocity"].shape[1]

    def pooled(mask_fn):
        num = sum(o["error_sq"][mask_fn(o["steps"])].sum() for o in one_step)
        den = sum(o["increment_sq"][mask_fn(o["steps"])].sum() for o in one_step)
        return float(np.sqrt(num / den))

    def one_step_mean(key, mask_fn):
        return float(np.mean(np.concatenate([o[key][mask_fn(o["steps"])] for o in one_step])))

    in_window = lambda s: s <= window  # noqa: E731
    everywhere = lambda s: s > 0  # noqa: E731
    one_step_summary = {}
    for label, mask_fn in (("window", in_window), ("all", everywhere)):
        one_step_summary[label] = {
            "rel_l2_velocity": one_step_mean("rel_l2_velocity", mask_fn),
            "rel_l2_velocity_increment": pooled(mask_fn),
            "rel_l2_pressure": one_step_mean("rel_l2_pressure", mask_fn),
            "rmse_velocity": one_step_mean("rmse_velocity", mask_fn),
            "rmse_pressure": one_step_mean("rmse_pressure", mask_fn),
        }

    horizons = {}
    for h in cfg.rollout.horizons:
        horizons[str(h)] = {
            name: _stats(curves[name][:, h - 1])
            for name in ("rel_l2_velocity", "rel_l2_pressure", "rel_l2_vorticity", "lagmin_rel_l2_velocity", "rmse_velocity", "rmse_pressure", "persistence_rel_l2_velocity")
        }
        ke_err = np.abs(curves["kinetic_energy_pred"][:, h - 1] - curves["kinetic_energy_true"][:, h - 1]) / curves["kinetic_energy_true"][:, h - 1]
        horizons[str(h)]["kinetic_energy_rel_error"] = _stats(ke_err)

    velocity = curves["rel_l2_velocity"]
    exceed = [metrics.first_exceedance(v, 1.0) for v in velocity]
    time_mean = {
        "window": {name: _stats(curves[name][:, :window].mean(axis=1)) for name in ("rel_l2_velocity", "rel_l2_pressure", "lagmin_rel_l2_velocity", "persistence_rel_l2_velocity")},
        "all": {name: _stats(curves[name].mean(axis=1)) for name in ("rel_l2_velocity", "rel_l2_pressure", "lagmin_rel_l2_velocity", "persistence_rel_l2_velocity")},
    }
    evaluation = {
        "run": run_dir.name,
        "processor_size": processor_size,
        "test_trajectories": n_traj,
        "rollout_steps": int(steps),
        "training_window_steps": int(window),
        "one_step_stride": int(cfg.rollout.one_step_stride),
        "one_step": one_step_summary,
        "rollout_horizons": horizons,
        "rollout_time_mean": time_mean,
        "trajectories_with_nonfinite_error": int(sum(not np.isfinite(v).all() for v in velocity)),
        "trajectories_with_velocity_error_above_1": int(sum(e >= 0 for e in exceed)),
        "first_step_with_velocity_error_above_1": [e + 1 if e >= 0 else None for e in exceed],
        "rollout_ms_per_step": 1000.0 * rollout_seconds / (n_traj * steps),
        "evaluation_device": str(device),
    }
    return evaluation, curves, fields


def reference_fluctuation(cfg: DictConfig, run_dir: Path) -> np.ndarray:
    """Unsteadiness of every test trajectory, from the reference data alone."""
    dataset = build_dataset(cfg, "test", run_dir)
    ratios = []
    for traj in range(cfg.num_test_samples):
        result = rollout(lambda x, e, g: g.y, dataset, traj, torch.device("cpu"), autoregressive=False)
        weights = mesh_ops.nodal_areas(result["mesh_pos"], result["cells"])
        ratios.append(metrics.fluctuation_ratio(result["exact"][..., 0:2], weights))
    return np.asarray(ratios)
