"""The loop over processor sizes and seeds, and the summary tables."""

import csv
import json
import statistics
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from . import engine, evaluation
from .config import steps_per_run
from .data import check_data
from .provenance import collect_metadata, resolve_device, write_json

RUN_FIELDS = [
    "run",
    "processor_size",
    "seed",
    "parameters",
    "steps",
    "final_train_loss",
    "final_valid_loss",
    "train_seconds",
    "optimisation_seconds",
    "steps_per_second",
    "peak_train_memory_mb",
]
EVAL_FIELDS = [
    "one_step_rel_l2_velocity",
    "one_step_rel_l2_velocity_increment",
    "one_step_rel_l2_pressure",
    "one_step_rmse_velocity",
    "one_step_rmse_pressure",
    "rollout_mean_rel_l2_velocity_window",
    "rollout_mean_rel_l2_pressure_window",
    "rollout_mean_lagmin_rel_l2_velocity_window",
    "rollout_mean_rel_l2_velocity_all",
    "rollout_mean_rel_l2_pressure_all",
    "trajectories_with_velocity_error_above_1",
    "trajectories_with_nonfinite_error",
    "rollout_ms_per_step",
]
HORIZON_METRICS = ["rel_l2_velocity", "rel_l2_pressure", "rel_l2_vorticity", "lagmin_rel_l2_velocity", "kinetic_energy_rel_error"]
NOT_AGGREGATED = {"run", "processor_size", "seed", "parameters", "steps"}


def output_dir(cfg: DictConfig, root: Path) -> Path:
    path = Path(cfg.output_dir)
    return path if path.is_absolute() else root / path


def summary_fields(cfg: DictConfig) -> list[str]:
    horizon = [f"rollout_{m}_h{h}" for h in cfg.rollout.horizons for m in HORIZON_METRICS]
    return RUN_FIELDS + EVAL_FIELDS + horizon


def train_study(cfg: DictConfig, root: Path, resume: bool = False) -> Path:
    """Train every (processor size, seed) run of the study. ``resume`` skips runs that already finished."""
    out_dir = output_dir(cfg, root)
    device = resolve_device(cfg.device)
    dataset = check_data(cfg)
    if not (resume and (out_dir / "study_metadata.json").exists()):
        meta = collect_metadata(device)
        meta.update(
            {
                "name": cfg.name,
                "steps_per_run": steps_per_run(cfg),
                "dataset": dataset,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
        )
        write_json(out_dir / "study_metadata.json", meta)
    for processor_size, seed in engine.study_runs(cfg):
        run_dir = out_dir / engine.run_name(processor_size, seed)
        if resume and (run_dir / "train_metrics.json").exists():
            print(f"[skip] {run_dir.name} already trained", flush=True)
            continue
        metrics = engine.train_run(cfg, processor_size, seed, run_dir, device)
        print(
            f"[train] {run_dir.name}: {metrics['parameters']} parameters, final train loss "
            f"{metrics['final_train_loss']:.3e}, {metrics['train_seconds']:.1f} s on {device}",
            flush=True,
        )
    return out_dir


def flatten(train: dict, ev: dict, cfg: DictConfig) -> dict:
    """One summary row from the training and evaluation records of a run."""
    row = {k: train[k] for k in RUN_FIELDS}
    one, mean = ev["one_step"]["window"], ev["rollout_time_mean"]
    for key in ("rel_l2_velocity", "rel_l2_velocity_increment", "rel_l2_pressure", "rmse_velocity", "rmse_pressure"):
        row[f"one_step_{key}"] = one[key]
    for span in ("window", "all"):
        row[f"rollout_mean_rel_l2_velocity_{span}"] = mean[span]["rel_l2_velocity"]["mean"]
        row[f"rollout_mean_rel_l2_pressure_{span}"] = mean[span]["rel_l2_pressure"]["mean"]
    row["rollout_mean_lagmin_rel_l2_velocity_window"] = mean["window"]["lagmin_rel_l2_velocity"]["mean"]
    for key in ("trajectories_with_velocity_error_above_1", "trajectories_with_nonfinite_error", "rollout_ms_per_step"):
        row[key] = ev[key]
    for h in cfg.rollout.horizons:
        for m in HORIZON_METRICS:
            row[f"rollout_{m}_h{h}"] = ev["rollout_horizons"][str(h)][m]["mean"]
    return row


def evaluate_study(cfg: DictConfig, root: Path) -> dict:
    """Evaluate every trained run from its checkpoint and write the summary files and plotting fields."""
    out_dir = output_dir(cfg, root)
    device = resolve_device(cfg.device)
    check_data(cfg)
    trained = {(k, s) for k, s in engine.study_runs(cfg) if (out_dir / engine.run_name(k, s) / "train_metrics.json").exists()}
    # only seeds that are trained for every processor size are evaluated, so that all sizes are compared on the same seeds
    seeds = [int(s) for s in cfg.study.seeds if all((int(k), int(s)) in trained for k in cfg.study.processor_sizes)]
    runs = [(k, s) for k, s in engine.study_runs(cfg) if s in seeds]
    if not runs:
        raise SystemExit(f"no seed of {out_dir} is trained for every processor size")
    missing = [engine.run_name(k, s) for k, s in engine.study_runs(cfg) if (k, s) not in runs]

    # the plotted trajectory is the most unsteady reference trajectory: chosen from the data, not from a model
    fluctuation = evaluation.reference_fluctuation(cfg, out_dir / engine.run_name(*runs[0]))
    field_trajectory = int(np.argmax(fluctuation))
    plot_seed = runs[0][1]

    rows, fields, persistence = [], {}, None
    for processor_size, seed in runs:
        run_dir = out_dir / engine.run_name(processor_size, seed)
        ev, curves, run_fields = evaluation.evaluate_run(
            cfg, processor_size, run_dir, device, field_trajectory if seed == plot_seed else None
        )
        write_json(run_dir / "eval_metrics.json", ev)
        np.savez_compressed(run_dir / "rollout_curves.npz", **{k: v.astype(np.float32) for k, v in curves.items()})
        train = json.loads((run_dir / "train_metrics.json").read_text())
        rows.append(flatten(train, ev, cfg))
        if persistence is None:
            persistence = curves["persistence_rel_l2_velocity"]
        if run_fields is not None:
            fields.setdefault("mesh_pos", run_fields["mesh_pos"])
            fields.setdefault("cells", run_fields["cells"])
            fields.setdefault("times", run_fields["times"])
            fields.setdefault("truth", run_fields["truth"])
            fields[f"prediction_mp{processor_size:02d}"] = run_fields["prediction"]
        print(
            f"[eval] {run_dir.name}: one-step velocity {rows[-1]['one_step_rel_l2_velocity']:.3e}, "
            f"rollout mean over the training window {rows[-1]['rollout_mean_rel_l2_velocity_window']:.3e}",
            flush=True,
        )

    names = summary_fields(cfg)
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "name": cfg.name,
        "steps_per_run": steps_per_run(cfg),
        "evaluation_device": str(device),
        "seeds": seeds,
        "missing_runs": missing,
        "test_trajectories": cfg.num_test_samples,
        "reference_fluctuation_ratio": fluctuation.tolist(),
        "field_trajectory": field_trajectory,
        "plot_seed": plot_seed,
        "persistence_mean_rel_l2_velocity_window": float(persistence[:, : cfg.num_training_time_steps - 1].mean()),
        "persistence_mean_rel_l2_velocity_all": float(persistence.mean()),
        "by_processor_size": aggregate(rows, names),
        "runs": rows,
    }
    write_json(out_dir / "summary.json", summary)
    np.savez_compressed(out_dir / "fields.npz", trajectory=field_trajectory, plot_seed=plot_seed, **fields)
    return summary


def aggregate(rows: list[dict], names: list[str]) -> list[dict]:
    """Mean and sample standard deviation over seeds for every processor size."""
    out = []
    for size in sorted({r["processor_size"] for r in rows}):
        group = [r for r in rows if r["processor_size"] == size]
        entry = {"processor_size": size, "parameters": group[0]["parameters"], "n_seeds": len(group)}
        for key in names:
            if key in NOT_AGGREGATED:
                continue
            values = [r[key] for r in group if r[key] is not None]
            entry[f"{key}_mean"] = statistics.fmean(values) if values else None
            entry[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else None
        out.append(entry)
    return out
