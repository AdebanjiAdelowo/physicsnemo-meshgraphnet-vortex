"""Train and evaluate K=10 cutoffs of a matched-data trajectory for one seed."""

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from _bootstrap import ROOT
from long_horizon import wake_series
from long_horizon_analysis import curve_metrics, probe_metrics, shedding_trajectories, to_json
from mgn_vortex import diagnostics, engine, evaluation
from mgn_vortex.budget import matched_data, train_budget
from mgn_vortex.config import load_config
from mgn_vortex.provenance import collect_metadata, resolve_device, write_json


def evaluate_snapshot(cfg, run_dir, device, dataset_record):
    target = run_dir / "evaluation"
    with (run_dir / "budget_weights.pt").open("rb") as weights:
        checksum = hashlib.file_digest(weights, "sha256").hexdigest()
    if target.exists():
        prior = json.loads((target / "eval_metrics.json").read_text())
        if prior["checkpoint_sha256"] != checksum:
            raise ValueError("Evaluation checkpoint hash differs")
        print(f"{target}: evaluation already exists", flush=True)
        return
    provenance = json.loads((run_dir / "training_provenance.json").read_text())
    if provenance["dataset"] != dataset_record:
        raise ValueError("Evaluation data differ from the training manifest")

    wakes = []
    geometries = []

    def collect_wake(traj, result, dataset):
        graph, _, _ = dataset[traj * (dataset.num_steps - 1)]
        codes = graph.x[:, 2:6].argmax(dim=1).numpy()
        geo = diagnostics.wake_geometry(result["mesh_pos"], wall=codes == 3, interior=codes == 0)
        geometries.append(geo)
        wakes.append(wake_series(result, geo))

    started = time.perf_counter()
    engine.reset_peak_memory(device)
    metrics, curves, _ = evaluation.evaluate_run(
        cfg, cfg.processor_size, run_dir, device, on_rollout=collect_wake,
    )
    pred, ref = (
        {key.removeprefix(prefix): np.stack([row[key] for row in wakes])
         for key in wakes[0] if key.startswith(prefix)}
        for prefix in ("pred_", "true_")
    )
    ref["probe_names"] = np.asarray(list(diagnostics.PROBES))
    ref["line_nodes"] = np.stack([geo["line_nodes"] for geo in geometries])
    shedding = set(shedding_trajectories(ref))
    per_trajectory = []
    for traj in range(cfg.num_test_samples):
        row = curve_metrics(curves, traj)
        row.update(trajectory=traj, shedding_reference=traj in shedding)
        if not row["nonfinite"]:
            row.update(probe_metrics(pred, ref, traj, traj in shedding))
            row.pop("_sliding", None)
        per_trajectory.append(row)
    metrics["shedding_reference_count"] = len(shedding)
    metrics["shedding_preserved_counts"] = {
        window: sum(row.get(f"shedding_preserved_{window}", 0) for row in per_trajectory)
        for window in diagnostics.SPECTRAL_WINDOWS
    }
    metrics["checkpoint_sha256"] = checksum
    metrics["evaluation_seconds"] = time.perf_counter() - started
    metrics["peak_evaluation_memory_mb"] = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    metrics["environment"] = collect_metadata(device)
    metrics["environment"]["notebook_version"] = os.environ.get("MGN_NOTEBOOK_VERSION")

    with tempfile.TemporaryDirectory(dir=run_dir) as scratch:
        staged = Path(scratch) / "evaluation"
        staged.mkdir()
        write_json(staged / "eval_metrics.json", to_json(metrics))
        write_json(staged / "wake_metrics.json", to_json(per_trajectory))
        np.savez_compressed(staged / "rollout_curves.npz", **curves)
        np.savez_compressed(staged / "wake_series.npz",
                            **{f"pred_{key}": value for key, value in pred.items()},
                            **{f"ref_{key}": value for key, value in ref.items()})
        staged.rename(target)
    print(f"Evaluated {target}", flush=True)


def compare_seed(output, cutoffs, seed, processor_size):
    endpoints = {}
    for cutoff in cutoffs:
        folder = output / f"steps_{cutoff:06d}" / engine.run_name(processor_size, seed) / "evaluation"
        if not folder.is_dir():
            continue
        ev = json.loads((folder / "eval_metrics.json").read_text())
        wakes = json.loads((folder / "wake_metrics.json").read_text())
        shedding = [row for row in wakes if row["shedding_reference"]]
        loss_times = [row["t_envelope<0.5"] for row in shedding if row.get("t_envelope<0.5") is not None]
        endpoints[cutoff] = {
            "one_step_velocity_increment": ev["one_step"]["window"]["rel_l2_velocity_increment"],
            "one_step_pressure": ev["one_step"]["window"]["rel_l2_pressure"],
            "velocity_extrapolation_mean": float(np.asarray([row["E_u_mean_extrapolation"] for row in wakes], dtype=float).mean()),
            "energy_ratio_step_599": float(np.asarray([row["KE_ratio_h599"] for row in wakes], dtype=float).mean()),
            "shedding_preserved_extrapolation_fraction": (
                ev["shedding_preserved_counts"]["extrapolation"] / len(shedding) if shedding else None
            ),
            "shedding_reference_count": len(shedding),
            "amplitude_loss_median_step_censored_at_600": float(np.median(loss_times)) if loss_times else None,
            "amplitude_loss_trajectory_count": len(loss_times),
            "nonfinite_trajectories": ev["trajectories_with_nonfinite_error"],
            "rollout_horizons": ev["rollout_horizons"],
        }
    if not endpoints:
        return
    base = endpoints.get(cutoffs[0])
    differences = {}
    if base is not None:
        for cutoff, endpoint in endpoints.items():
            if cutoff != cutoffs[0]:
                differences[cutoff] = {
                    key: value - base[key] for key, value in endpoint.items()
                    if isinstance(value, float) and isinstance(base[key], float)
                }
    write_json(output / f"seed{seed}_comparison.json", to_json({
        "seed": seed, "processor_size": processor_size, "by_cutoff": endpoints,
        "differences_from_baseline": differences,
    }))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "training_budget")
    parser.add_argument("--stop-at", type=int, default=None, help="optional step limit for this invocation")
    parser.add_argument("--train-hours", type=float, default=None, help="checkpoint and return at this wall-time limit")
    parser.add_argument("--mode", choices=("train", "evaluate"), required=True)
    parser.add_argument("--cutoff", type=int, choices=(11960, 59800, 119600))
    args = parser.parse_args()
    cfg = load_config("training_budget")
    device = resolve_device(cfg.device)
    if not torch.cuda.is_available():
        raise RuntimeError("The training-budget study requires CUDA")
    if args.train_hours is not None and args.train_hours <= 0:
        parser.error("--train-hours must be positive")
    if args.mode == "evaluate" and args.cutoff is None:
        parser.error("--mode evaluate requires --cutoff")

    dataset = matched_data(cfg)
    if args.mode == "train":
        deadline = None if args.train_hours is None else time.time() + 3600 * args.train_hours
        step = train_budget(cfg, args.output.resolve(), args.seed, device, dataset=dataset,
                            stop_at=args.stop_at, deadline=deadline)
        print(f"Saved through step {step}", flush=True)
    else:
        run_dir = args.output / f"steps_{args.cutoff:06d}" / engine.run_name(cfg.processor_size, args.seed)
        evaluate_snapshot(cfg, run_dir, device, dataset)
        compare_seed(args.output, list(cfg.budget.cutoffs), args.seed, cfg.processor_size)


if __name__ == "__main__":
    main()
