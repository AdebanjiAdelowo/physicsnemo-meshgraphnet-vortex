"""Measure training and rollout speed on this device and project the runtime of a study.

    python scripts/probe.py --config full --budget-hours 9

Trains every processor size of the config for a few hundred gradient steps on a
handful of trajectories (speed does not depend on the number of trajectories), times
a rollout, and extrapolates to all runs of the config. Nothing of the study is
started. With ``--budget-hours`` the exit code is 3 if the projection does not fit;
the config is never reduced.
"""

import argparse
import shutil
import time

import torch
from _bootstrap import ROOT
from omegaconf import OmegaConf

from mgn_vortex import engine
from mgn_vortex.config import load_config, steps_per_run
from mgn_vortex.data import build_dataset, check_data
from mgn_vortex.model import build_model
from mgn_vortex.provenance import collect_metadata, resolve_device, write_json
from mgn_vortex.rollout import rollout

PROBE_TRAJECTORIES = 4
# Allowance on top of the measured optimisation time: dataset construction, validation,
# checkpoints and the difference between a short probe and a long run.
SAFETY_FACTOR = 1.15


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=300, help="measured gradient steps per processor size")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--budget-hours", type=float, default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    check_data(cfg)
    device = resolve_device(cfg.device)
    out_dir = ROOT / "runs" / f"{cfg.name}_probe"
    small = OmegaConf.merge(cfg, {"num_training_samples": PROBE_TRAJECTORIES, "num_test_samples": 1})

    sizes = [int(k) for k in cfg.study.processor_sizes]
    n_seeds = len(cfg.study.seeds)
    steps = steps_per_run(cfg)
    rows, train_hours, eval_hours = [], 0.0, 0.0
    for size in sizes:
        run_dir = out_dir / engine.run_name(size, 0)
        row = engine.probe_speed(small, size, run_dir, device, args.steps, args.warmup)
        # rollout speed with an untrained model of the same size
        dataset = build_dataset(small, "test", run_dir)
        model = build_model(cfg, size).to(device).eval()
        engine.synchronize(device)
        tick = time.perf_counter()
        rollout(model, dataset, 0, device)
        engine.synchronize(device)
        row["rollout_seconds_per_trajectory"] = time.perf_counter() - tick
        row["projected_train_hours_per_run"] = SAFETY_FACTOR * steps / row["steps_per_second"] / 3600
        # per test trajectory: one rollout and one strided one-step pass, plus the metrics on the CPU
        row["projected_eval_hours_per_run"] = (
            cfg.num_test_samples * row["rollout_seconds_per_trajectory"] * (1 + 1 / cfg.rollout.one_step_stride) * 1.5 / 3600
        )
        train_hours += n_seeds * row["projected_train_hours_per_run"]
        eval_hours += n_seeds * row["projected_eval_hours_per_run"]
        rows.append(row)
        memory = f", peak memory {row['peak_train_memory_mb']:.0f} MB" if row["peak_train_memory_mb"] else ""
        print(
            f"processor size {size:>2}: {row['steps_per_second']:.1f} steps/s{memory}, "
            f"{row['projected_train_hours_per_run'] * 60:.0f} min per training run of {steps:,} steps",
            flush=True,
        )
    shutil.rmtree(out_dir, ignore_errors=True)

    per_seed = (train_hours + eval_hours) / n_seeds
    total = train_hours + eval_hours
    result = {
        "config": cfg.name,
        "environment": collect_metadata(device),
        "steps_per_run": steps,
        "runs": len(sizes) * n_seeds,
        "safety_factor": SAFETY_FACTOR,
        "per_processor_size": rows,
        "projected_hours_per_seed": per_seed,
        "projected_hours_total": total,
        "budget_hours": args.budget_hours,
        "fits": None if args.budget_hours is None else total <= args.budget_hours,
        "complete_seeds_within_budget": None if args.budget_hours is None else int(args.budget_hours // per_seed),
    }
    write_json(ROOT / "runs" / f"{cfg.name}_probe.json", result)
    print(f"PROJECTION for '{cfg.name}': {total:.2f} h for {result['runs']} runs ({per_seed:.2f} h per seed) on {device}.")
    if args.budget_hours is not None:
        if result["fits"]:
            print(f"FITS the budget of {args.budget_hours:.2f} h.")
        else:
            print(
                f"DOES NOT FIT the budget of {args.budget_hours:.2f} h. Nothing was reduced. "
                f"Complete seeds that would fit: {result['complete_seeds_within_budget']}."
            )
            raise SystemExit(3)


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
    main()
