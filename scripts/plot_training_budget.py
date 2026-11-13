"""Plot velocity error and persistence from verified training-budget rollout curves."""

import argparse
from pathlib import Path

import numpy as np
from _bootstrap import ROOT
from mgn_vortex.plotting import plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "results" / "training_budget")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    name = f"mp10_seed{args.seed}"
    paths = sorted(args.study.glob(f"steps_*/{name}/evaluation/rollout_curves.npz"))
    if not paths:
        parser.error(f"No evaluated budget cutoffs for {name}")

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    persistence = None
    for path in paths:
        cutoff = int(path.parents[2].name.split("_")[1])
        with np.load(path) as curves:
            error = curves["rel_l2_velocity"]
            baseline = curves["persistence_rel_l2_velocity"]
        if persistence is not None and not np.array_equal(persistence, baseline):
            raise ValueError("Persistence differs between evaluated checkpoints")
        persistence = baseline
        steps = np.arange(1, error.shape[1] + 1)
        ax.plot(steps, error.mean(axis=0), label=f"{cutoff:,} updates")
    ax.plot(steps, persistence.mean(axis=0), color="0.3", ls="--", label="Persistence")
    ax.axvspan(300, 599, color="0.3", alpha=0.08)
    ax.set(xlabel="Rollout step", ylabel="Mean relative velocity error",
           title=f"MeshGraphNet training budgets, seed {args.seed}", xlim=(1, 599))
    ax.legend()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
