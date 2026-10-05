"""Evaluate the checkpoints of a trained study: one-step errors, rollouts and diagnostics.

    python scripts/evaluate.py --config smoke

Writes ``eval_metrics.json`` and ``rollout_curves.npz`` per run and ``summary.csv``,
``summary.json`` and ``fields.npz`` to the study directory. Pass the same overrides that
were used for training. Runs of the config that are not trained yet are listed as missing.
"""

import argparse

from _bootstrap import ROOT
from mgn_vortex.config import load_config
from mgn_vortex.study import evaluate_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    summary = evaluate_study(load_config(args.config, args.overrides), ROOT)
    for entry in summary["by_processor_size"]:
        print(
            f"processor size {entry['processor_size']:>2}: {entry['parameters']:>8} parameters, one-step velocity "
            f"{entry['one_step_rel_l2_velocity_mean']:.3e}, rollout mean (training window) "
            f"{entry['rollout_mean_rel_l2_velocity_window_mean']:.3e} over {entry['n_seeds']} seed(s)"
        )
    if summary["missing_runs"]:
        print("not trained yet:", ", ".join(summary["missing_runs"]))


if __name__ == "__main__":
    main()
