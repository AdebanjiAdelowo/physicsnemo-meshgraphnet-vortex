"""Cross-check the recorded training-budget metrics against their numerical outputs."""

import json

import numpy as np

from mgn_vortex.config import REPO_ROOT


def test_budget_results_match_rollout_curves_and_wake_counts():
    root = REPO_ROOT / "results" / "training_budget"
    for file in root.glob("seed*_comparison.json"):
        comparison = json.loads(file.read_text())
        seed = comparison["seed"]
        reference = None
        manifests = []
        for cutoff, summary in comparison["by_cutoff"].items():
            run = root / f"steps_{int(cutoff):06d}" / f"mp10_seed{seed}"
            training = json.loads((run / "train_metrics.json").read_text())
            provenance = json.loads((run / "training_provenance.json").read_text())
            assert training["steps"] == int(cutoff)
            assert training["seed"] == provenance["seed"] == seed
            manifests.append(provenance["dataset"])
            evaluation = run / "evaluation"
            metrics = json.loads((evaluation / "eval_metrics.json").read_text())
            wake = json.loads((evaluation / "wake_metrics.json").read_text())
            shedding = [row for row in wake if row["shedding_reference"]]
            count = sum(row.get("shedding_preserved_extrapolation", 0) for row in shedding)
            assert count == metrics["shedding_preserved_counts"]["extrapolation"]
            assert count / len(shedding) == summary["shedding_preserved_extrapolation_fraction"]
            assert len(wake) == metrics["test_trajectories"] == 50
            with np.load(evaluation / "rollout_curves.npz") as curves:
                persistence = curves["persistence_rel_l2_velocity"]
                if reference is not None:
                    np.testing.assert_array_equal(reference, persistence)
                reference = persistence
                velocity = curves["rel_l2_velocity"]
                assert velocity.shape == (50, 599)
                np.testing.assert_allclose(velocity[:, 299:].mean(), summary["velocity_extrapolation_mean"])
                for step in (1, 10, 100, 299, 599):
                    horizon = metrics["rollout_horizons"][str(step)]
                    for key in ("rel_l2_velocity", "rel_l2_pressure", "rel_l2_vorticity", "persistence_rel_l2_velocity"):
                        np.testing.assert_allclose(curves[key][:, step - 1].mean(), horizon[key]["mean"])
        assert all(manifest == manifests[0] for manifest in manifests)
