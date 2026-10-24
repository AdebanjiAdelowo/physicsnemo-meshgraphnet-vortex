"""Consistency of the published local study with its raw curves, and of the README claims with the data.

These tests read only the tracked files in ``results/local`` and need neither the dataset nor a GPU.
"""

import json
import statistics

import numpy as np
import pytest

from mgn_vortex.config import REPO_ROOT

STUDY = REPO_ROOT / "results" / "local"
SIZES, SEEDS = (5, 10, 15), (0, 1, 2)
HORIZONS = [1, 10, 50, 100, 200, 299, 450, 599]
WINDOW = 299

pytestmark = pytest.mark.skipif(not (STUDY / "summary.json").exists(), reason="results/local is not present")


def run(size, seed):
    path = STUDY / f"mp{size:02d}_seed{seed}"
    return {
        "curves": dict(np.load(path / "rollout_curves.npz")),
        "eval": json.loads((path / "eval_metrics.json").read_text()),
        "train": json.loads((path / "train_metrics.json").read_text()),
        "node_stats": (path / "node_stats.json").read_text(),
    }


@pytest.fixture(scope="module")
def runs():
    return {(k, s): run(k, s) for k in SIZES for s in SEEDS}


def seed_mean_curve(runs, size, name):
    return np.mean([runs[size, s]["curves"][name].astype(np.float64).mean(axis=0) for s in SEEDS], axis=0)


def test_design_nine_runs_equal_budget_paired_noise(runs):
    assert {r["train"]["steps"] for r in runs.values()} == {11960}
    assert {r["train"]["training_trajectories"] for r in runs.values()} == {40}
    assert {(k, r["train"]["parameters"]) for (k, _), r in runs.items()} == {(5, 845059), (10, 1588739), (15, 2332419)}
    for s in SEEDS:  # the same noise realisation, hence the same statistics, for every depth of a seed
        assert len({runs[k, s]["node_stats"] for k in SIZES}) == 1
    assert len({runs[5, s]["node_stats"] for s in SEEDS}) == 3


def test_eval_metrics_recompute_from_the_curves(runs):
    for r in runs.values():
        c, ev = r["curves"], r["eval"]
        assert c["rel_l2_velocity"].shape == (10, 599) and np.isfinite(c["rel_l2_velocity"]).all()
        for h in HORIZONS:
            for name in ("rel_l2_velocity", "rel_l2_pressure", "rel_l2_vorticity", "lagmin_rel_l2_velocity", "persistence_rel_l2_velocity"):
                assert np.isclose(c[name][:, h - 1].astype(np.float64).mean(), ev["rollout_horizons"][str(h)][name]["mean"], rtol=1e-4)
            ke = np.abs(c["kinetic_energy_pred"][:, h - 1] - c["kinetic_energy_true"][:, h - 1]) / c["kinetic_energy_true"][:, h - 1]
            assert np.isclose(ke.mean(), ev["rollout_horizons"][str(h)]["kinetic_energy_rel_error"]["mean"], rtol=1e-4)
        # time means: per trajectory over steps 1..299 (training window) or 1..599, then over trajectories
        for span, sl in (("window", slice(0, WINDOW)), ("all", slice(None))):
            mine = c["rel_l2_velocity"][:, sl].astype(np.float64).mean(axis=1).mean()
            assert np.isclose(mine, ev["rollout_time_mean"][span]["rel_l2_velocity"]["mean"], rtol=1e-4)
        assert int((c["rel_l2_velocity"] > 1).any(axis=1).sum()) == ev["trajectories_with_velocity_error_above_1"]


def test_seed_aggregates_in_summary(runs):
    summary = json.loads((STUDY / "summary.json").read_text())
    agg = {e["processor_size"]: e for e in summary["by_processor_size"]}
    for k in SIZES:
        for key, h in (("rollout_mean_rel_l2_velocity_window", None), ("rollout_rel_l2_velocity_h299", 299)):
            if h is None:
                values = [runs[k, s]["eval"]["rollout_time_mean"]["window"]["rel_l2_velocity"]["mean"] for s in SEEDS]
            else:
                values = [runs[k, s]["eval"]["rollout_horizons"][str(h)]["rel_l2_velocity"]["mean"] for s in SEEDS]
            assert np.isclose(statistics.fmean(values), agg[k][key + "_mean"])
            assert np.isclose(statistics.stdev(values), agg[k][key + "_std"])  # sample standard deviation over seeds


def test_readme_claims_on_the_persistence_baseline(runs):
    persistence = runs[5, 0]["curves"]["persistence_rel_l2_velocity"].astype(np.float64)
    baseline = persistence.mean(axis=0)
    crossings = {}
    for k in SIZES:
        above = seed_mean_curve(runs, k, "rel_l2_velocity") > baseline
        first = int(np.argmax(above))
        assert above[first:].all()  # once above the baseline, the mean error stays above it
        crossings[k] = first + 1
    assert crossings == {5: 86, 10: 100, 15: 78}
    # only the seed-0 runs with 5 and 10 steps have a lower time-mean error than the baseline over the training window
    below = {key for key, r in runs.items() if r["curves"]["rel_l2_velocity"][:, :WINDOW].mean() < persistence[:, :WINDOW].mean()}
    assert below == {(5, 0), (10, 0)}
    # but neither stays below it at every step of the window
    for k in (5, 10):
        assert (runs[k, 0]["curves"]["rel_l2_velocity"].astype(np.float64).mean(axis=0)[:WINDOW] > baseline[:WINDOW]).any()


def test_readme_claims_on_energy_and_phase(runs):
    # the seed-mean kinetic energy ratio at step 299 lies between 2 and 4 for every depth
    for k in SIZES:
        ratio = np.mean([(runs[k, s]["curves"]["kinetic_energy_pred"][:, 298] / runs[k, s]["curves"]["kinetic_energy_true"][:, 298]).mean() for s in SEEDS])
        assert 1.9 < ratio < 4.0
    # single trajectories gain far more energy: the largest ratio at step 599 exceeds 100
    worst = max((r["curves"]["kinetic_energy_pred"][:, 598] / r["curves"]["kinetic_energy_true"][:, 598]).max() for r in runs.values())
    assert worst > 100
    # the best time shift removes at most a few per cent of the error over the training window
    for k in SIZES:
        plain = np.mean([runs[k, s]["curves"]["rel_l2_velocity"][:, :WINDOW].mean() for s in SEEDS])
        shifted = np.mean([runs[k, s]["curves"]["lagmin_rel_l2_velocity"][:, :WINDOW].mean() for s in SEEDS])
        assert 0.0 < 1 - shifted / plain < 0.03


def test_readme_claim_on_the_pressure_error(runs):
    crossings = {k: int(np.argmax(seed_mean_curve(runs, k, "rel_l2_pressure") > 1)) + 1 for k in SIZES}
    assert crossings == {5: 145, 10: 109, 15: 167}
