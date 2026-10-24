"""Consistency of the nine-run long-horizon diagnostics with their tracked inputs.

Reads only ``results/local`` (the probe series written by ``scripts/long_horizon.py`` and the
stored rollout curves); needs neither the dataset, the checkpoints nor a GPU.
"""

import json

import numpy as np
import pytest

from mgn_vortex import diagnostics as dg
from mgn_vortex.config import REPO_ROOT

STUDY = REPO_ROOT / "results" / "local"
LH = STUDY / "long_horizon"
RUNS = [f"mp{k:02d}_seed{s}" for k in (5, 10, 15) for s in (0, 1, 2)]
SHEDDING = [0, 1, 2, 4, 5, 6, 9]

pytestmark = pytest.mark.skipif(not (LH / "summary.json").exists(), reason="long-horizon results are not present")


@pytest.fixture(scope="module")
def summary():
    return json.loads((LH / "summary.json").read_text())


@pytest.fixture(scope="module")
def ref():
    return dict(np.load(LH / "reference.npz"))


def probe(run):
    return dict(np.load(LH / f"{run}.npz"))


def test_all_nine_runs_regenerated_and_match_the_stored_curves(summary):
    assert summary["runs"] == RUNS and summary["runs_missing_probe_series"] == []
    for run in RUNS:
        meta = json.loads(str(np.load(LH / f"{run}.npz")["meta"]))
        assert meta["device"] == "cpu"
        assert max(meta["max_abs_difference_to_stored_curves"].values()) < 1e-4


def test_shedding_classification_matches_the_data_checks(summary):
    assert summary["shedding_trajectories"] == SHEDDING


def test_the_earlier_three_run_probe_result_is_reproduced(ref):
    """ROLLOUT_VERIFICATION.md section 4: band power ratios 0.000-0.050 (steps 100-299) and 0.000-0.036 (300-599)."""
    i = list(ref["probe_names"]).index(dg.MAIN_PROBE)
    cases = {"mp05_seed0": SHEDDING, "mp10_seed0": [0, 6], "mp15_seed1": [0, 6]}
    for run, trajs in cases.items():
        p = probe(run)
        for t in trajs:
            for (lo, hi), bound in (((100, 299), 0.0505), ((300, 599), 0.0365)):
                true = ref["probe"][t, lo - 1 : hi, i, 1].astype(np.float64)
                pred = p["probe"][t, lo - 1 : hi, i, 1].astype(np.float64)
                assert dg.band_power_ratio(pred, true, dg.peak_bin(true)) <= bound
    true = ref["probe"][6, 99:299, i, 1].astype(np.float64)
    assert dg.correlation(probe("mp05_seed0")["probe"][6, 99:299, i, 1].astype(np.float64), true) == pytest.approx(0.73, abs=0.01)


def test_no_run_preserves_shedding_after_step_100(summary):
    for run, rows in summary["per_trajectory"].items():
        for row in rows:
            if row["shedding"]:
                assert row["shedding_preserved_late_train"] == 0 and row["shedding_preserved_extrapolation"] == 0, (run, row["trajectory"])
                assert row["frequency_ratio_extrapolation"] < 0.7
                envelope = row["envelope_step100"]
                assert envelope is None or envelope < 0.5


def test_curve_metrics_recompute_from_the_stored_curves(summary):
    for run in RUNS:
        c = np.load(STUDY / run / "rollout_curves.npz")
        eu = c["rel_l2_velocity"].astype(np.float64)
        pers = c["persistence_rel_l2_velocity"].astype(np.float64)
        for row in summary["per_trajectory"][run]:
            t = row["trajectory"]
            assert row["E_u_mean_extrapolation"] == pytest.approx(eu[t, 299:599].mean(), rel=1e-9)
            above = np.where(eu[t, 1:] > pers[t, 1:])[0]
            assert row["t_E_u>persistence"] == (above[0] + 2 if above.size else 600)


def test_depth_aggregates_and_intervals(summary):
    m = summary["metrics"]["E_u_mean_extrapolation"]
    for k in ("5", "10", "15"):
        vals = [m["runs"][f"mp{int(k):02d}_seed{s}"] for s in (0, 1, 2)]
        e = m["by_depth"][k]
        assert e["mean"] == pytest.approx(np.mean(vals)) and e["sd"] == pytest.approx(np.std(vals, ddof=1))
        half = summary["t_interval_factor"] * e["sd"] / np.sqrt(3)
        assert e["ci95"] == pytest.approx([e["mean"] - half, e["mean"] + half])
    assert summary["t_interval_factor"] == pytest.approx(4.303, abs=1e-3)


def test_documented_headline_numbers(summary):
    """Values quoted in docs/LONG_HORIZON_DIAGNOSTICS.md."""
    M = summary["metrics"]
    by = lambda key, k: M[key]["by_depth"][str(k)]["mean"]  # noqa: E731
    assert [round(by("one_step_increment_error", k), 3) for k in (5, 10, 15)] == [0.581, 0.574, 0.550]
    assert [round(by("one_step_pressure_error", k), 3) for k in (5, 10, 15)] == [0.488, 0.445, 0.381]
    assert [round(by("E_u_mean_extrapolation", k), 2) for k in (5, 10, 15)] == [1.44, 1.02, 1.35]
    assert [round(by("t_envelope<0.5", k)) for k in (5, 10, 15)] == [53, 52, 47]
    assert M["one_step_pressure_error"]["slope_per_5_layers"]["ci95"][1] < 0  # the one interval clear of zero
    assert M["E_u_mean_extrapolation"]["slope_per_5_layers"]["ci95"][0] < 0 < M["E_u_mean_extrapolation"]["slope_per_5_layers"]["ci95"][1]
    rows = [r for v in summary["per_trajectory"].values() for r in v]
    assert all(not r["nonfinite"] for r in rows)
    assert np.mean([r["KE_ratio_h599"] > 1 for r in rows]) == pytest.approx(0.856, abs=1e-3)
    assert sum(r["KE_ratio_above_10"] for r in rows) == 12
