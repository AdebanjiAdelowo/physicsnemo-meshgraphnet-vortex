import json

import numpy as np
import pytest
import torch

from mgn_vortex import diagnostics as dg

N = 300


def wave(period, amp=1.0, phase=0.0, n=N, trend=0.0):
    t = np.arange(n)
    return amp * np.sin(2 * np.pi * t / period + phase) + trend * t


def test_peak_frequency_is_accurate_between_bins():
    for period in (25.0, 37.3, 60.0):
        f = dg.peak_frequency(wave(period, trend=0.01))
        assert abs(1 / f - period) / period < 0.01


def test_band_power_ratio_identity_amplitude_and_no_shedding():
    ref = wave(40.0)
    k = dg.peak_bin(ref)
    assert dg.peak_bin(ref) == round(N / 40.0)
    assert dg.band_power_ratio(ref, ref, k) == pytest.approx(1.0)
    assert dg.band_power_ratio(0.5 * ref, ref, k) == pytest.approx(0.25)
    drift = 0.3 * np.linspace(0, 1, N) ** 2  # slow drift, no shedding
    assert dg.band_power_ratio(drift, ref, k) < 0.01
    assert dg.band_fraction(ref, k) > 0.95 and dg.band_fraction(drift, k) < 0.05


def test_phase_and_lag_recover_a_known_shift():
    ref = wave(40.0)
    shifted = wave(40.0, phase=np.pi / 2)
    assert dg.phase_at(shifted, ref, dg.peak_bin(ref)) == pytest.approx(90.0, abs=3.0)
    lag, c = dg.best_lag_correlation(shifted, ref, max_lag=40)
    # pred(t) = sin(w t + pi/2) = ref(t + 10): the best lag is +10 (or -30, one period earlier)
    assert lag in (10, -30) and c > 0.99
    assert dg.correlation(ref, ref) == pytest.approx(1.0)


def test_sliding_amplitude_and_correlation_detect_loss_of_shedding():
    ref = wave(40.0, n=600)
    decay = ref * np.where(np.arange(600) < 300, 1.0, 0.0)  # shedding stops at step 300
    amp = dg.sliding_amplitude_ratio(decay, ref)
    assert np.isnan(amp[: dg.SLIDING - 1]).all()
    t = dg.first_below(amp, 0.5, start=dg.SLIDING - 1)
    assert 300 <= t <= 300 + dg.SLIDING
    cor = dg.sliding_correlation(wave(40.0, phase=np.pi, n=600), ref)
    assert dg.first_below(cor, 0.5, start=dg.SLIDING - 1) == dg.SLIDING - 1


def test_threshold_helpers():
    e = np.array([0.1, 0.2, 0.6, 0.4, np.nan])
    assert dg.first_above(e, 0.5) == 2
    assert dg.first_above(e, 1.0) == 4  # non-finite counts as exceeded
    assert dg.first_above(np.array([0.1, 0.2]), 1.0) == -1
    assert dg.first_above(e, np.array([0.0, 0.3, 0.7, 0.3, 1.0]), start=1) == 3
    assert dg.first_below(np.array([np.nan, 0.9, 0.4]), 0.5) == 2


def test_log_growth_per_100():
    steps = np.arange(1, 600)
    assert dg.log_growth_per_100(np.exp(0.01 * steps), 100, 299) == pytest.approx(1.0)
    assert dg.log_growth_per_100(np.ones(599), 300, 599) == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(dg.log_growth_per_100(np.r_[np.ones(400), np.zeros(199)], 300, 599))


def test_window_unsteadiness():
    w = np.ones(4)
    steady = np.ones((10, 4, 2))
    assert dg.window_unsteadiness(steady, w, np.ones(4, bool)) == 0.0
    t = np.arange(10)[:, None, None]
    osc = np.ones((10, 4, 2)) + 0.1 * np.sin(t) * np.array([0, 1])
    assert 0 < dg.window_unsteadiness(osc, w, np.ones(4, bool)) < 0.1


def test_wake_geometry_finds_the_cylinder_and_probes():
    # channel 1.6 x 0.41 with a ring of wall nodes of diameter 0.1 centred at (0.2, 0.2)
    x, y = np.meshgrid(np.linspace(0, 1.6, 161), np.linspace(0, 0.41, 42))
    pos = np.stack([x.ravel(), y.ravel()], 1)
    ang = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    ring = np.stack([0.2 + 0.05 * np.cos(ang), 0.2 + 0.05 * np.sin(ang)], 1)
    keep = np.linalg.norm(pos - [0.2, 0.2], axis=1) > 0.05
    pos = np.concatenate([pos[keep], ring])
    wall = np.zeros(len(pos), bool)
    wall[-len(ring):] = True
    wall |= (pos[:, 1] == 0) | (pos[:, 1] == 0.41)
    g = dg.wake_geometry(pos, wall=wall, interior=~wall)
    assert np.allclose(g["centre"], [0.2, 0.2], atol=1e-9) and g["diameter"] == pytest.approx(0.1)
    main = list(dg.PROBES).index(dg.MAIN_PROBE)
    assert np.allclose(g["probe_pos"][main], [0.5, 0.2], atol=0.01)
    assert not wall[g["probe_nodes"]].any()
    beyond = np.asarray(dg.LINE_X_OVER_D) * 0.1 + 0.2 >= 1.6
    assert (g["line_nodes"][beyond] == -1).all() and (g["line_nodes"][~beyond] >= 0).all()


def test_long_horizon_script_regenerates_and_checks_the_stored_curves(cfg, tmp_path, monkeypatch):
    """End to end on the synthetic data: train, evaluate, regenerate, compare; a tampered curve is rejected."""
    import long_horizon

    from mgn_vortex.study import evaluate_study, train_study

    study = train_study(cfg, tmp_path)
    evaluate_study(cfg, tmp_path)

    def fixed_geometry(pos, wall, interior):  # the synthetic channel has no cylinder
        nodes = np.where(interior)[0]
        return {"centre": np.array([0.2, 0.2]), "diameter": 0.1, "probe_nodes": nodes[: len(dg.PROBES)],
                "probe_pos": pos[nodes[: len(dg.PROBES)]], "line_nodes": np.r_[nodes[:3], -np.ones(len(dg.LINE_X_OVER_D) - 3, int)],
                "line_pos": np.zeros((len(dg.LINE_X_OVER_D), 2))}

    monkeypatch.setattr(dg, "wake_geometry", fixed_geometry)
    monkeypatch.setattr(dg, "WINDOWS", {"a": (1, 5), "b": (6, 11)})
    torch.set_num_threads(1)
    run = sorted(p.name for p in study.glob("mp*_seed*"))[0]
    out = tmp_path / "lh"
    out.mkdir()
    long_horizon.process_run(cfg, run, study, out, tolerance=1e-4)
    z = np.load(out / f"{run}.npz")
    meta = json.loads(str(z["meta"]))
    assert max(meta["max_abs_difference_to_stored_curves"].values()) < 1e-5
    n_steps = cfg.num_test_time_steps - 1
    assert z["probe"].shape == (cfg.num_test_samples, n_steps, len(dg.PROBES), 3)
    assert z["line_rms_v"].shape == (cfg.num_test_samples, 2, len(dg.LINE_X_OVER_D))
    ref = np.load(out / "reference.npz")
    assert ref["probe"].shape == z["probe"].shape and str(ref["source_run"]) == run

    curves = dict(np.load(study / run / "rollout_curves.npz"))
    curves["rel_l2_velocity"] = curves["rel_l2_velocity"] + 1e-2
    np.savez(study / run / "rollout_curves.npz", **curves)
    (out / f"{run}.npz").unlink()
    with pytest.raises(RuntimeError, match="differ from rollout_curves"):
        long_horizon.process_run(cfg, run, study, out, tolerance=1e-4)
    assert not (out / f"{run}.npz").exists()


def test_shedding_preserved_needs_the_right_frequency_and_power():
    ref = wave(40.0)
    assert dg.shedding_preserved(ref, ref)
    assert dg.shedding_preserved(0.6 * wave(40.0, phase=1.0), ref)  # 36 % of the power, other phase
    assert not dg.shedding_preserved(0.3 * ref, ref)  # 9 % of the power
    assert not dg.shedding_preserved(wave(150.0), ref)  # slow oscillation, wrong frequency
    assert not dg.shedding_preserved(0.3 * np.linspace(0, 1, N) ** 2, ref)


def test_phase_tracking_recovers_phase_and_loss_of_shedding():
    n = 599
    for period in (26.0, 40.0, 57.0):  # the range of reference shedding periods
        ref = wave(period, n=n)
        shifted = wave(period, phase=np.pi / 3, n=n)
        track = dg.phase_tracking(shifted, ref, 1 / period)
        edge = int(round(period))
        assert np.isnan(track["envelope_ratio"][:edge]).all() and np.isnan(track["envelope_ratio"][-edge:]).all()
        inner = slice(edge, n - edge)
        assert np.allclose(track["envelope_ratio"][inner], 1.0, atol=0.1)
        assert np.allclose(track["phase_deg"][inner], 60.0, atol=5.0)
        half = dg.phase_tracking(0.5 * shifted, ref, 1 / period)["envelope_ratio"][inner]
        assert np.allclose(half, 0.5, atol=0.05)
        cut = np.where(np.arange(n) < 300, ref, 0.0)
        track = dg.phase_tracking(cut, ref, 1 / period)
        t = dg.first_below(track["envelope_ratio"], 0.5)
        assert 295 <= t <= 305
        assert np.isnan(track["phase_deg"][t + 20])
