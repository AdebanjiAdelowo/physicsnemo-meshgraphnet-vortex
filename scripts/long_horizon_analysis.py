"""Long-horizon diagnostics of a study across depths and seeds, from tracked files only.

    python scripts/long_horizon_analysis.py --study results/local

Reads ``rollout_curves.npz`` and ``eval_metrics.json`` of every run and the probe and wake
series of ``long_horizon/`` written by ``scripts/long_horizon.py``. Writes
``long_horizon/summary.json`` (per trajectory, per run and per depth), ``long_horizon/tables.md``
and the ``figures/<study>_lh_*.png`` figures. Runs without probe series are reported as
missing for the spectral and wake diagnostics; nothing is filled in for them.

Statistics. The seed is the unit of replication. Run values are means (errors, ratios) or
medians (threshold times, censored at 600 = never) over trajectories. Per depth: mean, sample
SD and a 95 % t-interval over the 3 seeds (t = 4.30, 2 degrees of freedom). Depth effects:
within-seed paired differences and the within-seed least-squares slope per 5 layers, each with
a t-interval over seeds and the number of seeds with the same sign. A two-level bootstrap
(seeds, then trajectories) is reported as descriptive only: with 3 seeds it understates the
uncertainty. Associations between one-step and long-horizon metrics are Spearman correlations
over the 9 runs with an exact within-seed permutation p-value (6^3 = 216 relabellings).
"""

import argparse
import itertools
import json
from pathlib import Path

import matplotlib.ticker
import numpy as np
from _bootstrap import ROOT
from mgn_vortex import diagnostics as dg
from mgn_vortex.plotting import INK, MUTED, SERIES, plt  # noqa: F401  (rcParams of the repository)
from scipy import stats

DEPTHS, SEEDS = (5, 10, 15), (0, 1, 2)
T95 = float(stats.t.ppf(0.975, df=len(SEEDS) - 1))
NEVER = 600  # censoring value of a threshold time that is never reached within 599 steps
E_U_THRESHOLDS = (0.1, 0.25, 0.5, 1.0)
E_P_THRESHOLDS = (0.5, 1.0)
KE_BAND = 1.5  # kinetic-energy ratio outside [1/1.5, 1.5]
AMPLITUDE_LOSS, DECORRELATION = 0.5, 0.5
RNG = np.random.default_rng(0)


def run_name(k, s):
    return f"mp{k:02d}_seed{s}"


def time_of(index):
    return NEVER if index < 0 else index + 1


def load(study: Path):
    runs = {}
    for k, s in itertools.product(DEPTHS, SEEDS):
        name = run_name(k, s)
        d = study / name
        runs[name] = {
            "K": k, "seed": s,
            "curves": {key: v.astype(np.float64) for key, v in np.load(d / "rollout_curves.npz").items()},
            "eval": json.loads((d / "eval_metrics.json").read_text()),
            "probe": None,
        }
        p = study / "long_horizon" / f"{name}.npz"
        if p.exists():
            z = np.load(p)
            runs[name]["probe"] = {key: z[key] for key in z.files if key != "meta"}
            runs[name]["probe_meta"] = json.loads(str(z["meta"]))
    ref_path = study / "long_horizon" / "reference.npz"
    ref = {key: v for key, v in np.load(ref_path).items()} if ref_path.exists() else None
    return runs, ref


def shedding_trajectories(ref) -> list[int]:
    """Trajectories whose reference probe (3 D downstream) fluctuates, as in audit/data_checks.py."""
    i = list(ref["probe_names"]).index(dg.MAIN_PROBE)
    v = ref["probe"][:, 299:, i, 1].astype(np.float64)
    return [int(t) for t in np.where(v.std(axis=1) > 1e-3)[0]]


def curve_metrics(c: dict, traj: int) -> dict:
    eu, ep = c["rel_l2_velocity"][traj], c["rel_l2_pressure"][traj]
    pers = c["persistence_rel_l2_velocity"][traj]
    ke = c["kinetic_energy_pred"][traj] / c["kinetic_energy_true"][traj]
    m = {}
    for w, (lo, hi) in dg.WINDOWS.items():
        m[f"E_u_mean_{w}"] = float(eu[lo - 1 : hi].mean())
        m[f"E_p_mean_{w}"] = float(ep[lo - 1 : hi].mean())
    for h in (1, 10, 50, 99, 100, 299, 599):
        m[f"E_u_h{h}"], m[f"E_p_h{h}"], m[f"KE_ratio_h{h}"] = float(eu[h - 1]), float(ep[h - 1]), float(ke[h - 1])
    for th in E_U_THRESHOLDS:
        m[f"t_E_u>{th:g}"] = time_of(dg.first_above(eu, th))
    for th in E_P_THRESHOLDS:
        m[f"t_E_p>{th:g}"] = time_of(dg.first_above(ep, th))
    m["t_E_u>persistence"] = time_of(dg.first_above(eu, pers, start=1))  # step 1 is below persistence in every run
    m[f"t_KE_outside_{KE_BAND:g}"] = time_of(dg.first_above(np.abs(np.log(ke)), np.log(KE_BAND)))
    m["t_KE>2"] = time_of(dg.first_above(ke, 2.0))
    m["KE_ratio_max"], m["KE_ratio_min"] = float(np.nanmax(ke)), float(np.nanmin(ke))
    m["KE_log_drift_per_100_steps_extrapolation"] = dg.log_growth_per_100(ke, 300, 599)
    m["E_u_log_growth_per_100_steps_late_train"] = dg.log_growth_per_100(eu, 100, 299)
    m["E_u_log_growth_per_100_steps_extrapolation"] = dg.log_growth_per_100(eu, 300, 599)
    m["nonfinite"] = bool(not np.all(np.isfinite(eu)))
    m["KE_ratio_above_10"] = bool(np.nanmax(ke) > 10)
    return m


def probe_metrics(pred: dict, ref: dict, traj: int, shedding: bool) -> dict:
    names = list(ref["probe_names"])
    i3 = names.index(dg.MAIN_PROBE)
    m = {}
    sp_ratio = pred["speed_max"][traj] / ref["speed_max"][traj]
    m["peak_speed_ratio_max"] = float(np.nanmax(sp_ratio))
    windows = list(dg.WINDOWS)
    for wi, w in enumerate(windows):
        m[f"wake_unsteadiness_ratio_{w}"] = float(pred["wake_unsteadiness"][traj, wi] / ref["wake_unsteadiness"][traj, wi])
        valid = ref["line_nodes"][traj] >= 0
        rms_p, rms_t = pred["line_rms_v"][traj, wi, valid], ref["line_rms_v"][traj, wi, valid]
        mu_p, mu_t = pred["line_mean_u"][traj, wi, valid], ref["line_mean_u"][traj, wi, valid]
        m[f"centreline_rms_v_ratio_{w}"] = float(rms_p.sum() / rms_t.sum())
        m[f"centreline_mean_u_rel_error_{w}"] = float(np.abs(mu_p - mu_t).sum() / np.abs(mu_t).sum())
    if not shedding:
        return m
    true3 = ref["probe"][traj, :, i3, 1].astype(np.float64)
    pred3 = pred["probe"][traj, :, i3, 1].astype(np.float64)
    period = int(round(1 / dg.peak_frequency(true3[99:])))  # reference shedding period, steps 100-599
    for w in dg.SPECTRAL_WINDOWS:
        lo, hi = dg.WINDOWS[w]
        t, p = true3[lo - 1 : hi], pred3[lo - 1 : hi]
        k = dg.peak_bin(t)
        f_ref, f_pred = dg.peak_frequency(t), dg.peak_frequency(p)
        bpr = dg.band_power_ratio(p, t, k)
        lag, lag_corr = dg.best_lag_correlation(p, t, max_lag=min(period, len(t) // 2))
        m.update({
            f"period_ref_{w}": 1 / f_ref,
            f"period_pred_{w}": 1 / f_pred,
            f"frequency_ratio_{w}": f_pred / f_ref,
            f"band_power_ratio_{w}": bpr,
            f"pred_band_fraction_{w}": dg.band_fraction(p, k),
            f"ref_band_fraction_{w}": dg.band_fraction(t, k),
            f"rms_ratio_{w}": float(dg.detrend(p).std() / dg.detrend(t).std()),
            f"corr_{w}": dg.correlation(p, t),
            f"best_lag_corr_{w}": lag_corr,
            f"best_lag_steps_{w}": lag,
            f"phase_deg_{w}": dg.phase_at(p, t, k) if dg.shedding_preserved(p, t) else float("nan"),
        })
        # band power ratio at every probe along and across the wake
        for j, name in enumerate(names):
            tj, pj = ref["probe"][traj, lo - 1 : hi, j, 1].astype(np.float64), pred["probe"][traj, lo - 1 : hi, j, 1].astype(np.float64)
            m[f"band_power_ratio_{w}_{name}"] = dg.band_power_ratio(pj, tj, dg.peak_bin(tj))
        # pressure at the main probe
        tp, pp = ref["probe"][traj, lo - 1 : hi, i3, 2].astype(np.float64), pred["probe"][traj, lo - 1 : hi, i3, 2].astype(np.float64)
        m[f"pressure_band_power_ratio_{w}"] = dg.band_power_ratio(pp, tp, dg.peak_bin(tp))
        m[f"shedding_preserved_{w}"] = float(dg.shedding_preserved(p, t))
    # phase and in-band amplitude over time (band-passed, Hilbert transform)
    m["reference_period"] = period
    track = dg.phase_tracking(pred3, true3, 1.0 / period)
    env, phase = track["envelope_ratio"], track["phase_deg"]
    m["t_envelope<0.5"] = time_of(dg.first_below(env, AMPLITUDE_LOSS))  # the first period is NaN (filter edge)
    first = int(np.argmax(np.isfinite(env)))
    m["lost_by_first_measurable_step"] = float(env[first] < AMPLITUDE_LOSS)  # left-censored loss time
    for h in (50, 100, 200):
        m[f"envelope_step{h}"] = float(env[h - 1])
        m[f"phase_deg_step{h}"] = float(phase[h - 1])
        m[f"abs_phase_deg_step{h}"] = abs(m[f"phase_deg_step{h}"])
    cor = dg.sliding_correlation(pred3, true3, period)
    m["t_corr_zero_lag<0.5"] = time_of(dg.first_below(cor, DECORRELATION, start=period - 1))
    m["_sliding"] = {"envelope_ratio": env, "phase_deg": phase, "corr_zero": cor}
    return m


# ---------------------------------------------------------------- statistics

def seed_stats(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"mean": None, "sd": None, "ci95": None, "n": 0, "values": []}
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if v.size > 1 else None
    ci = [mean - T95 * sd / np.sqrt(v.size), mean + T95 * sd / np.sqrt(v.size)] if sd is not None and v.size == len(SEEDS) else None
    return {"mean": mean, "sd": sd, "ci95": ci, "n": int(v.size), "values": [float(x) for x in np.asarray(values, dtype=np.float64)]}


def paired(per_seed_a, per_seed_b) -> dict:
    d = np.asarray(per_seed_b) - np.asarray(per_seed_a)
    s = seed_stats(d)
    s["n_negative"], s["n_positive"] = int((d < 0).sum()), int((d > 0).sum())
    return s


def slope_per_5_layers(table) -> dict:
    """Within-seed least-squares slope of the run value against K, per 5 layers; t-interval over seeds."""
    slopes = [5 * np.polyfit(DEPTHS, [table[(k, s)] for k in DEPTHS], 1)[0] for s in SEEDS]
    out = seed_stats(slopes)
    out["n_negative"], out["n_positive"] = int(np.sum(np.asarray(slopes) < 0)), int(np.sum(np.asarray(slopes) > 0))
    return out


def bootstrap_depth_mean(per_traj: dict, k: int, reducer, n: int = 4000) -> list:
    """Two-level bootstrap: seeds with replacement, then trajectories within each. Descriptive only."""
    vals = []
    for _ in range(n):
        seeds = RNG.choice(SEEDS, len(SEEDS))
        runs = []
        for s in seeds:
            x = np.asarray(per_traj[(k, s)], dtype=np.float64)
            runs.append(reducer(RNG.choice(x, len(x))))
        vals.append(np.mean(runs))
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def within_seed_permutation_spearman(x: dict, y: dict) -> dict:
    """Spearman rho over the 9 runs; p from all 216 relabellings of x across depths within each seed."""
    keys = [(k, s) for s in SEEDS for k in DEPTHS]
    xs, ys = np.array([x[q] for q in keys]), np.array([y[q] for q in keys])
    if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
        return {"rho": None, "p_within_seed_permutation": None}
    rho = float(stats.spearmanr(xs, ys)[0])
    perms = list(itertools.permutations(range(len(DEPTHS))))
    null = []
    for combo in itertools.product(perms, repeat=len(SEEDS)):
        xp = np.concatenate([xs[3 * i : 3 * i + 3][list(p)] for i, p in enumerate(combo)])
        null.append(stats.spearmanr(xp, ys)[0])
    p = float(np.mean(np.abs(null) >= abs(rho) - 1e-12))
    return {"rho": rho, "p_within_seed_permutation": p}


# ---------------------------------------------------------------- main analysis

def analyse(study: Path) -> dict:
    runs, ref = load(study)
    missing = [n for n, r in runs.items() if r["probe"] is None]
    shed = shedding_trajectories(ref) if ref is not None else []
    n_traj = next(iter(runs.values()))["curves"]["rel_l2_velocity"].shape[0]
    steady = [t for t in range(n_traj) if t not in shed]
    per_traj = {}
    for name, r in runs.items():
        rows = []
        for t in range(n_traj):
            m = curve_metrics(r["curves"], t)
            if r["probe"] is not None:
                m.update(probe_metrics(r["probe"], ref, t, t in shed))
            m["trajectory"], m["shedding"] = t, t in shed
            rows.append(m)
        per_traj[name] = rows

    one_step = {(r["K"], r["seed"]): {
        "increment": r["eval"]["one_step"]["window"]["rel_l2_velocity_increment"],
        "velocity": r["eval"]["one_step"]["window"]["rel_l2_velocity"],
        "pressure": r["eval"]["one_step"]["window"]["rel_l2_pressure"],
    } for r in runs.values()}

    # metric -> (per-trajectory key, trajectory subset, reducer over trajectories, description)
    allt = list(range(n_traj))
    mean, median = np.mean, np.median
    metrics = {
        "one_step_increment_error": None, "one_step_velocity_error": None, "one_step_pressure_error": None,
        **{f"E_u_mean_{w}": (f"E_u_mean_{w}", allt, mean, f"velocity error, mean over steps {lo}-{hi}") for w, (lo, hi) in dg.WINDOWS.items()},
        "E_u_mean_extrapolation_shedding": ("E_u_mean_extrapolation", shed, mean, "velocity error, steps 300-599, shedding trajectories"),
        "E_u_mean_extrapolation_steady": ("E_u_mean_extrapolation", steady, mean, "velocity error, steps 300-599, steady trajectories"),
        "E_u_h599": ("E_u_h599", allt, mean, "velocity error at step 599"),
        **{f"E_p_mean_{w}": (f"E_p_mean_{w}", allt, mean, f"pressure error, mean over steps {lo}-{hi}") for w, (lo, hi) in dg.WINDOWS.items()},
        "E_p_h599": ("E_p_h599", allt, mean, "pressure error at step 599"),
        "KE_ratio_h99": ("KE_ratio_h99", allt, mean, "kinetic-energy ratio at step 99"),
        "KE_ratio_h299": ("KE_ratio_h299", allt, mean, "kinetic-energy ratio at step 299"),
        "KE_ratio_h599": ("KE_ratio_h599", allt, mean, "kinetic-energy ratio at step 599"),
        "KE_ratio_h599_median": ("KE_ratio_h599", allt, median, "kinetic-energy ratio at step 599 (median over trajectories)"),
        "KE_log_drift_per_100_steps_extrapolation": ("KE_log_drift_per_100_steps_extrapolation", allt, mean, "growth of log KE ratio per 100 steps, steps 300-599"),
        "E_u_log_growth_per_100_steps_late_train": ("E_u_log_growth_per_100_steps_late_train", allt, mean, "growth of log E_u per 100 steps, steps 100-299"),
        "E_u_log_growth_per_100_steps_extrapolation": ("E_u_log_growth_per_100_steps_extrapolation", allt, mean, "growth of log E_u per 100 steps, steps 300-599"),
        "t_E_u>persistence": ("t_E_u>persistence", allt, median, "first step above the persistence error (median)"),
        **{f"t_E_u>{th:g}": (f"t_E_u>{th:g}", allt, median, f"first step with E_u > {th:g} (median)") for th in E_U_THRESHOLDS},
        **{f"t_E_p>{th:g}": (f"t_E_p>{th:g}", allt, median, f"first step with E_p > {th:g} (median)") for th in E_P_THRESHOLDS},
        f"t_KE_outside_{KE_BAND:g}": (f"t_KE_outside_{KE_BAND:g}", allt, median, f"first step with KE ratio outside [1/{KE_BAND:g}, {KE_BAND:g}] (median)"),
        "t_KE>2": ("t_KE>2", allt, median, "first step with KE ratio > 2 (median)"),
    }
    if not missing and ref is not None:
        for w in dg.SPECTRAL_WINDOWS:
            lo, hi = dg.WINDOWS[w]
            metrics.update({
                f"shedding_preserved_{w}": (f"shedding_preserved_{w}", shed, mean, f"share of shedding trajectories with shedding preserved, steps {lo}-{hi}"),
                f"band_power_ratio_{w}": (f"band_power_ratio_{w}", shed, median, f"band power ratio, steps {lo}-{hi} (median)"),
                f"frequency_ratio_{w}": (f"frequency_ratio_{w}", shed, median, f"dominant frequency / reference, steps {lo}-{hi} (median)"),
                f"pred_band_fraction_{w}": (f"pred_band_fraction_{w}", shed, median, f"share of predicted fluctuation power in the band, steps {lo}-{hi} (median)"),
                f"rms_ratio_{w}": (f"rms_ratio_{w}", shed, median, f"probe RMS ratio, steps {lo}-{hi} (median)"),
                f"corr_{w}": (f"corr_{w}", shed, mean, f"zero-lag correlation, steps {lo}-{hi}"),
                f"best_lag_corr_{w}": (f"best_lag_corr_{w}", shed, mean, f"best-lag correlation (|lag| <= 1 period), steps {lo}-{hi}"),
                f"pressure_band_power_ratio_{w}": (f"pressure_band_power_ratio_{w}", shed, median, f"pressure band power ratio, steps {lo}-{hi} (median)"),
            })
        metrics.update({
            "t_corr_zero_lag<0.5": ("t_corr_zero_lag<0.5", shed, median, "first step the one-period zero-lag correlation falls below 0.5 (median)"),
            "t_envelope<0.5": ("t_envelope<0.5", shed, median, "first step the in-band amplitude ratio falls below 0.5: loss of shedding (median)"),
            **{f"abs_phase_deg_step{h}": (f"abs_phase_deg_step{h}", shed, lambda x: np.nanmean(x) if np.isfinite(x).any() else np.nan,
                                          f"|phase error| at step {h} (deg), trajectories still shedding") for h in (50, 100)},
            **{f"shedding_step{h}": (f"envelope_step{h}", shed, lambda x: np.mean(x[np.isfinite(x)] >= AMPLITUDE_LOSS) if np.isfinite(x).any() else np.nan,
                                     f"share still shedding (in-band amplitude >= 0.5) at step {h}, of trajectories measurable there") for h in (50, 100, 200)},
            "lost_by_first_measurable_step": ("lost_by_first_measurable_step", shed, mean, "share with in-band amplitude < 0.5 already one period into the rollout"),
            "peak_speed_ratio_max": ("peak_speed_ratio_max", allt, median, "largest ratio of peak nodal speed, prediction / reference (median)"),
            **{f"band_power_ratio_extrapolation_{q}": (f"band_power_ratio_extrapolation_{q}", shed, median, f"band power ratio, steps 300-599, probe {q} (median)") for q in dg.PROBES},
            **{f"wake_unsteadiness_ratio_{w}": (f"wake_unsteadiness_ratio_{w}", shed, mean, f"wake unsteadiness ratio, {w}, shedding trajectories") for w in dg.WINDOWS},
            **{f"centreline_rms_v_ratio_{w}": (f"centreline_rms_v_ratio_{w}", shed, mean, f"centreline RMS v' ratio, {w}, shedding trajectories") for w in dg.WINDOWS},
            **{f"centreline_mean_u_rel_error_{w}": (f"centreline_mean_u_rel_error_{w}", allt, mean, f"centreline mean-u relative error, {w}") for w in dg.WINDOWS},
        })

    def values(name, row_key, subset):
        return [row[row_key] for row in per_traj[name] if row["trajectory"] in subset and row_key in row]

    tables = {}
    for key, spec in metrics.items():
        if spec is None:
            sub = key.removeprefix("one_step_").removesuffix("_error")
            table = {q: one_step[q][sub] for q in one_step}
            desc = f"one-step {sub} error (training window{', pooled' if sub == 'increment' else ', mean'})"
        else:
            row_key, subset, reducer, desc = spec
            raw = {(runs[n]["K"], runs[n]["seed"]): values(n, row_key, subset) for n in runs}
            table = {q: float(reducer(np.asarray(v, dtype=np.float64))) if v else float("nan") for q, v in raw.items()}
        entry = {"description": desc, "runs": {run_name(*q): v for q, v in table.items()}, "by_depth": {}, "paired": {}}
        for k in DEPTHS:
            entry["by_depth"][str(k)] = seed_stats([table[(k, s)] for s in SEEDS])
            if spec is not None and all(len(raw[(k, s)]) for s in SEEDS):
                entry["by_depth"][str(k)]["bootstrap95_descriptive"] = bootstrap_depth_mean(raw, k, spec[2], n=1000)
        for a, b in ((5, 10), (5, 15), (10, 15)):
            entry["paired"][f"K{b}-K{a}"] = paired([table[(a, s)] for s in SEEDS], [table[(b, s)] for s in SEEDS])
        if all(np.isfinite(v) for v in table.values()):
            entry["slope_per_5_layers"] = slope_per_5_layers(table)
        tables[key] = {"table": table, **entry}

    assoc = {}
    for x in ("one_step_increment_error", "one_step_pressure_error"):
        for y in ("E_u_mean_late_train", "E_u_mean_extrapolation", "t_E_u>persistence", "KE_ratio_h299", "KE_ratio_h599_median",
                  "band_power_ratio_late_train", "t_envelope<0.5", "E_u_log_growth_per_100_steps_late_train"):
            if y in tables:
                assoc[f"{x} vs {y}"] = within_seed_permutation_spearman(tables[x]["table"], tables[y]["table"])

    sliding = {name: {row["trajectory"]: row.pop("_sliding") for row in rows if "_sliding" in row} for name, rows in per_traj.items()}
    consistency = {name: {"max_abs_difference_to_stored_curves": r.get("probe_meta", {}).get("max_abs_difference_to_stored_curves"),
                          "seconds": r.get("probe_meta", {}).get("seconds")} for name, r in runs.items()}
    summary = {
        "study": study.name,
        "runs": list(runs),
        "runs_missing_probe_series": missing,
        "shedding_trajectories": shed,
        "steady_trajectories": steady,
        "windows": dg.WINDOWS,
        "probes_in_diameters": dg.PROBES,
        "main_probe": dg.MAIN_PROBE,
        "band": dg.BAND,
        "sliding_window_steps": dg.SLIDING,
        "t_interval_factor": T95,
        "never_value": NEVER,
        "regeneration_check": consistency,
        "metrics": {k: {kk: vv for kk, vv in v.items() if kk != "table"} for k, v in tables.items()},
        "associations": assoc,
        "instability": {name: {"nonfinite_trajectories": int(sum(r["nonfinite"] for r in rows)),
                               "trajectories_KE_ratio_above_10": int(sum(r["KE_ratio_above_10"] for r in rows))}
                        for name, rows in per_traj.items()},
        "per_trajectory": per_traj,
    }
    return summary, runs, ref, sliding


# ---------------------------------------------------------------- output

def fmt(x, digits=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    if isinstance(x, (int, np.integer)) or (float(x).is_integer() and abs(x) >= 10):
        return f"{int(x)}"
    return f"{x:.{digits}g}" if abs(x) < 1e-2 else f"{x:.{digits}f}"


def depth_cell(e, digits=3):
    if e["mean"] is None:
        return "n/a"
    ci = e["ci95"]
    return f"{fmt(e['mean'], digits)} ± {fmt(e['sd'], digits)} [{fmt(ci[0], digits)}, {fmt(ci[1], digits)}]" if ci else fmt(e["mean"], digits)


def write_tables(summary: dict, path: Path):
    M = summary["metrics"]
    lines = [
        "# Long-horizon diagnostics, all nine runs",
        "",
        "Generated by `scripts/long_horizon_analysis.py` from tracked files. Values per depth: mean ± SD over 3 seeds "
        "[95 % t-interval, 2 degrees of freedom]. Threshold times are medians over trajectories, 600 = not reached in 599 steps. "
        f"Shedding trajectories: {summary['shedding_trajectories']}; steady: {summary['steady_trajectories']}.",
        "",
    ]
    groups = {
        "One-step accuracy (training window)": ["one_step_increment_error", "one_step_velocity_error", "one_step_pressure_error"],
        "Autoregressive error": ["E_u_mean_early", "E_u_mean_late_train", "E_u_mean_extrapolation", "E_u_mean_extrapolation_shedding",
                                 "E_u_mean_extrapolation_steady", "E_u_h599", "E_p_mean_early", "E_p_mean_late_train", "E_p_mean_extrapolation", "E_p_h599"],
        "Threshold times (steps)": ["t_E_u>0.1", "t_E_u>0.25", "t_E_u>0.5", "t_E_u>1", "t_E_u>persistence", "t_E_p>0.5", "t_E_p>1",
                                    "t_KE_outside_1.5", "t_KE>2", "t_corr_zero_lag<0.5", "t_envelope<0.5"],
        "Energy and stability": ["KE_ratio_h99", "KE_ratio_h299", "KE_ratio_h599", "KE_ratio_h599_median", "KE_log_drift_per_100_steps_extrapolation",
                                 "E_u_log_growth_per_100_steps_late_train", "E_u_log_growth_per_100_steps_extrapolation", "peak_speed_ratio_max"],
        "Vortex shedding at the 3 D probe (shedding trajectories)": [f"{q}_{w}" for q in ("shedding_preserved", "band_power_ratio", "frequency_ratio",
                                                                                              "pred_band_fraction", "rms_ratio", "corr", "best_lag_corr",
                                                                                              "pressure_band_power_ratio")
                                                                     for w in dg.SPECTRAL_WINDOWS],
        "Shedding persistence and phase (3 D probe, band-passed, Hilbert transform)": ["lost_by_first_measurable_step", "shedding_step50", "shedding_step100", "shedding_step200", "abs_phase_deg_step50", "abs_phase_deg_step100"],
        "Downstream wake (shedding trajectories unless stated)": [f"band_power_ratio_extrapolation_{q}" for q in dg.PROBES]
        + [f"{q}_{w}" for q in ("wake_unsteadiness_ratio", "centreline_rms_v_ratio", "centreline_mean_u_rel_error") for w in dg.WINDOWS],
    }
    for title, keys in groups.items():
        keys = [k for k in keys if k in M]
        if not keys:
            continue
        lines += [f"## {title}", "", "| Metric | K = 5 | K = 10 | K = 15 | slope per 5 layers [95 % CI] (seeds -/+) |", "|---|---|---|---|---|"]
        for k in keys:
            e = M[k]
            s = e.get("slope_per_5_layers")
            sl = f"{fmt(s['mean'])} [{fmt(s['ci95'][0])}, {fmt(s['ci95'][1])}] ({s['n_negative']}/{s['n_positive']})" if s and s["ci95"] else "n/a"
            lines.append(f"| {e['description']} (`{k}`) | " + " | ".join(depth_cell(e["by_depth"][str(K)]) for K in DEPTHS) + f" | {sl} |")
        lines.append("")
    lines += ["## Run-level values", "", "| Metric | " + " | ".join(summary["runs"]) + " |", "|---|" + "---|" * len(summary["runs"])]
    for k in ("one_step_increment_error", "E_u_mean_late_train", "E_u_mean_extrapolation", "t_E_u>persistence", "KE_ratio_h299",
              "KE_ratio_h599_median", "band_power_ratio_late_train", "band_power_ratio_extrapolation", "shedding_preserved_late_train",
              "t_corr_zero_lag<0.5", "t_envelope<0.5"):
        if k in M:
            lines.append(f"| `{k}` | " + " | ".join(fmt(M[k]["runs"][r]) for r in summary["runs"]) + " |")
    lines += ["", "## One-step accuracy against long-horizon behaviour", "",
              "Spearman correlation over the 9 runs; p from the 216 within-seed relabellings of depth (exact, small sample).", "",
              "| Pair | rho | p |", "|---|---|---|"]
    for k, v in summary["associations"].items():
        lines.append(f"| {k} | {fmt(v['rho'])} | {fmt(v['p_within_seed_permutation'])} |")
    lines += ["", "## Instability", "", "| Run | non-finite trajectories | trajectories with KE ratio > 10 |", "|---|---|---|"]
    for r, v in summary["instability"].items():
        lines.append(f"| {r} | {v['nonfinite_trajectories']} | {v['trajectories_KE_ratio_above_10']} |")
    lines += ["", "## Regeneration check", "",
              "Largest difference between the CPU regeneration and the stored MPS curves (absolute; relative for kinetic energy).", "",
              "| Run | E_u | E_p | E_omega | KE | lag-min E_u |", "|---|---|---|---|---|---|"]
    for r, v in summary["regeneration_check"].items():
        d = v["max_abs_difference_to_stored_curves"]
        lines.append(f"| {r} | " + (" | ".join(f"{d[k]:.1e}" for k in ("rel_l2_velocity", "rel_l2_pressure", "rel_l2_vorticity", "kinetic_energy_pred", "lagmin_rel_l2_velocity")) if d else "missing | | | |") + " |")
    path.write_text("\n".join(lines) + "\n")


def plot(summary, runs, ref, sliding, fig_dir: Path, prefix: str):
    shed = summary["shedding_trajectories"]
    steps = np.arange(1, 600)
    colour = {k: SERIES[i] for i, k in enumerate(DEPTHS)}

    # 1. errors and energy against time
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, key, title in ((axes[0], "rel_l2_velocity", "Velocity error $E_u$"), (axes[1], "rel_l2_pressure", "Pressure error $E_p$"),
                           (axes[2], "ke", "Kinetic energy, prediction / reference")):
        for k in DEPTHS:
            c, m, ls = colour[k]
            per_seed = []
            for s in SEEDS:
                cv = runs[run_name(k, s)]["curves"]
                y = cv["kinetic_energy_pred"] / cv["kinetic_energy_true"] if key == "ke" else cv[key]
                y = np.median(y, axis=0) if key == "ke" else y.mean(axis=0)
                per_seed.append(y)
                ax.plot(steps, y, color=c, lw=0.8, alpha=0.35, ls=ls)
            ax.plot(steps, np.mean(per_seed, axis=0), color=c, lw=2, ls=ls, label=f"K = {k}")
        if key == "rel_l2_velocity":
            ax.plot(steps, runs["mp05_seed0"]["curves"]["persistence_rel_l2_velocity"].mean(0), color=MUTED, lw=1.5, ls=":", label="persistence")
        if key == "ke":
            ax.set_yscale("log"); ax.axhline(1, color=MUTED, lw=1)
            ax.set_yticks([1, 2, 4, 8]); ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.axvline(299, color=MUTED, lw=0.8, ls="--")
        ax.set_title(title); ax.set_xlabel("rollout step")
    axes[2].text(0.02, 0.95, "median over trajectories", transform=axes[2].transAxes, fontsize=7, color=MUTED, va="top")
    axes[0].text(0.02, 0.95, "thin: seeds; thick: seed mean", transform=axes[0].transAxes, fontsize=7, color=MUTED, va="top")
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(fig_dir / f"{prefix}_lh_errors_energy.png", dpi=150); plt.close(fig)

    if ref is None or summary["runs_missing_probe_series"]:
        return
    i3 = list(ref["probe_names"]).index(dg.MAIN_PROBE)

    # 2. normalised spectra at the 3 D probe
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    grid = np.linspace(0.05, 3.0, 300)
    for ax, w in zip(axes, dg.SPECTRAL_WINDOWS):
        lo, hi = dg.WINDOWS[w]
        refs, preds = [], {k: [] for k in DEPTHS}
        for t in shed:
            true = ref["probe"][t, lo - 1 : hi, i3, 1].astype(np.float64)
            k_ref = dg.peak_bin(true)
            f = np.fft.rfftfreq(len(true)) / (k_ref / len(true))
            norm = dg.spectrum(true)[dg.band(k_ref)].sum()
            refs.append(np.interp(grid, f, dg.spectrum(true) / norm))
            for k in DEPTHS:
                for s in SEEDS:
                    pred = runs[run_name(k, s)]["probe"]["probe"][t, lo - 1 : hi, i3, 1].astype(np.float64)
                    preds[k].append(np.interp(grid, f, dg.spectrum(pred) / norm))
        ax.plot(grid, np.mean(refs, 0), color=INK, lw=2, label="reference")
        for k in DEPTHS:
            c, m, ls = colour[k]
            ax.plot(grid, np.mean(preds[k], 0), color=c, lw=1.6, ls=ls, label=f"K = {k}")
        ax.axvspan(*dg.BAND, color="#e4e3df", alpha=0.6, lw=0)
        ax.set_yscale("log"); ax.set_ylim(1e-4, None)
        ax.set_title(f"Probe 3 D downstream, steps {lo}-{hi}"); ax.set_xlabel("frequency / reference shedding frequency")
    axes[0].set_ylabel("power / reference band power")
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].text(0.98, 0.95, "shaded: band 0.7-1.3", transform=axes[1].transAxes, fontsize=7, color=MUTED, ha="right", va="top")
    fig.tight_layout(); fig.savefig(fig_dir / f"{prefix}_lh_spectra.png", dpi=150); plt.close(fig)

    # 3. probe traces of one shedding trajectory, every run
    t = 6 if 6 in shed else shed[0]
    fig, axes = plt.subplots(3, 1, figsize=(11, 6), sharex=True, sharey=True)
    for ax, k in zip(axes, DEPTHS):
        ax.plot(steps, ref["probe"][t, :, i3, 1], color=INK, lw=1.4, label="reference")
        c, m, ls = colour[k]
        for s, alpha in zip(SEEDS, (1.0, 0.6, 0.35)):
            ax.plot(steps, runs[run_name(k, s)]["probe"]["probe"][t, :, i3, 1], color=c, lw=1, alpha=alpha, ls=ls, label=f"seed {s}")
        ax.axvline(299, color=MUTED, lw=0.8, ls="--")
        ax.set_title(f"K = {k}", loc="left"); ax.set_ylabel("v at 3 D")
        ax.legend(fontsize=7, frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0.62, 1.0))
    axes[-1].set_xlabel(f"rollout step (test trajectory {t})")
    fig.tight_layout(); fig.savefig(fig_dir / f"{prefix}_lh_probe_traces.png", dpi=150); plt.close(fig)

    # 4. wake profiles along the centreline, extrapolation window
    wi = list(dg.WINDOWS).index("extrapolation")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    x = np.asarray(ref["line_x_over_d"], dtype=np.float64)
    for ax, key, title in ((axes[0], "line_rms_v", "RMS of v' on the centreline"), (axes[1], "line_mean_u", "Mean u on the centreline")):
        def prof(src):
            rows = []
            for tt in shed:
                y = src[key][tt, wi].astype(np.float64).copy()
                y[ref["line_nodes"][tt] < 0] = np.nan
                rows.append(y)
            return np.nanmean(rows, 0)
        ax.plot(x, prof(ref), color=INK, lw=2, label="reference")
        for k in DEPTHS:
            c, m, ls = colour[k]
            ax.plot(x, np.mean([prof(runs[run_name(k, s)]["probe"]) for s in SEEDS], 0), color=c, lw=1.6, ls=ls, marker=m, ms=3, markevery=8, label=f"K = {k}")
        ax.set_title(f"{title}, steps 300-599"); ax.set_xlabel("distance downstream of the cylinder centre / D")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(fig_dir / f"{prefix}_lh_wake_profiles.png", dpi=150); plt.close(fig)

    # 5. one-step accuracy against long-horizon behaviour
    M = summary["metrics"]
    pairs = (("E_u_mean_extrapolation", "velocity error, steps 300-599"), ("t_E_u>persistence", "first step above persistence (median)"),
             ("band_power_ratio_late_train", "band power ratio, steps 100-299"), ("KE_ratio_h299", "KE ratio at step 299"))
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    for ax, (key, label) in zip(axes, pairs):
        for k in DEPTHS:
            c, m, ls = colour[k]
            xs = [M["one_step_increment_error"]["runs"][run_name(k, s)] for s in SEEDS]
            ys = [M[key]["runs"][run_name(k, s)] for s in SEEDS]
            ax.scatter(xs, ys, color=c, marker=m, s=40, edgecolor="#fcfcfb", linewidth=1.5, label=f"K = {k}", zorder=3)
        for s in SEEDS:  # join the depths of one seed: the paired design
            xs = [M["one_step_increment_error"]["runs"][run_name(k, s)] for k in DEPTHS]
            ys = [M[key]["runs"][run_name(k, s)] for k in DEPTHS]
            ax.plot(xs, ys, color=MUTED, lw=0.7, zorder=1)
            ax.annotate(f"s{s}", (xs[-1], ys[-1]), fontsize=7, color=MUTED, xytext=(3, 0), textcoords="offset points")
        a = summary["associations"].get(f"one_step_increment_error vs {key}")
        stat = f"\nSpearman {a['rho']:+.2f}, within-seed p {a['p_within_seed_permutation']:.2f}" if a and a["rho"] is not None else ""
        ax.set_title(label + stat, fontsize=9); ax.set_xlabel("one-step increment error")
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(fig_dir / f"{prefix}_lh_onestep_vs_rollout.png", dpi=150); plt.close(fig)

    # 6. shedding persistence and phase: in-band amplitude ratio and phase error against time
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for k in DEPTHS:
        c, m, ls = colour[k]
        env = np.nanmedian([sliding[run_name(k, s)][t]["envelope_ratio"] for s in SEEDS for t in shed], axis=0)
        axes[0].plot(steps, env, color=c, lw=1.6, ls=ls, label=f"K = {k}")
        for s in SEEDS:
            for t in shed:
                ph = sliding[run_name(k, s)][t]["phase_deg"].copy()
                ph[1:][np.abs(np.diff(ph)) > 180] = np.nan  # do not join across a wrap of +-180 degrees
                axes[1].plot(steps, ph, color=c, lw=0.8, ls=ls, alpha=0.6, label=f"K = {k}" if (s, t) == (SEEDS[0], shed[0]) else None)
    axes[0].axhline(0.5, color=MUTED, lw=0.8, ls=":"); axes[0].set_yscale("log")
    axes[0].set_title("In-band amplitude, prediction / reference (median)")
    axes[1].axhline(90, color=MUTED, lw=0.8, ls=":"); axes[1].axhline(-90, color=MUTED, lw=0.8, ls=":")
    axes[1].set_title("Phase error while still shedding (per run and trajectory)")
    axes[1].set_ylabel("degrees")
    for ax in axes:
        ax.axvline(299, color=MUTED, lw=0.8, ls="--"); ax.set_xlabel("rollout step")
    axes[1].set_xlim(0, 150); axes[1].set_ylim(-180, 180); axes[1].set_yticks([-180, -90, 0, 90, 180])
    axes[0].text(0.98, 0.95, "3 D probe, shedding trajectories", transform=axes[0].transAxes, fontsize=7, color=MUTED, ha="right", va="top")
    axes[0].legend(fontsize=7, frameon=False, loc="lower left"); axes[1].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(fig_dir / f"{prefix}_lh_phase_tracking.png", dpi=150); plt.close(fig)

    # 7. threshold times by depth, each run a point
    keys = ["t_E_u>0.1", "t_E_u>0.25", "t_E_u>persistence", "t_E_u>0.5", "t_E_p>1", "t_KE_outside_1.5", "t_corr_zero_lag<0.5", "t_envelope<0.5"]
    keys = [k for k in keys if k in M]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for j, key in enumerate(keys):
        for i, k in enumerate(DEPTHS):
            c, m, ls = colour[k]
            ys = [M[key]["runs"][run_name(k, s)] for s in SEEDS]
            ax.scatter(np.full(3, j + (i - 1) * 0.22), ys, color=c, marker=m, s=30, edgecolor="#fcfcfb", linewidth=1.2, label=f"K = {k}" if j == 0 else None, zorder=3)
            ax.plot([j + (i - 1) * 0.22 - 0.08, j + (i - 1) * 0.22 + 0.08], [np.mean(ys)] * 2, color=c, lw=2)
    ax.set_xticks(range(len(keys)), [k.removeprefix("t_").replace("_", " ") for k in keys], fontsize=7, rotation=20)
    ax.axhline(599, color=MUTED, lw=0.8, ls=":"); ax.set_ylabel("step (median over trajectories)")
    ax.set_title("Time to exceed thresholds (points: seeds; bar: seed mean)")
    ax.set_ylim(0, 640); ax.text(len(keys) - 0.5, 605, "600 = not reached", fontsize=7, color=MUTED, ha="right")
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0.0, 0.92))
    fig.tight_layout(); fig.savefig(fig_dir / f"{prefix}_lh_threshold_times.png", dpi=150); plt.close(fig)


def to_json(obj):
    if isinstance(obj, dict):
        return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", default=str(ROOT / "results" / "local"))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    study = Path(args.study)
    summary, runs, ref, sliding = analyse(study)
    out = study / "long_horizon"
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(to_json(summary), indent=1))
    write_tables(summary, out / "tables.md")
    if not args.no_figures:
        plot(summary, runs, ref, sliding, ROOT / "figures", study.name)
    print(f"written {out / 'summary.json'} and tables.md; missing probe series: {summary['runs_missing_probe_series'] or 'none'}")


if __name__ == "__main__":
    main()
