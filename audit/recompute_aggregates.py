"""Independent recomputation of the stored local-study aggregates from rollout_curves.npz."""
import json
import statistics
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
H = [1, 10, 50, 100, 200, 299, 450, 599]
W = 299
summary = json.loads((root / "summary.json").read_text())
runs = sorted(p.name for p in root.glob("mp*_seed*"))
worst = 0.0
rows = {}
for run in runs:
    c = dict(np.load(root / run / "rollout_curves.npz"))
    ev = json.loads((root / run / "eval_metrics.json").read_text())
    tm = json.loads((root / run / "train_metrics.json").read_text())
    K, S = int(run[2:4]), int(run[-1])
    v = c["rel_l2_velocity"].astype(np.float64)
    assert v.shape == (10, 599), v.shape
    assert np.isfinite(v).all()
    r = {"K": K, "S": S, "steps": tm["steps"], "params": tm["parameters"]}
    for h in H:
        for m in ("rel_l2_velocity", "rel_l2_pressure", "rel_l2_vorticity", "lagmin_rel_l2_velocity", "persistence_rel_l2_velocity"):
            mine = c[m][:, h - 1].astype(np.float64).mean()
            stored = ev["rollout_horizons"][str(h)][m]["mean"]
            worst = max(worst, abs(mine - stored) / abs(stored))
            r[f"{m}_h{h}"] = mine
        ke = np.abs(c["kinetic_energy_pred"][:, h - 1] - c["kinetic_energy_true"][:, h - 1]) / c["kinetic_energy_true"][:, h - 1]
        worst = max(worst, abs(ke.mean() - ev["rollout_horizons"][str(h)]["kinetic_energy_rel_error"]["mean"]) / ke.mean())
        r[f"ke_rel_h{h}"] = ke.mean()
        r[f"ke_ratio_h{h}"] = (c["kinetic_energy_pred"][:, h - 1] / c["kinetic_energy_true"][:, h - 1]).mean()
    for span, sl in (("window", slice(0, W)), ("all", slice(None))):
        for m in ("rel_l2_velocity", "rel_l2_pressure", "lagmin_rel_l2_velocity", "persistence_rel_l2_velocity"):
            mine = c[m][:, sl].astype(np.float64).mean(axis=1).mean()
            stored = ev["rollout_time_mean"][span][m]["mean"]
            worst = max(worst, abs(mine - stored) / stored)
            r[f"mean_{m}_{span}"] = mine
    exceed = int(((v > 1.0).any(axis=1)).sum())
    assert exceed == ev["trajectories_with_velocity_error_above_1"], (run, exceed)
    r["exceed"] = exceed
    r["per_traj_window"] = v[:, :W].mean(axis=1)
    r["per_traj_persist_window"] = c["persistence_rel_l2_velocity"][:, :W].astype(np.float64).mean(axis=1)
    r["one_step_inc"] = ev["one_step"]["window"]["rel_l2_velocity_increment"]
    r["one_step_p"] = ev["one_step"]["window"]["rel_l2_pressure"]
    r["one_step_u"] = ev["one_step"]["window"]["rel_l2_velocity"]
    r["valid"] = tm["final_valid_loss"]
    r["curves"] = c
    rows[run] = r
print(f"max relative discrepancy, recomputed vs eval_metrics.json: {worst:.2e}")

# seed aggregates vs summary.json
agg = {e["processor_size"]: e for e in summary["by_processor_size"]}
worst = 0.0
for K in (5, 10, 15):
    g = [rows[f"mp{K:02d}_seed{s}"] for s in (0, 1, 2)]
    for key, skey in (("mean_rel_l2_velocity_window", "rollout_mean_rel_l2_velocity_window"), ("rel_l2_velocity_h299", "rollout_rel_l2_velocity_h299"), ("rel_l2_velocity_h599", "rollout_rel_l2_velocity_h599")):
        vals = [x[key] for x in g]
        worst = max(worst, abs(statistics.fmean(vals) - agg[K][skey + "_mean"]) / agg[K][skey + "_mean"], abs(statistics.stdev(vals) - agg[K][skey + "_std"]) / agg[K][skey + "_std"])
print(f"max relative discrepancy, seed mean/std vs summary.json: {worst:.2e}")

def ms(K, key):
    vals = [rows[f"mp{K:02d}_seed{s}"][key] for s in (0, 1, 2)]
    return statistics.fmean(vals), statistics.stdev(vals)

print("\nsteps/params:", sorted({(r['K'], r['steps'], r['params']) for r in rows.values()}))
print("\nOne-step (mean ± sd over seeds): K, valid, inc, u, p")
for K in (5, 10, 15):
    print(K, *("%.4f ± %.4f" % ms(K, k) for k in ("valid", "one_step_inc", "one_step_u", "one_step_p")))
print("\nRollout velocity rel L2 at horizons (mean ± sd over seeds of trajectory means)")
for K in (5, 10, 15):
    print(K, " ".join("h%d=%.3f±%.3f" % (h, *ms(K, f"rel_l2_velocity_h{h}")) for h in H), " win=%.3f±%.3f" % ms(K, "mean_rel_l2_velocity_window"), " all=%.3f±%.3f" % ms(K, "mean_rel_l2_velocity_all"))
print("persistence at horizons:", " ".join("h%d=%.3f" % (h, rows['mp05_seed0'][f'persistence_rel_l2_velocity_h{h}']) for h in H), "win=%.3f all=%.3f" % (rows['mp05_seed0']['mean_persistence_rel_l2_velocity_window'], rows['mp05_seed0']['mean_persistence_rel_l2_velocity_all']))
for name in ("rel_l2_pressure", "rel_l2_vorticity", "lagmin_rel_l2_velocity", "ke_rel", "ke_ratio"):
    print(f"\n{name}:")
    for K in (5, 10, 15):
        print(K, " ".join("h%d=%.3f±%.3f" % (h, *ms(K, f"{name}_h{h}")) for h in H))
print("\nlag-min window mean vs plain:")
for K in (5, 10, 15):
    print(K, "%.3f±%.3f" % ms(K, "mean_lagmin_rel_l2_velocity_window"), "%.3f±%.3f" % ms(K, "mean_rel_l2_velocity_window"))

print("\nPer-seed window mean (exceed):")
for K in (5, 10, 15):
    print(K, " ".join("%.3f (%d)" % (rows[f'mp{K:02d}_seed{s}']['mean_rel_l2_velocity_window'], rows[f'mp{K:02d}_seed{s}']['exceed']) for s in (0, 1, 2)))

# Crossing of the persistence baseline: first step where seed-and-trajectory mean model error exceeds the persistence mean
pers = rows["mp05_seed0"]["curves"]["persistence_rel_l2_velocity"].astype(np.float64).mean(axis=0)
for K in (5, 10, 15):
    m = np.mean([rows[f"mp{K:02d}_seed{s}"]["curves"]["rel_l2_velocity"].astype(np.float64).mean(axis=0) for s in (0, 1, 2)], axis=0)
    above = np.where(m > pers)[0]
    stays = next((i for i in range(len(m)) if (m[i:] > pers[i:]).all()), None)
    print(f"K={K}: first step model mean > persistence mean: {above[0]+1 if len(above) else None}; permanently above from step {stays+1 if stays is not None else None}")

# Runs below persistence over training window, per trajectory beats
print("\nRuns whose window-mean error < persistence window-mean (trajectory means); trajectories beating persistence:")
for run, r in rows.items():
    beat = int((r["per_traj_window"] < r["per_traj_persist_window"]).sum())
    print(run, "%.3f vs %.3f" % (r["mean_rel_l2_velocity_window"], r["mean_persistence_rel_l2_velocity_window"]), f"trajectories beating persistence: {beat}/10")

# Paired depth comparison: within each seed, per trajectory
print("\nPaired, per seed: ranking of window-mean rollout error by K, and of one-step increment error")
for s in (0, 1, 2):
    w = {K: rows[f"mp{K:02d}_seed{s}"]["mean_rel_l2_velocity_window"] for K in (5, 10, 15)}
    o = {K: rows[f"mp{K:02d}_seed{s}"]["one_step_inc"] for K in (5, 10, 15)}
    p = {K: rows[f"mp{K:02d}_seed{s}"]["one_step_p"] for K in (5, 10, 15)}
    print(s, "rollout:", sorted(w, key=w.get), "one-step inc:", sorted(o, key=o.get), "one-step p:", sorted(p, key=p.get))
print("\nPer trajectory x seed (30 pairs): fraction where K=15 rollout window error < K=5:")
for a, b in ((15, 5), (10, 5), (15, 10)):
    d = np.concatenate([rows[f"mp{a:02d}_seed{s}"]["per_traj_window"] - rows[f"mp{b:02d}_seed{s}"]["per_traj_window"] for s in (0, 1, 2)])
    print(f"K={a} vs K={b}: {int((d < 0).sum())}/30 lower, median diff {np.median(d):+.3f}")
print("\nreference fluctuation ratio per test trajectory:", np.round(summary["reference_fluctuation_ratio"], 4).tolist())
