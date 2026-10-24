"""Further diagnostics from stored rollout_curves.npz: tails, crossings, per-trajectory phase."""
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
UNSTEADY = [0, 1, 2, 4, 5, 6, 9]
STEADY = [3, 7, 8]
C = {p.name: dict(np.load(p / "rollout_curves.npz")) for p in sorted(root.glob("mp*_seed*"))}
pers = C["mp05_seed0"]["persistence_rel_l2_velocity"].astype(np.float64)

print("Worst trajectory per run at steps 299/599: velocity error, KE ratio, vorticity error")
for run, c in C.items():
    v = c["rel_l2_velocity"]; ke = c["kinetic_energy_pred"] / c["kinetic_energy_true"]; w = c["rel_l2_vorticity"]
    print(f"{run}: maxE299 {v[:,298].max():.2f} maxE599 {v[:,598].max():.2f} (traj {v[:,598].argmax()}), "
          f"max KE ratio 599 {ke[:,598].max():.1f}, max vort 599 {w[:,598].max():.1f}, "
          f"growth E599/E299 max {np.max(v[:,598]/v[:,298]):.2f}")

print("\nPressure: first step where seed-mean of trajectory-mean pressure error > 1, per K")
for K in (5, 10, 15):
    m = np.mean([C[f"mp{K:02d}_seed{s}"]["rel_l2_pressure"].astype(np.float64).mean(0) for s in range(3)], 0)
    print(K, int(np.argmax(m > 1)) + 1, "min over steps 1-20 at step", int(np.argmin(m[:20])) + 1, f"{m[:20].min():.3f}", f"step1 {m[0]:.3f}")

print("\nSeed-0 runs: fraction of steps 1-299 where trajectory-mean error < persistence")
for K in (5, 10):
    m = C[f"mp{K:02d}_seed0"]["rel_l2_velocity"].astype(np.float64).mean(0)[:299]
    print(K, f"{(m < pers.mean(0)[:299]).mean():.2f}", "first step above:", int(np.argmax(m > pers.mean(0)[:299])) + 1)

print("\nLag-minimised vs plain error, steps 100-299, unsteady vs steady trajectories (mean over 9 runs)")
for name, idx in (("unsteady", UNSTEADY), ("steady", STEADY)):
    red = [1 - c["lagmin_rel_l2_velocity"][idx, 99:299].mean() / c["rel_l2_velocity"][idx, 99:299].mean() for c in C.values()]
    red_early = [1 - c["lagmin_rel_l2_velocity"][idx, 9:99].mean() / c["rel_l2_velocity"][idx, 9:99].mean() for c in C.values()]
    print(f"{name}: reduction steps 10-99 {np.mean(red_early)*100:.1f}% (range {np.min(red_early)*100:.1f}-{np.max(red_early)*100:.1f}), "
          f"steps 100-299 {np.mean(red)*100:.1f}% (range {np.min(red)*100:.1f}-{np.max(red)*100:.1f})")
print("per unsteady trajectory, steps 100-299, best run reduction:")
for i in UNSTEADY:
    red = [1 - c["lagmin_rel_l2_velocity"][i, 99:299].mean() / c["rel_l2_velocity"][i, 99:299].mean() for c in C.values()]
    print(f"  traj {i}: max {np.max(red)*100:.1f}%, mean {np.mean(red)*100:.1f}%")

print("\nUnsteady-only vs all-trajectory window mean velocity error (seed mean)")
for K in (5, 10, 15):
    a = np.mean([C[f"mp{K:02d}_seed{s}"]["rel_l2_velocity"][:, :299].mean() for s in range(3)])
    u = np.mean([C[f"mp{K:02d}_seed{s}"]["rel_l2_velocity"][UNSTEADY, :299].mean() for s in range(3)])
    st = np.mean([C[f"mp{K:02d}_seed{s}"]["rel_l2_velocity"][STEADY, :299].mean() for s in range(3)])
    print(K, f"all {a:.3f} unsteady {u:.3f} steady {st:.3f}")
print("persistence: all %.3f unsteady %.3f steady %.3f" % (pers[:, :299].mean(), pers[UNSTEADY, :299].mean(), pers[STEADY, :299].mean()))

print("\nKE ratio pred/true at step 299 by trajectory class (seed mean of trajectory means)")
for K in (5, 10, 15):
    for name, idx in (("unsteady", UNSTEADY), ("steady", STEADY)):
        r = np.mean([(C[f"mp{K:02d}_seed{s}"]["kinetic_energy_pred"][idx, 298] / C[f"mp{K:02d}_seed{s}"]["kinetic_energy_true"][idx, 298]).mean() for s in range(3)])
        print(K, name, f"{r:.2f}")
print("\nKE ratio at step 50, 100: fraction of (run, traj) with ratio > 1:")
allke = np.stack([c["kinetic_energy_pred"] / c["kinetic_energy_true"] for c in C.values()])
for h in (10, 50, 100, 299):
    print(h, f"{(allke[:, :, h-1] > 1).mean():.2f}")

print("\nSeed-to-seed vs depth-to-depth spread of window-mean error, per trajectory (std over seeds at fixed K vs std over K at fixed seed)")
arr = np.array([[C[f"mp{K:02d}_seed{s}"]["rel_l2_velocity"][:, :299].mean(1) for s in range(3)] for K in (5, 10, 15)])  # K, S, traj
print("mean std over seeds:", f"{arr.std(axis=1, ddof=1).mean():.3f}", " mean std over depths:", f"{arr.std(axis=0, ddof=1).mean():.3f}")
# two-way decomposition (no inference): K effect, S effect, residual on run-level means
run_means = arr.mean(2)
grand = run_means.mean()
print("run-level means (K x S):\n", np.round(run_means, 3))
print("K effect range:", np.round(run_means.mean(1) - grand, 3), " S effect range:", np.round(run_means.mean(0) - grand, 3))
