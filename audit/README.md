# Audit scripts

Independent checks behind [`docs/SCIENTIFIC_AUDIT.md`](../docs/SCIENTIFIC_AUDIT.md). They do not use the repository's metric or rollout code, except where stated.

| Script | Needs | Checks |
|---|---|---|
| `recompute_aggregates.py results/local` | tracked results only | recomputes horizon values, time means, exceedance counts and seed aggregates from `rollout_curves.npz` and compares them with `eval_metrics.json` and `summary.json`; prints the tables of the docs |
| `rollout_diagnostics.py results/local` | tracked results only | worst trajectories, persistence and pressure crossings, lag-minimised error by trajectory class, energy ratios, seed and depth spread |
| `data_checks.py data/cylinder_flow` | downloaded data | node types, constancy of inflow and wall velocities, mesh overlap between splits, mesh area, cylinder geometry and shedding period of each test trajectory |
| `independent_rollout.py <repo> <out.npz> <run> ...` | data and checkpoints in `runs/local` | rollout from raw TFRecords with its own graph construction, normalisation and metrics; compares with the stored curves. `TRAJ=0,6` restricts the trajectories |
| `probe_spectrum.py <out.npz>` | output of `independent_rollout.py` | shedding period, band power and correlation of the predicted and reference probe signal; pooled one-step increment error |

Run them with the project environment, for example `.venv/bin/python audit/recompute_aggregates.py results/local`. The first two are also covered by `tests/test_results.py`.
