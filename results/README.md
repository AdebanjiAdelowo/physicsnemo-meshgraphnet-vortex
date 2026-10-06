# Results

Tracked result files. Each study directory is a copy of a run directory without its checkpoints, written by `python scripts/results.py publish <run directory or archive>`.

| Path | Content |
|---|---|
| `<study>/study_metadata.json` | timestamp, commit and dirty state of this repository, PhysicsNeMo version and commit, device, package versions, dataset subset with file hashes, resolved configuration |
| `<study>/summary.csv` | one row per run |
| `<study>/summary.json` | the same rows, mean and standard deviation over seeds per processor size, the fixed-state baseline and the unsteadiness of every test trajectory |
| `<study>/fields.npz` | mesh, reference and predicted fields (u, v, p) of the plotted test trajectory at the stored steps, for every processor size of the first seed |
| `<study>/probe.json` | training and rollout speed per processor size from `scripts/probe.py` |
| `<study>/mp<K>_seed<S>/history.csv` | learning rate, training loss and validation loss against the gradient step |
| `<study>/mp<K>_seed<S>/train_metrics.json` | seed, processor size, parameter count, steps, timings, memory |
| `<study>/mp<K>_seed<S>/eval_metrics.json` | one-step errors, rollout errors at the tabulated horizons, time means, counts of trajectories with large or non-finite errors |
| `<study>/mp<K>_seed<S>/rollout_curves.npz` | per test trajectory and rollout step: velocity, pressure and vorticity errors, RMSE, lag-minimised error and best lag, kinetic energy of prediction and reference, fixed-state baseline |
| `<study>/mp<K>_seed<S>/node_stats.json`, `edge_stats.json` | normalisation statistics written by the upstream dataset class |

Studies:

| Study | Configuration | Device | Status |
|---|---|---|---|
| `local` | `configs/local.yaml`, 11,960 gradient steps per model | Apple MPS | LOCAL/REDUCED: run, 3 depths by 3 seeds |
| `full` | `configs/full.yaml`, 119,600 gradient steps per model | CUDA | CUDA/FULL: PENDING, not run |
| `official` | `configs/official.yaml`, 2,990,000 gradient steps | CUDA | PENDING, requires larger compute, not run |

Notes on `local`:

- `study_metadata.json` records the commit the models were trained with (`60e28d1`, clean tree). The evaluation was run afterwards at commit `a3cb34b`, which differs from the training commit by a resume option of the evaluation and by figure, probe and packaging scripts; the training, rollout and metric code is the same.
- The durations in `train_metrics.json` and `rollout_ms_per_step` in `eval_metrics.json` were measured while other computations shared the machine. `probe.json` holds the cost measurement taken on the idle machine.
- Peak memory fields are CUDA allocations and are empty on MPS.

Units: losses are mean squared errors of the normalised targets. `rel_l2_*` are relative $L^2$ errors with lumped nodal area weights, as fractions. `rmse_*` are nodal root mean square errors in the units of the data. `kinetic_energy_*` is half the area integral of the squared velocity. Steps are counted from 1, the first predicted time level.
