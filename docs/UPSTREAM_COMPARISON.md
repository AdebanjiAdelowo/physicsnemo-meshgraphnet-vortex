# Comparison with the NVIDIA PhysicsNeMo example

Reference: `examples/cfd/vortex_shedding_mgn` of [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo/tree/b45a5c810c741e6b41f8515be24c51121f8fc21f/examples/cfd/vortex_shedding_mgn) at commit `b45a5c810c741e6b41f8515be24c51121f8fc21f` (`train.py`, `inference.py`, `conf/config.yaml`, `README.md`) and the dataset class `physicsnemo/datapipes/gnn/vortex_shedding_dataset.py` at the same commit. The comparison was made line by line against a local clone checked out at that commit. The installed package records the same commit (`study_metadata.json`, `physicsnemo.commit`).

Status labels: **VERIFIED** (checked in code or by a test or recomputation), **SUPPORTED BUT LIMITED** (correct with a stated qualification), **UNVERIFIED** (not checkable here), **INCORRECT**.

## Summary

The repository uses the upstream model, dataset class, graph construction, normalisation, loss, optimiser, schedule and rollout integration unchanged. It imports them; it does not copy them. The only change to the model is the explicit `processor_size` argument. A test runs NVIDIA's `MGNRollout.predict` and the repository's `rollout` on the same model and data, and the fields agree to 1e-5 (`tests/test_rollout.py::test_matches_the_upstream_rollout_if_a_clone_is_available`, passed in this audit with `PHYSICSNEMO_UPSTREAM` set). No deviation that would change the meaning of the depth study was found.

There is one difference in what the configurations mean. NVIDIA reports its training on 8 GPUs with data parallelism. `configs/official.yaml` run on one device reproduces the configuration file, not that optimisation trajectory (section 4).

## 1. Component comparison

| Component | Upstream (`b45a5c8`) | This repository | Status |
|---|---|---|---|
| Graph | one graph per trajectory from `cells[0]`; edges `(0,1),(1,2),(2,0)` of each triangle, `to_undirected` | upstream class, unchanged | VERIFIED (`tests/test_data.py::test_training_graph`: every mesh edge once in each direction) |
| Edge features | `pos[row] - pos[col]` and its norm, standardised with training-set `edge_stats.json` | unchanged; statistics written per run directory | VERIFIED |
| Node features | velocity at `t` (2) and one-hot node type, 4 classes (0, 4, 5, 6 mapped to 0..3) | unchanged | VERIFIED; the downloaded files contain only node types 0, 4, 5, 6 (`audit/data_checks.py`) |
| Targets | `u(t+1) - u(t)` (2) and `p(t+1)` (1) | unchanged | VERIFIED |
| Training noise | N(0, 0.02) on interior-node velocity, subtracted from the increment target, drawn once in the constructor with the global torch generator | unchanged; the generator is seeded per run | VERIFIED; statistics are byte-identical across depths for a seed and differ between seeds |
| Normalisation statistics | per-trajectory means averaged over trajectories, `std = sqrt(E[x^2] - E[x]^2)`, computed after the noise; test/valid splits load the training statistics | unchanged; each run writes and reads its own statistics | VERIFIED; no test or validation data enter the statistics |
| Model | `MeshGraphNet(6, 3, 3)` with default `processor_size=15`, hidden 128, 2-layer MLPs, sum aggregation, LayerNorm except the decoder | same call with `processor_size` explicit | VERIFIED; parameter counts 845,059 / 1,588,739 / 2,332,419 |
| Loss | `MSELoss` over all nodes and all 3 normalised targets | unchanged | VERIFIED |
| Optimiser | Adam, lr 1e-4, `fused=torch.cuda.is_available()` | Adam, lr 1e-4, `fused` on CUDA only | VERIFIED (identical on CUDA; fused Adam is unavailable on MPS) |
| Schedule | `LambdaLR(0.9999991**step)`, stepped every iteration | unchanged | VERIFIED; `history.csv` ends at lr 9.8929e-05 = 1e-4 x 0.9999991^11960 |
| Batch | 1 per rank, `DistributedSampler(shuffle=True, drop_last=True)` | 1, `DataLoader(shuffle=True, drop_last=True)` with a seeded generator | VERIFIED; different permutation, same distribution |
| AMP / compile | off | off | VERIFIED |
| Checkpointing | `save_checkpoint` per epoch; training resumes from `ckpt_path` | per epoch; every run starts fresh | VERIFIED |
| Rollout | `MGNRollout.predict`: from the initial condition of each test trajectory, predicted velocity fed back, inflow and wall nodes not advanced | function over one trajectory, same arithmetic, no in-place modification of the batch | VERIFIED by the upstream comparison test (agreement 1e-5) |
| Boundary mask | `rollout_mask` = node type 0 or 5 (interior, outflow) are advanced. The upstream comment says "do not update the wall_boundary & outflow nodes", which does not match its own mask | interior and outflow advanced; the comment in `rollout.py` describes the mask correctly | VERIFIED; the upstream comment is wrong, its code is right |
| Pressure | output only, never an input | same | VERIFIED |
| Metrics | none (the example writes animations) | relative L2 with lumped nodal areas, RMSE, lag-minimised error, kinetic energy, vorticity, persistence baseline | added; see `ROLLOUT_VERIFICATION.md` |

## 2. Configuration keys

`configs/official.yaml` holds every upstream key with the upstream value, except that the `hydra` block is removed and `data_dir` points into the repository (VERIFIED by `tests/test_config.py` and by reading both files). The reduced studies change only the budget keys (`epochs`, `num_training_samples`, `num_test_samples`, `num_test_time_steps`, workers, logging) and the study keys.

| Key | Upstream | `local` (reported) | `full` | Status |
|---|---|---|---|---|
| `num_training_samples` | 400 | 40 | 200 | VERIFIED |
| `num_training_time_steps` | 300 | 300 | 300 | VERIFIED |
| `epochs` | 25 | 1 | 2 | VERIFIED |
| `num_test_samples` | 10 | 10 | 50 | VERIFIED |
| `num_test_time_steps` | 300 | 600 | 600 | VERIFIED; rollouts of 599 steps extend 300 steps past the training window |
| `processor_size` | 15 (default) | 5, 10, 15 | 5, 10, 15 | VERIFIED |

## 3. Upstream README against upstream code

The upstream README describes 1000 training samples and 600 time steps, while `config.yaml` uses 400 trajectories and 300 time steps. The repository follows the code and says so. VERIFIED.

## 4. Training budget: configuration against NVIDIA's reported run

| Quantity | NVIDIA, 8 GPUs (README) | `official.yaml` on 1 device | `local` |
|---|---|---|---|
| Sample presentations | 2,990,000 | 2,990,000 | 11,960 |
| Samples per optimiser step | 8 (DDP gradient average) | 1 | 1 |
| Optimiser steps | 373,750 | 2,990,000 | 11,960 |
| Final learning rate / initial | 0.9999991^373,750 = 0.714 | 0.9999991^2,990,000 = 0.068 | 0.989 |

The original README stated the local study as 0.4 % of the upstream budget. That is correct in sample presentations. It is 3.2 % of NVIDIA's optimiser steps, each averaging the gradient over 1 sample instead of 8, at an essentially constant learning rate. The README now gives both shares and states that `official.yaml` on one device does not reproduce NVIDIA's 8-GPU optimisation trajectory. Before this audit the README did not draw the distinction: **SUPPORTED BUT LIMITED, now clarified**.

## 5. Inherited differences from DeepMind's MeshGraphNets

These are properties of PhysicsNeMo's example, not defects of this repository. They matter when the results are compared with Pfaff et al. (2021). They were checked against `deepmind-research/meshgraphnets` (`cfd_model.py`, `dataset.py`, `run_model.py`, master branch, read during this audit).

| Aspect | DeepMind | PhysicsNeMo (and this study) |
|---|---|---|
| Training noise | redrawn every time a sample is used (`tf.data` map before `shuffle` and `repeat`) | drawn once per dataset construction |
| Loss nodes | interior and outflow only | all nodes |
| Outputs | velocity increment only (`size=2`) | velocity increment and pressure |
| Training length | `num_training_steps` default 10,000,000 | 2,990,000 presentations (configuration) |
| Learning rate | 1e-4 x 0.1^(step / 5e6) + 1e-6 | 1e-4 x 0.9999991^step |
| Normaliser | online accumulation | precomputed from the training subset |

Both pipelines compute the increment statistics on noisy targets, so the noise-dominated `velocity_diff_std` (0.019 to 0.020) is not specific to PhysicsNeMo. VERIFIED from both sources.

## 6. Points that could not be verified here

- CUDA behaviour of the repository, including the CPU-only `torch_scatter` build on a CUDA machine. The argument that only the sum reduction is used is correct for this model (`physicsnemo/nn/module/gnn_layers/utils.py` calls `torch_scatter.scatter(..., reduce=aggregation)` with `aggregation="sum"`), but it was not executed on CUDA. **UNVERIFIED**.
- Units. The files give `dt = 0.01` but no units for velocity or pressure, so "m/s" in the README is an assumption about the COMSOL export. **UNVERIFIED** (minor).
