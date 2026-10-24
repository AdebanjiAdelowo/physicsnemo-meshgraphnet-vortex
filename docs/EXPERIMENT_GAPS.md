# Experiment gaps and CUDA plan

What the reported `local` study cannot answer, and the experiments that would answer it, ordered by expected value per GPU hour. Nothing here has been run. All CUDA throughput figures are projections from the Apple MPS probe (`results/local/probe.json`) and must be replaced by `scripts/probe.py` on the target GPU before a study is started.

## 1. Open scientific questions

| # | Question | Why the current study cannot answer it |
|---|---|---|
| G1 | Are the poor rollouts caused by under-training or by a capacity or architecture limit? | One budget (11,960 steps, 3.2 % of NVIDIA's optimiser steps) and validation loss still falling at the last evaluation. Depth is varied only at that budget. |
| G2 | Does the ranking of depths by one-step error (15 < 10 < 5 in every seed) persist with training? | The gaps (0.031 in increment error) are of the size of the seed spread; the order is consistent in 3 of 3 seeds, which is weak evidence. |
| G3 | Does any depth produce a rollout better than persistence at a realistic budget? | At this budget none does on average after step 78 to 100. |
| G4 | How much of the seed effect is checkpoint noise? | Each run is evaluated at a single final checkpoint. Validation loss changes by up to 53 % between the last two evaluations (for example 0.053 to 0.077 for mp15_seed1), so the run-to-run spread mixes seed effects and where the last step landed. |
| G5 | Does per-use noise resampling (as in DeepMind's original) or a loss restricted to interior and outflow nodes improve stability? | Both are inherited PhysicsNeMo choices and are fixed in every run. |
| G6 | Are shedding frequency and amplitude reproduced for the shedding trajectories? | Only a probe-signal check on a few (run, trajectory) pairs exists (`ROLLOUT_VERIFICATION.md`); no Strouhal number or force coefficients. |
| G7 | Does the result hold on a representative test set? | 10 trajectories, the first of the test file; 3 of them have a steady wake. |
| G8 | Comparison at equal compute instead of equal steps. | Not run. |

## 2. Prioritised CUDA plan

Per-step training cost measured on MPS (idle machine): 37.7, 70.7, 104.0 ms for K = 5, 10, 15; rollout 17.5, 30.9, 44.5 ms per step. Projections below use MPS speed as a conservative bound and a speed-up factor `f` for the GPU (`f` is unmeasured; batch-1 graphs of about 2,000 nodes are launch-bound, so `f` may be small, roughly 1 to 4 on a T4). Divide the MPS-equivalent hours by `f`.

| Priority | Experiment | Config | Runs | MPS-equivalent hours (train + eval) | Data | Expected value |
|---|---|---|---|---|---|---|
| P1 | **Budget scaling at fixed depth**: K = 15 and K = 5 trained for 358,800 steps (3 epochs of 400 trajectories, close to NVIDIA's 373,750 optimiser steps), 3 seeds, evaluated at the per-epoch checkpoints (119,600, 239,200, 358,800 steps) | `full` with `num_training_samples=400 epochs=3 study.processor_sizes=[5,15]` | 6 long runs (checkpoints give the shorter budgets) | 358,800 x (37.7 + 104.0) ms x 3 seeds = 42.4 h, plus about 5.6 h evaluation of 3 checkpoints | 400 train trajectories, about 5.4 GB; 50 test, 0.7 GB (approval needed: > 500 MB) | Answers G1, G2 and G3 directly. Highest value. |
| P2 | **Planned depth study** `configs/full.yaml` (K = 5, 10, 15; 3 seeds; 119,600 steps; 50 test trajectories) | `full` as committed | 9 | 119,600 x 212.4 ms x 3 = 21.2 h train, about 2.8 h eval | 200 / 10 / 50 trajectories, about 3.5 GB | G2, G7. Partly subsumed by P1; run P1 first. |
| P3 | **Checkpoint averaging and variance**: evaluate the last 5 checkpoints (or an EMA of weights) of each P1 or P2 run | evaluation only | 0 new training | about 5 x eval cost of the study | none | G4: separates checkpoint noise from seed effects. Cheap. |
| P4 | **Noise resampled every epoch** and **loss on interior and outflow nodes only**, each at K = 15, 3 seeds, the P1 middle budget | requires a dataset wrapper; documented deviation from upstream | 6 | 119,600 x 104 ms x 6 = 20.7 h | as P2 | G5: tests two DeepMind choices that plausibly affect rollout stability. |
| P5 | **More seeds**: 5 to 10 seeds at the best P1 budget for K = 5 and 15 | `study.seeds=[0..9]` | 10 to 20 | proportional | as P2 | Needed before any claim of a depth effect on rollout error. |
| P6 | NVIDIA's 8-GPU reference (`official`, 15 steps) | `official` with 8-rank DDP | 1 | NVIDIA reports 8 A100; not projected here | 400 / 10 trajectories | Anchors the study to the published example. Low value per hour for the depth question. |

Commands (on the CUDA machine, after `python scripts/probe.py --config full --budget-hours <H>` passes):

```bash
python scripts/download_data.py --train 400 --valid 10 --test 50     # P1 data (approval: about 6 GB)
python scripts/train.py --config full num_training_samples=400 epochs=3 "study.processor_sizes=[5,15]"   # P1, 358,800 steps
python scripts/train.py --config full --resume                         # P2
```

`save_checkpoint` writes one file per epoch (`MeshGraphNet.0.<epoch>.mdlus`), but `engine.load_trained_model` calls `load_checkpoint` without an epoch and so loads the latest. Evaluating the intermediate budgets needs an epoch argument passed through to `load_checkpoint`; that small change is not made in this audit. The 11,960-step point is the existing `local` study (on 40 trajectories, which is a confound to state).

## 3. Statistical design for the next study

- Treat the seed as the unit of replication for model effects and the trajectory as a paired factor: every depth sees the same test trajectories, and for a fixed seed the same noise and sample order.
- Report per-trajectory paired differences between depths with bootstrap intervals over seeds and trajectories (hierarchical resampling), and the number of seeds in which an ordering holds. Do not report p-values from 3 seeds.
- Report shedding and steady trajectories separately; the steady cases dominate the large relative errors at low speed (for example test trajectory 7, peak inflow 0.34).
- Evaluate at several checkpoints, or average weights, so that the reported number is not the value at one arbitrary step.

## 4. Local follow-ups (CPU, no GPU needed)

- Full independent CPU rollout of all nine runs with `audit/independent_rollout.py` (about 1 to 6 hours per run on a loaded laptop CPU). Only run mp05_seed0 (all trajectories) and two shedding trajectories of mp10_seed0 and mp15_seed1 were repeated in this audit.
- Probe-spectrum analysis for every run and shedding trajectory (`audit/probe_spectrum.py`), to quantify frequency and amplitude errors beyond the lag-minimised error.
