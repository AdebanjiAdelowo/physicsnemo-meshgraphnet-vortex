# Scientific audit

Independent audit of the message-passing depth study built on NVIDIA PhysicsNeMo's MeshGraphNet vortex-shedding example. It covers the implementation, the data handling, the experimental control, the metrics and every quantitative claim of the README.

- Repository state audited: commit `68b94c2` and the corrections made with this audit.
- Upstream reference: `NVIDIA/physicsnemo` commit `b45a5c810c741e6b41f8515be24c51121f8fc21f`, `examples/cfd/vortex_shedding_mgn`.
- Scope of the reported experiment: 9 models (K = 5, 10, 15 message-passing steps; seeds 0, 1, 2), each trained for 11,960 gradient steps of batch 1 on 40 training trajectories on Apple MPS (M3 Pro, PyTorch 2.13.0) and evaluated on 10 test trajectories with 599-step rollouts. This is a fixed-budget reduced study. It is 0.4 % of the 2,990,000 sample presentations of NVIDIA's configuration and 3.2 % of the 373,750 optimiser steps of NVIDIA's reported 8-GPU run. It is not a reproduction of NVIDIA's training. VERIFIED from `results/local/*/train_metrics.json` and `study_metadata.json`.

Companion documents: [UPSTREAM_COMPARISON.md](UPSTREAM_COMPARISON.md), [ROLLOUT_VERIFICATION.md](ROLLOUT_VERIFICATION.md), [EXPERIMENT_GAPS.md](EXPERIMENT_GAPS.md). Scripts: [`audit/`](../audit). Regression tests: `tests/test_results.py`, `tests/test_rollout.py::test_rollout_feeds_back_predictions_not_reference_states`.

## 1. Verdict

The implementation is correct and faithful to the upstream example. Processor depth is the only varied model setting. Data, noise realisation, sample order and gradient steps are matched across depths within each seed. Every published number recomputes from the stored per-step curves, and an independent CPU rollout from the checkpoints reproduces the stored MPS curves to about 1e-7. No implementation or metric error was found.

Some interpretive statements in the README were stronger than the data. They were corrected (section 4): the persistence crossing step, the wording on runs below persistence, the pressure crossing step, "not explosive", "not a phase error" and "adds energy at every step". The README also did not distinguish NVIDIA's 8-GPU optimiser steps from the single-device configuration; that is now clarified.

At this budget the study shows a consistent but small one-step gain with depth, and no rollout benefit from depth. No depth gives a rollout that beats holding the initial state fixed beyond about 100 steps. The study is suitable as evidence of careful experimental practice on a reduced budget. It is not evidence about MeshGraphNet's capability at convergence.

## 2. Evidence table

| # | Claim or property | Evidence | Status |
|---|---|---|---|
| E1 | Model, dataset class, graph, features, normalisation, loss, optimiser, schedule and rollout integration are PhysicsNeMo's, unchanged | line-by-line comparison; imports, not copies; upstream rollout comparison test passes (1e-5) | VERIFIED |
| E2 | Processor depth is the only varied model setting | `model.py` passes only `processor_size`; configs; parameter counts 845,059 / 1,588,739 / 2,332,419 | VERIFIED |
| E3 | Equal data and optimisation budget across depths | all 9 runs: 11,960 steps, 40 trajectories x 299 steps, 1 epoch, batch 1, same schedule | VERIFIED |
| E4 | Paired design within a seed (same noise, same order) | `node_stats.json` byte-identical across K per seed and different across seeds; the loader generator is seeded per seed | VERIFIED |
| E5 | Train / validation / test separation | separate DeepMind files; first 40 / 5 / 10 records; no mesh shared among the 90 downloaded trajectories; statistics only from the training subset of each run | VERIFIED |
| E6 | Inflow and wall values held fixed is exact | raw data: largest change over time on inflow and wall nodes 2.4e-7 | VERIFIED |
| E7 | Predicted, not reference, states fed back | code; new regression test; independent rollout | VERIFIED |
| E8 | Area weights and norms | nodal areas sum to channel minus cylinder within 0.02 %; unit tests | VERIFIED |
| E9 | Published aggregates (`eval_metrics.json`, `summary.json`, README tables) match the per-step curves | recomputation, max relative discrepancy 7e-6; `tests/test_results.py` | VERIFIED |
| E10 | Stored curves reproducible from checkpoints on other hardware | independent CPU rollout vs stored MPS curves (ROLLOUT_VERIFICATION.md section 4) | VERIFIED for the runs repeated |
| E11 | One-step error decreases with depth (increment error 0.581, 0.574, 0.550; ordering in all 3 seeds) | recomputation | VERIFIED; effect of the size of the seed SD |
| E12 | One-step pressure gain from 5 to 15 steps (0.488 to 0.381) exceeds the seed spread | SDs 0.007, 0.066, 0.021 | VERIFIED (descriptive; 3 seeds) |
| E13 | No depth gives a usable rollout; error above persistence from about step 100 | seed-mean crossing at steps 78 (K=15), 86 (K=5), 100 (K=10) | SUPPORTED BUT LIMITED, corrected to the exact steps |
| E14 | Two seed-0 runs "stay below" persistence over the training window | true of the window mean only; they cross the baseline at steps 176 and 274 | SUPPORTED BUT LIMITED, wording corrected |
| E15 | The one-step ranking does not carry over to the rollout; no ranking of depths by rollout error is supported | K = 15 best in 1 of 3 seeds; depth effects (±0.08) smaller than seed effects (±0.20) | VERIFIED |
| E16 | The seed matters more than the depth | additive decomposition of run means; includes checkpoint noise (validation loss changes up to 53 % between the last two evaluations) | SUPPORTED BUT LIMITED |
| E17 | Pressure error above 1 "from about step 150" | crossings at steps 109, 145, 167 | SUPPORTED BUT LIMITED, corrected |
| E18 | Growth gradual, "not explosive" | averages: yes; no non-finite values; single trajectories reach E_u 10.6 and 115x reference energy | SUPPORTED BUT LIMITED, qualified |
| E19 | "The error is not a phase error" | shedding periods 25 to 60 steps (within the ±40 window); a shift removes about 10 % of the error over steps 10 to 99 and about 1 % after step 100 on shedding trajectories | SUPPORTED BUT LIMITED, qualified ("not mainly") |
| E19b | Predicted wakes stop shedding | probe band power at most 5 % of the reference in mp05_seed0 (7 shedding trajectories), mp10_seed0 and mp15_seed1 (2 each) | SUPPORTED BUT LIMITED (3 of 9 runs) |
| E20 | Kinetic energy ratio 2 to 4 at step 299, smallest for K = 10 | seed means 3.99, 1.96, 2.97 | VERIFIED |
| E21 | "The models add energy at every step" | at step 10, 46 % of (run, trajectory) pairs are below the reference energy | INCORRECT as stated; replaced |
| E22 | Cost linear in depth (MPS probe) | 37.7, 70.7, 104.0 ms per step; increments 33.0 and 33.3 ms per 5 layers | VERIFIED (MPS only) |
| E23 | `official.yaml` is NVIDIA's training | reproduces the configuration; on one device it takes 8 times as many optimiser steps of batch 1 as NVIDIA's 8-GPU run and ends at a different learning rate | SUPPORTED BUT LIMITED, clarified |
| E24 | CPU-only `torch_scatter` build works on CUDA | correct by reading the code (sum reduction only); not executed on CUDA | UNVERIFIED |
| E25 | Kaggle launcher and `full` / `official` studies | statically tested only | UNVERIFIED (as the README states) |
| E26 | Velocity in m/s | the files do not state units | UNVERIFIED (minor) |

## 3. Experimental control

- **Intended variable.** Only `processor_size` differs between runs of a seed (E2, E4). Hidden size, MLP depth, aggregation, noise level, learning rate and training-set size are fixed. Interactions with these are not studied, which the README states.
- **Optimisation budget and data exposure.** Equal in steps and samples, each training sample seen exactly once (1 epoch, `drop_last` with batch 1 keeps all 11,960 samples) (E3). Not equal in compute: K = 15 costs 2.8 times as much per step on MPS (E22).
- **Convergence.** None of the models is converged. The validation loss falls by a factor of 2.1 to 5.2 between the first and last evaluation and is not flat at the end, and the learning rate stays at 0.99 of its initial value. The one-step increment error of 0.55 to 0.58 means more than half of the true one-step change is wrong. The conclusions describe an early-training regime.
- **Inherited choices.** Noise drawn once, statistics computed on noisy targets, loss over all nodes, pressure as an extra output. All are PhysicsNeMo's, and three of them differ from DeepMind's original (UPSTREAM_COMPARISON.md section 5). They are held fixed across runs, so they do not confound the depth comparison. They do limit comparison with published MeshGraphNet results.
- **Test set.** First 10 test trajectories. Seven shed vortices (reference periods 25 to 60 steps) and three are steady. Averages mix the two classes; ROLLOUT_VERIFICATION.md reports them separately where it matters.

## 4. Corrections made

| Location | Before | After | Basis |
|---|---|---|---|
| README, NVIDIA example | budget in sample presentations only | adds the 8-rank batch, 373,750 optimiser steps and final learning-rate factors (0.71 vs 0.07) | upstream `train.py` (DDP, per-rank batch 1) and README (8 A100) |
| README, budget table | "Gradient steps per model", "Share of the upstream budget" | one-device steps; share of sample presentations; share of NVIDIA's optimiser steps (3.2 %, 32 %) | as above |
| README, rollout | "After roughly 100 steps ... stay below the fixed-state baseline" | exact crossing steps 78 / 86 / 100; the seed-0 runs are below in the window mean and cross at steps 176 and 274 | `tests/test_results.py::test_readme_claims_on_the_persistence_baseline` |
| README, pressure | "above 1 from about step 150" | "from step 109 to 167 depending on depth" | `test_readme_claim_on_the_pressure_error` |
| README, accumulation | "Growth is gradual, not explosive" | averages gradual; single-trajectory errors up to 10.6 and energy up to 115x | `test_readme_claims_on_energy_and_phase` |
| README, phase | "not a phase error ... not at all afterwards" | "not mainly a phase error", per-class shift reductions, periods within the shift window, phase and amplitude caveat | `audit/rollout_diagnostics.py`, `audit/data_checks.py` |
| README, energy | "The models add energy at every step" | the drift is not monotone; ratio within 3 % of 1 at step 10; 86 % above the reference at step 299 | `audit/rollout_diagnostics.py` |
| README, limitations | none | differences from DeepMind's training; steady test cases | DeepMind source, data checks |
| Tests | no check of published results; feedback tested indirectly | `tests/test_results.py` (6 tests), explicit feedback test | |
| Environment | `requirements.txt` with lower bounds only | `requirements-lock.txt`, exact versions of the environment that produced the results | `uv pip freeze`, matching `study_metadata.json` |

The stored results, figures and metric code are unchanged. No historical result was altered.

## 5. Numerical limitations

- Relative errors are normalised per trajectory and time. Slow and steady trajectories have small norms and produce the largest relative errors, which dominate means and SDs at long horizons.
- The vorticity is a first-order, area-averaged P1 gradient. It is adequate for comparing fields, not for wall quantities.
- The lag-minimised error uses a shift of at most 40 steps chosen per time and normalised by the reference norm at the unshifted time. It cannot separate phase from amplitude errors when both are present.
- One-step errors are evaluated at every fifth step. The pooled increment error weights steps with large increments more.
- Seed SDs are from 3 values and are themselves uncertain by a large factor. No significance tests are reported, and none would be meaningful.
- Timings in `train_metrics.json` were taken on a shared machine and are not used for cost claims; `probe.json` was taken on an idle machine.
- MPS training determinism was not tested. Reproducibility of training is tested on CPU only. The evaluation reproduces across MPS and CPU to about 1e-7.

## 6. Reproducibility assessment

| Aspect | Assessment |
|---|---|
| Code provenance | training commit `60e28d1` (clean); evaluation at `a3cb34b`, which differs only in resume logic and plotting (verified with `git diff`) |
| Upstream pin | PhysicsNeMo installed from the pinned commit, recorded in metadata |
| Environment | full versions recorded per study; `requirements-lock.txt` added. No `uv.lock`; `requirements.txt` uses lower bounds |
| Data | subset selection by record order, byte counts and SHA-256 recorded; the hashes are of the downloaded files (60/10/20 records), not of the 40/5/10 used, so a fresh `--train 40` download has a different hash but identical records |
| Randomness | seeds control noise, initialisation and order; CPU training bit-reproducible (test) |
| Outputs | per-step curves for every run and trajectory tracked in git; checkpoints and data not tracked (sizes: about 1.2 GB of data) |
| Re-execution | the whole local study is about 1.1 hours of idle-machine MPS time per seed (probe projection: 0.81 h training, 0.28 h evaluation); the audit's checks run from tracked files except the independent rollout (needs data and checkpoints) |

Overall: **high** for the reported reduced study. The CUDA studies have not been executed.

## 7. Defensible statements for a CV or application

Supported by the evidence above:

- "Designed and ran a controlled study of message-passing depth (5, 10, 15 layers, 3 seeds each) in NVIDIA PhysicsNeMo's MeshGraphNet on the DeepMind cylinder-flow dataset, at a fixed reduced budget of 11,960 gradient steps per model, with paired seeds, matched data and full provenance."
- "Built an evaluation of autoregressive rollouts beyond the upstream example: area-weighted relative errors, a persistence baseline, lag-minimised (phase-aligned) error, kinetic-energy and vorticity diagnostics, and a test that matches NVIDIA's rollout to 1e-5."
- "Found that at this budget depth improves one-step accuracy consistently but does not improve 600-step rollout accuracy; all models fall behind a persistence baseline after about 80 to 100 steps, with error dominated by loss of vortex shedding and energy drift rather than phase error."
- "Verified the results independently: aggregates recomputed from per-step curves, and rollouts reproduced from checkpoints on a second backend (CPU vs Apple MPS) to about 1e-7."

Not supported, and to be avoided:

- Any statement that the models reproduce vortex shedding, predict accurate long rollouts, or match NVIDIA's or DeepMind's accuracy.
- Any claim of a statistically significant depth effect, or of an optimal depth.
- Any statement that NVIDIA's configuration or a CUDA study was run, or that the study is a reproduction of the upstream training.
- "Shallower models are better for rollouts" (K = 10 is best in the mean and in 2 of 3 seeds, within seed noise).

## 8. Outstanding scientific work

Ordered by value (details, commands and resource estimates in [EXPERIMENT_GAPS.md](EXPERIMENT_GAPS.md)):

1. Budget scaling at fixed depth (K = 5 and 15, up to about 360,000 steps, 3 seeds) to separate under-training from capacity limits.
2. The planned `full` depth study on 50 test trajectories, after (1).
3. Checkpoint averaging and multi-checkpoint evaluation to separate checkpoint noise from seed effects.
4. Per-use noise resampling and an interior-and-outflow loss mask, as in DeepMind's original.
5. More seeds (5 to 10) before any claim about a depth effect on rollouts.
6. Frequency and amplitude diagnostics for every shedding trajectory and run (done for 3 of 9 runs: no shedding in the predictions).
