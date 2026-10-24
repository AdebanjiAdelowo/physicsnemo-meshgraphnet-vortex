# Rollout verification

Checks of the autoregressive rollout, the one-step evaluation and the reported rollout metrics of the `local` study (9 runs: K = 5, 10, 15 message-passing steps, seeds 0, 1, 2, 11,960 gradient steps each on Apple MPS, 10 test trajectories, 599-step rollouts). The scripts are in `audit/`; the consistency checks of the published numbers are also regression tests (`tests/test_results.py`).

Status labels: **VERIFIED**, **SUPPORTED BUT LIMITED**, **UNVERIFIED**, **INCORRECT**.

## 1. Rollout mechanics

| Check | Evidence | Status |
|---|---|---|
| Predicted states, not reference states, are fed back | `rollout.py` uses `pred[-1][:, 0:2]` as the next input for `i > 0`. New test `test_rollout_feeds_back_predictions_not_reference_states` records the model inputs and requires input k = prediction k-1 and input k != reference k. A zero-increment model accumulates the mean increment over all steps (`test_boundary_velocity_is_never_updated`). The independent CPU rollout (section 4), which feeds back its own predictions, reproduces the stored curves | VERIFIED |
| Only the initial state is taken from the reference | the first input is `x_velocity` of step 0; the new test checks it equals the reference initial condition | VERIFIED |
| One-step evaluation uses reference inputs | `autoregressive=False` uses `x_velocity` at every step; `test_one_step_uses_reference_inputs` | VERIFIED |
| Boundary treatment | inflow (type 4) and wall (type 6) nodes keep their input value; interior (0) and outflow (5) are advanced, as in NVIDIA's and DeepMind's rollouts. In the raw files the inflow and wall velocities are constant in time (largest change 2.4e-7 over 90 downloaded trajectories), so holding them at the initial value is exact. The independent rollout gives a largest boundary-node error of zero to float precision | VERIFIED |
| Pressure feedback | pressure is an output only; it never enters the input. Pressure errors therefore follow from the velocity state | VERIFIED |
| Normalisation inversion | input normalised with `velocity_mean/std`; output increment denormalised with `velocity_diff_mean/std`, pressure with `pressure_mean/std`, all from the run's training statistics. The oracle test reproduces the reference to 2e-5 | VERIFIED |
| Time indexing | prediction k is time level k (k = 1..599); horizon h is curve index h-1; the training window is levels 1..299 (`num_training_time_steps = 300`) | VERIFIED |
| Graph-to-mesh reconstruction | node order of the graph equals the order of `mesh_pos`; `cells` of the same trajectory are used for areas and gradients. Nodal areas sum to the channel area minus the cylinder within 0.02 % on all 10 test meshes | VERIFIED |
| Agreement with NVIDIA's `MGNRollout.predict` | upstream comparison test, 1e-5 | VERIFIED |
| Hardware independence | the independent CPU rollout matches the stored MPS curves to about 1e-7 (section 4) | VERIFIED for the runs repeated |

## 2. Metric definitions and aggregation

| Quantity | Definition in code | Aggregation | Status |
|---|---|---|---|
| E_u(t), E_p(t) | area-weighted relative L2, lumped nodal areas | per trajectory and step | VERIFIED (unit tests with known answers; independent reimplementation) |
| Horizon value | E(t = h) | mean over 10 trajectories per run, then mean and sample SD over 3 seeds | VERIFIED by recomputation (max relative discrepancy 7e-6) |
| Time mean "window" | per trajectory mean over steps 1..299, then mean over trajectories | then seeds | VERIFIED |
| One-step velocity error | mean of per-step relative errors at steps 1, 6, ..., 296 | pooled over trajectories | VERIFIED |
| One-step increment error | sqrt(sum of squared errors / sum of squared true increments), pooled over steps and trajectories; 1 = no-change prediction | then seeds | VERIFIED |
| Lag-minimised error | min over shifts \|s\| <= 40 of the error against the reference at t+s, divided by the norm at t; shift chosen per t | as E_u | VERIFIED; see section 5 for interpretation |
| Kinetic energy error | \|KE_pred - KE_true\| / KE_true at horizon h | mean over trajectories | VERIFIED |
| Persistence baseline | the initial velocity held fixed for all t | as E_u | VERIFIED |
| Exceedance count | trajectories with E_u > 1 at some step | per run | VERIFIED |

The aggregation is consistent across tables. Two points apply to every average:

- Seed means of trajectory means weight all 10 trajectories equally. Steady trajectories (3 of 10) and slow trajectories have small reference norms, so their relative errors are large and dominate the tails.
- Means of relative errors over trajectories are not the same as relative errors of pooled norms. Both are legitimate; the README uses the first for rollouts and the pooled form for the one-step increment error, and says so.

## 3. Rollout error at the requested horizons

Relative L2 velocity error, mean ± SD over 3 seeds of the mean over 10 test trajectories (recomputed from `rollout_curves.npz`):

| K | h=1 | 10 | 50 | 100 | 200 | 299 | 450 | 599 |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.027 ± 0.009 | 0.085 ± 0.018 | 0.231 ± 0.062 | 0.434 ± 0.158 | 0.831 ± 0.338 | 1.155 ± 0.460 | 1.461 ± 0.581 | 1.652 ± 0.720 |
| 10 | 0.028 ± 0.009 | 0.083 ± 0.018 | 0.229 ± 0.044 | 0.387 ± 0.138 | 0.623 ± 0.242 | 0.801 ± 0.328 | 1.024 ± 0.464 | 1.246 ± 0.644 |
| 15 | 0.023 ± 0.003 | 0.082 ± 0.017 | 0.251 ± 0.078 | 0.450 ± 0.138 | 0.744 ± 0.297 | 0.976 ± 0.553 | 1.355 ± 1.041 | 1.720 ± 1.566 |
| persistence | 0.052 | 0.183 | 0.331 | 0.385 | 0.406 | 0.409 | 0.411 | 0.411 |

Pressure, vorticity, lag-minimised velocity error and kinetic-energy ratio (same aggregation):

| Quantity | K | h=1 | 10 | 50 | 100 | 200 | 299 | 450 | 599 |
|---|---|---|---|---|---|---|---|---|---|
| E_p | 5 | 0.748 | 0.485 | 0.602 | 0.814 | 1.258 | 1.823 | 2.024 | 2.090 |
| | 10 | 0.651 | 0.514 | 0.640 | 0.959 | 1.407 | 2.243 | 2.412 | 2.666 |
| | 15 | 0.595 | 0.409 | 0.572 | 0.780 | 1.265 | 1.997 | 2.571 | 2.929 |
| E_omega | 5 | 0.161 | 0.273 | 0.406 | 0.614 | 0.944 | 1.276 | 1.508 | 1.748 |
| | 10 | 0.172 | 0.301 | 0.404 | 0.586 | 0.933 | 1.279 | 1.966 | 2.865 |
| | 15 | 0.122 | 0.259 | 0.486 | 0.749 | 1.118 | 1.558 | 2.968 | 4.707 |
| E_u lag-min | 5 | 0.027 | 0.070 | 0.214 | 0.423 | 0.828 | 1.152 | 1.459 | 1.649 |
| | 10 | 0.028 | 0.066 | 0.207 | 0.373 | 0.618 | 0.798 | 1.021 | 1.245 |
| | 15 | 0.023 | 0.069 | 0.230 | 0.434 | 0.739 | 0.973 | 1.351 | 1.719 |
| KE_pred / KE_true | 5 | 0.994 | 1.008 | 1.179 | 1.496 | 2.613 | 3.990 | 5.746 | 7.188 |
| | 10 | 0.991 | 0.980 | 1.060 | 1.194 | 1.537 | 1.964 | 2.769 | 3.560 |
| | 15 | 0.997 | 1.028 | 1.192 | 1.440 | 2.129 | 2.969 | 5.241 | 8.708 |

Seed SDs for these rows are printed by `audit/rollout_diagnostics.py` and `audit/recompute_aggregates.py`. At h = 450 and 599 the K = 15 SDs exceed the means (vorticity 4.7 ± 6.4, KE ratio 8.7 ± 11.4) because one run (seed 1) is far worse than the others.

The pressure error at step 1 (0.60 to 0.75) is larger than at step 10 (0.41 to 0.51) and larger than the one-step pressure error averaged over the window (0.38 to 0.49). The first time levels lie in the start-up transient, where the one-step pressure prediction is worst. This is an observation, not a defect.

## 4. Independent recomputation from checkpoints

`audit/independent_rollout.py` repeats the rollout on CPU without the repository's data, rollout or metric code. It reads the raw TFRecords, builds the bidirectional edge list and normalised features from the run's stored statistics, loads the checkpoint into PhysicsNeMo's `MeshGraphNet`, feeds back its own predictions, and computes area-weighted errors with its own implementation.

Repeated: mp05_seed0 on all 10 test trajectories, and mp10_seed0 (the best run) and mp15_seed1 (the worst) on shedding trajectories 0 and 6. That is 14 of the 90 (run, trajectory) rollouts, run on a CPU shared with other jobs, 6 to 9 minutes per trajectory.

| Quantity | Largest difference, independent CPU vs stored MPS |
|---|---|
| E_u(t), all 599 steps, 14 rollouts | 9.7e-7 (absolute); at step 1 below 3e-9 |
| E_p(t), mp05_seed0 | 2.6e-6 |
| Predicted kinetic energy, mp05_seed0 | 1.3e-6 (relative) |
| Pooled one-step increment error, mp05_seed0 | 0.55451535 vs 0.55451536 stored |
| Velocity error on inflow and wall nodes | 3.0e-8 |

The stored curves are therefore reproducible from the checkpoints with independent code and on a second backend. Rollouts of this length did not amplify the float-level differences between CPU and MPS beyond 1e-6. **VERIFIED** for the rollouts repeated; the other 76 are covered by the aggregate recomputation only.

**Shedding at a probe** (`audit/probe_spectrum.py`). The probe is the interior node nearest to three diameters downstream of the cylinder centre, and the signal is the cross-stream velocity with the linear trend removed. The table gives the predicted power in the band 0.7 to 1.3 times the reference shedding frequency, as a fraction of the reference's power in that band, and the correlation with the reference:

| Run | Trajectories | Band power ratio, steps 100 to 299 | Band power ratio, steps 300 to 599 | Correlation |
|---|---|---|---|---|
| mp05_seed0 | 0, 1, 2, 4, 5, 6, 9 | 0.000 to 0.050 | 0.000 to 0.036 | -0.16 to +0.10, except +0.73 for trajectory 6 over steps 100 to 299 |
| mp10_seed0 | 0, 6 | 0.000 | 0.000 | -0.09 to +0.02 |
| mp15_seed1 | 0, 6 | 0.000 | 0.000 | -0.02 to +0.10 |

The predicted probe signals carry at most 5 % of the reference's power at the shedding frequency, and none after step 300 beyond 3.6 %. The dominant predicted period is the window length, a slow drift, instead of the reference's 25 to 60 steps. Trajectory 6 under mp05_seed0 tracks the reference oscillation for a while (correlation 0.73 over steps 100 to 299) and then loses it. In the rollouts examined the models do not shed vortices at the wrong phase or frequency: they stop shedding. This is consistent with the near-ineffectiveness of the time shift and with the plotted trajectory, and it is the main physical failure at this budget. The result covers 3 of 9 runs and 2 to 7 shedding trajectories each, so it is **SUPPORTED BUT LIMITED** for the study as a whole.

## 5. Error accumulation, phase drift, stability and physical plausibility

**Error accumulation.** All depths start near the one-step level (E_u = 0.023 to 0.028 at step 1, against 0.052 for persistence) and pass the persistence baseline between steps 78 and 100 in the seed mean. VERIFIED. Good one-step accuracy does not carry over: K = 15 has the lowest one-step increment and pressure errors in all 3 seeds, but the lowest rollout error over the window in 1 of 3 seeds. VERIFIED.

**Phase drift.** The lag-minimised error is the right tool only if a shift can align the phases. The seven shedding test trajectories have reference shedding periods of 25 to 60 steps at a probe three diameters downstream (`audit/data_checks.py`), so half a period is at most 30 steps, within the 40-step window. On these trajectories the best shift removes on average 9.7 % of the error over steps 10 to 99 (3.0 to 21.3 % across runs) and 1.2 % over steps 100 to 299 (0.3 to 3.1 %; the largest single-trajectory value is 9.2 %). On the three steady trajectories a shift cannot help by construction. Phase error is thus a visible part of the error only early in the rollout. The README conclusion "the error is not a phase error" was **SUPPORTED BUT LIMITED**, because it was based on averages that include steady cases. It is now qualified with the per-class figures and the caveat that the lag-minimised error cannot separate phase from amplitude when both are wrong.

**Stability.** No non-finite values occur in any of the 90 rollouts. VERIFIED. The averaged growth is gradual, but single trajectories grow strongly: the largest E_u at step 599 is 10.6 and the largest kinetic-energy ratio is 115 (mp15_seed1, test trajectory 7, a steady case with peak inflow 0.34). Five of nine runs have a trajectory whose energy exceeds the reference tenfold by step 599. The README statement "growth is gradual, not explosive" was **SUPPORTED BUT LIMITED** (true of the averages only) and is now qualified.

**Physical plausibility.** Energy rises on average: the seed-mean energy ratio at step 299 is 2.0 to 4.0, and 86 % of the 90 (run, trajectory) pairs exceed the reference energy at step 299. At step 10 the ratio is within 3 % of 1 and 46 % of the pairs are below the reference, so the README phrase "the models add energy at every step" was **INCORRECT** as a per-step statement and has been replaced. The shedding wake is not reproduced in the plotted trajectory (README figure). Lift, drag and Strouhal number are not computed. The README states the reason (no viscosity in the files, first-order gradients at the wall).

## 6. Persistence baseline

| Run | window-mean E_u | persistence | trajectories beating persistence |
|---|---|---|---|
| mp05_seed0 | 0.355 | 0.371 | 6 / 10 |
| mp10_seed0 | 0.291 | 0.371 | 8 / 10 |
| mp15_seed0 | 0.423 | 0.371 | 1 / 10 |
| mp05_seed1, mp10_seed1, mp15_seed1 | 0.815, 0.638, 0.843 | 0.371 | 0 / 10 each |
| mp05_seed2, mp10_seed2, mp15_seed2 | 0.703, 0.523, 0.451 | 0.371 | 0, 2, 1 / 10 |

Two of nine runs beat persistence in the window mean. Neither stays below it at every step: the trajectory-mean error crosses the baseline at step 176 (K = 5) and 274 (K = 10). The README sentence "stay below the fixed-state baseline" is now worded as a statement about the window mean. Persistence is a weak baseline for a shedding flow, and the learned models do not beat it reliably at this budget. The README conclusion "no depth gives a usable rollout at this budget" is **VERIFIED**.

## 7. Variability across seeds and trajectories

Run-level window-mean errors (rows K = 5, 10, 15; columns seeds 0, 1, 2):

```
K=5   0.355  0.815  0.703
K=10  0.291  0.638  0.523
K=15  0.423  0.843  0.451
```

Additive decomposition, descriptive only: seed effects -0.204, +0.205, -0.001; depth effects +0.064, -0.076, +0.012. Across (seed, trajectory) pairs, K = 10 has a lower window-mean error than K = 5 in 25 of 30 pairs (median difference -0.20), K = 15 than K = 5 in 14 of 30, and K = 15 than K = 10 in 9 of 30. These pairs are not independent (10 trajectories share each model, 3 seeds share data), so no significance is claimed. With 3 seeds, the consistent one-step ordering (15 < 10 < 5 in every seed) is the strongest depth effect in the data. The rollout ordering is not consistent across seeds.

Checkpoint noise is part of the seed spread. Each run is evaluated at its final step, and the validation loss changes by up to 53 % between the last two evaluations (mp10_seed1; for mp15_seed1 it is 0.053 at step 10,465 and 0.077 at step 11,960).
