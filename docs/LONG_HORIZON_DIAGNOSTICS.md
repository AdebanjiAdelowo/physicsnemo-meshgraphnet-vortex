# Long-horizon diagnostics of all nine runs

Vortex-shedding, phase, wake, energy and threshold diagnostics of the 599-step rollouts of the `local` depth study (K = 5, 10, 15 message-passing steps; seeds 0, 1, 2), applied with the same probes, estimators, windows and normalisation to every run. The earlier audit had applied a probe-spectrum check to 3 of the 9 runs (`ROLLOUT_VERIFICATION.md`, section 4); this document completes it for all 9 and adds the other diagnostics.

**Reduced-budget qualification.** Every model was trained for 11,960 gradient steps of batch 1 on 40 trajectories: 0.4 % of the sample presentations of NVIDIA's configuration and 3.2 % of the optimiser steps of NVIDIA's 8-GPU run, at a learning rate still at 0.99 of its initial value. None of the models is converged. Everything below describes this early-training regime, not MeshGraphNet at convergence.

Generated files: [`results/local/long_horizon/tables.md`](../results/local/long_horizon/tables.md) (all tables, every metric), `summary.json` (per trajectory, per run, per depth), the probe and wake series `mp*_seed*.npz` and `reference.npz`, and the figures `figures/local_lh_*.png`. Scripts: [`scripts/long_horizon.py`](../scripts/long_horizon.py) (inference) and [`scripts/long_horizon_analysis.py`](../scripts/long_horizon_analysis.py) (analysis, tracked files only). Estimators: [`src/mgn_vortex/diagnostics.py`](../src/mgn_vortex/diagnostics.py). Tests: `tests/test_diagnostics.py`, `tests/test_long_horizon.py`.

## 1. Source of the trajectories

The tracked results hold per-step error and energy curves (`rollout_curves.npz`), not predicted fields, so probe signals did not exist for any run. They were regenerated without training: each run's final checkpoint was rolled out again on the CPU through the repository's own dataset, model-loading, rollout and metric code (the code that produced the stored curves), from the same initial conditions of the same 10 test trajectories.

| Check | Result |
|---|---|
| Regenerated curves against the stored MPS curves, 9 runs x 10 trajectories x 599 steps | largest difference 9.3e-6 in E_u, 3.5e-5 in E_p, 1.6e-5 in E_omega, 1.8e-5 relative in kinetic energy (`tables.md`, regeneration check). A run exceeding 1e-4 would not have been written |
| Earlier 3-run probe result (`ROLLOUT_VERIFICATION.md` section 4) | reproduced exactly: band power ratios 0.000 to 0.050 (steps 100 to 299) and at most 0.036 (300 to 599); correlation +0.73 for trajectory 6 of mp05_seed0 (`tests/test_long_horizon.py`) |
| Device | CPU, 3 threads per process, 3 processes; 13.6 to 21.6 minutes per run, about 50 minutes wall time. The shared accelerator lock (`~/.cache/physicsnemo/gpu.lock`) was held by another project, so MPS was not used |

The stored curves, `eval_metrics.json`, `summary.json`, dataset splits and rollout protocol are unchanged. Only compact series are stored (5 MB): (u, v, p) at 7 probes, the peak nodal speed per step, and centreline and wake-region statistics per window.

## 2. Protocol

| Element | Definition (identical for all runs) |
|---|---|
| Trajectory classes | shedding: reference probe v' RMS > 1e-3 over steps 300 to 599, trajectories 0, 1, 2, 4, 5, 6, 9 (reference periods 26 to 57 steps); steady: 3, 7, 8. Same as `audit/data_checks.py` |
| Probes | interior node nearest to the cylinder centre plus (1, 2, 3, 5, 8) D on the centreline and 3 D at y = +-0.5 D. Main probe: 3 D on the centreline (the probe of the earlier audit). Centreline profile: 0.75 to 12 D in steps of 0.25 D |
| Windows | early 1 to 99, late training window 100 to 299, extrapolation 300 to 599 |
| Spectral estimator | cross-stream velocity v, linear trend removed, Hann window, periodogram; reference frequency f_ref = peak of the reference in the same window; band 0.7 to 1.3 f_ref. Band power ratio = predicted / reference power in the band. Dominant frequency from the zero-padded periodogram with parabolic refinement. Not applied to steps 1 to 99, where the slow trajectories hold fewer than two periods and the reference peak itself is wrong (66 to 132 steps against 44 to 57) |
| Shedding preserved | predicted dominant frequency within 0.7 to 1.3 f_ref and band power ratio >= 0.25 |
| Phase and in-band amplitude over time | complex demodulation at f_ref (period from reference steps 100 to 599), zero-phase low-pass at 0.3 f_ref. Envelope ratio = predicted / reference in-band amplitude; phase error = phase difference, reported only while the envelope ratio is >= 0.5. Undefined in the first and last reference period (filter edges). Accuracy on pure sinusoids: amplitude within 10 %, phase within 5 degrees (tests) |
| Wake | wake-region unsteadiness (area-weighted RMS fluctuation / RMS velocity, nodes downstream of x_c + D/2), centreline RMS v' and time-mean u per window |
| Thresholds | first step with E_u > 0.1, 0.25, 0.5, 1; E_u above the persistence error; E_p > 0.5, 1; kinetic-energy ratio outside [1/1.5, 1.5] and above 2; one-period zero-lag correlation < 0.5; envelope ratio < 0.5. 600 = not reached |
| Run value | mean over trajectories for errors and ratios; median for threshold times, spectral ratios and the peak-speed ratio (robust to the few trajectories with broadband growth) |
| Uncertainty | seed is the unit of replication. Per depth: mean, SD and 95 % t-interval over 3 seeds (t = 4.30, 2 degrees of freedom). Depth effect: within-seed least-squares slope per 5 layers and paired differences, t-intervals over seeds, number of seeds with each sign. A two-level bootstrap (seeds, then trajectories) is stored as descriptive only; with 3 seeds it understates the uncertainty. Associations: Spearman over 9 runs with an exact within-seed permutation p (216 relabellings) |

## 3. Results

Values: mean ± SD over 3 seeds [95 % t-interval]. Slope: change per 5 message-passing steps [95 % t-interval] (seeds with negative / positive slope). The full set is in `tables.md`.

### 3.1 Autoregressive velocity and pressure error

| Quantity | K = 5 | K = 10 | K = 15 | Slope per 5 layers |
|---|---|---|---|---|
| E_u, steps 1-99 | 0.236 ± 0.067 | 0.224 ± 0.052 | 0.249 ± 0.074 | +0.007 [-0.065, 0.078] (1/2) |
| E_u, steps 100-299 | 0.817 ± 0.326 | 0.613 ± 0.238 | 0.733 ± 0.315 | -0.042 [-0.333, 0.249] (1/2) |
| E_u, steps 300-599 | 1.44 ± 0.58 [-0.01, 2.89] | 1.02 ± 0.47 [-0.14, 2.18] | 1.35 ± 1.05 [-1.26, 3.96] | -0.05 [-1.16, 1.07] (2/1) |
| E_u, steps 300-599, shedding | 1.46 ± 0.65 | 0.88 ± 0.46 | 1.06 ± 0.65 | -0.20 [-1.44, 1.03] (2/1) |
| E_u, steps 300-599, steady | 1.40 ± 1.13 | 1.36 ± 0.49 | 2.03 ± 2.00 | +0.32 [-0.77, 1.40] (0/3) |
| E_p, steps 1-99 | 0.622 ± 0.081 | 0.671 ± 0.155 | 0.574 ± 0.045 | -0.024 [-0.079, 0.031] (2/1) |
| E_p, steps 100-299 | 1.31 ± 0.38 | 1.52 ± 0.73 | 1.30 ± 0.76 | -0.00 [-0.49, 0.49] (2/1) |
| E_p, steps 300-599 | 1.98 ± 0.76 | 2.42 ± 1.52 | 2.50 ± 2.34 | +0.26 [-1.71, 2.23] (2/1) |
| Growth of log E_u per 100 steps, 100-299 | 0.442 ± 0.032 | 0.313 ± 0.111 | 0.325 ± 0.111 | -0.059 [-0.179, 0.061] (3/0) |
| Growth of log E_u per 100 steps, 300-599 | 0.098 ± 0.039 | 0.116 ± 0.076 | 0.113 ± 0.084 | +0.008 [-0.132, 0.147] (2/1) |

The error grows fastest between steps 100 and 299 (a factor of about 1.4 to 1.6 per 100 steps) and slows during the extrapolation (about 1.1 per 100 steps), where many runs have saturated well above the persistence error (0.41). The pressure error exceeds 0.5 at step 1 in every run (the start-up transient, as noted in the audit).

A data feature, not a model effect: the reference pressure of test trajectory 7 (steady, peak inflow 0.34) jumps by 12 % in norm between time levels 326 and 327 while its velocity changes smoothly (mean pressure 0.0058 to 0.0080). Every run's E_p for this trajectory drops at that step, which shows as a small step in the pressure-error curves near step 327 (`figures/local_lh_errors_energy.png`).

### 3.2 Vortex-shedding frequency and spectral power

| Quantity (3 D probe, shedding trajectories, run median) | K = 5 | K = 10 | K = 15 |
|---|---|---|---|
| Shedding preserved, steps 100-299 / 300-599 (share) | 0 / 0 | 0 / 0 | 0 / 0 |
| Band power ratio, steps 100-299 | 0.0092 ± 0.0078 | 0.0035 ± 0.0031 | 0.0031 ± 0.0036 |
| Band power ratio, steps 300-599 | 0.0001 ± 0.0001 | 0.0030 ± 0.0037 | 0.0003 ± 0.0003 |
| Dominant frequency / f_ref, steps 100-299 | 0.22 ± 0.06 | 0.20 ± 0.01 | 0.23 ± 0.10 |
| Dominant frequency / f_ref, steps 300-599 | 0.13 ± 0.02 | 0.19 ± 0.05 | 0.19 ± 0.05 |
| Share of predicted fluctuation power in the band, 300-599 | 0.005 ± 0.004 | 0.031 ± 0.040 | 0.043 ± 0.059 |
| Probe RMS ratio, steps 100-299 | 0.27 ± 0.12 | 0.22 ± 0.11 | 0.13 ± 0.06 |
| Pressure band power ratio, steps 300-599 | 0.0018 ± 0.0014 | 0.022 ± 0.030 | 0.0074 ± 0.0066 |

None of the 63 (run, shedding trajectory) pairs preserves shedding after step 100 in either window. After step 300 no predicted dominant frequency lies in the band (largest 0.55 f_ref); between steps 100 and 299 three pairs have a dominant frequency in the band but keep less than 25 % of the band power. The predicted probe signals are dominated by slow drift at 0.1 to 0.3 f_ref (figure `local_lh_spectra.png`). Band power ratios above 1 occur in 2 pairs (3.3 and 1.7), both on trajectory 4, the weakest shedding case (fluctuation ratio 0.07); there the predicted signal has 4.5 and 5.7 times the reference RMS and at most 17 % of its own power in the band, so the ratio measures broadband growth leaking into the band, not shedding. This is why run values use medians. The same holds at every probe from 1 D to 8 D downstream and off the centreline (`tables.md`, downstream wake).

### 3.3 Phase relative to the reference

| Quantity (3 D probe, shedding trajectories) | K = 5 | K = 10 | K = 15 |
|---|---|---|---|
| First step with in-band amplitude < 0.5 of the reference (run median) | 53 ± 7 [35, 72] | 52 ± 6 [36, 67] | 47 ± 2 [42, 52] |
| Share already below 0.5 one period into the rollout | 0.38 ± 0.16 | 0.57 ± 0.14 | 0.48 ± 0.22 |
| Share still shedding at step 50 (of those measurable) | 0.47 ± 0.23 | 0.40 ± 0.20 | 0.13 ± 0.12 |
| Share still shedding at steps 100 and 200 | 0 | 0 | 0 |
| First step with one-period zero-lag correlation < 0.5 (run median) | 55 ± 7 | 46 ± 3 | 51 ± 7 |

Across the 63 pairs the in-band amplitude falls below half the reference at a median step of 53 (range 27 to 96, interquartile range 41 to 59), and in 48 % of them it is already below half at the first step where it can be measured, one reference period in. At step 100 the largest in-band amplitude ratio is 0.43 and the median 0.09. A phase error is defined only while shedding persists: at step 50 this leaves 15 of 45 measurable pairs, with a median absolute phase error of 95 degrees (8 above 90 degrees, 2 below 30). The predictions therefore lose phase coherence before or while they lose amplitude, and the shedding is lost within the first 30 to 100 steps, typically one to two shedding periods. After that a phase is undefined: there is no predicted oscillation to compare. This sharpens the audit's finding that the lag-minimised error removes about 10 % of the error before step 100 and 1 % after it.

### 3.4 Downstream wake

| Quantity (shedding trajectories unless stated) | K = 5 | K = 10 | K = 15 | Slope per 5 layers |
|---|---|---|---|---|
| Centreline RMS v' ratio, steps 100-299 | 1.28 ± 0.87 | 0.59 ± 0.15 | 0.39 ± 0.09 | -0.44 [-1.41, 0.52] (3/0) |
| Centreline RMS v' ratio, steps 300-599 | 0.87 ± 0.28 | 0.76 ± 0.45 | 0.63 ± 0.34 | -0.12 [-0.33, 0.09] (3/0) |
| Centreline mean-u relative error, steps 300-599 (all trajectories) | 2.66 ± 1.05 | 1.37 ± 0.64 | 1.77 ± 0.75 | -0.44 [-2.30, 1.41] (2/1) |
| Wake unsteadiness ratio, steps 100-299 | 1.80 ± 0.49 | 1.44 ± 0.53 | 1.24 ± 0.19 | -0.28 [-1.11, 0.55] (3/0) |
| Wake unsteadiness ratio, steps 300-599 | 1.05 ± 0.23 | 1.80 ± 0.78 | 1.42 ± 0.43 | +0.18 [-0.65, 1.01] (1/2) |
| Band power ratio, steps 300-599, probe 1 D / 8 D | 0.0004 / 0.0001 | 0.0002 / 0.0010 | 0.0003 / 0.0030 | n/a |

The predicted wakes are not quiescent: their wake-region unsteadiness is 1.0 to 1.8 times the reference's, and the centreline RMS of v' is 0.4 to 1.3 times. But that fluctuation is slow, broadband drift, not a vortex street. The time-mean centreline velocity is the clearest structural error: the reference has a recirculation region behind the cylinder and a recovering wake deficit (mean u from about -0.3 near the cylinder to 1.2 at 12 D), while the predicted wakes carry a mean u 1.7 to 3 times larger and lose the deficit (median relative error 1.73 over the shedding trajectories; `local_lh_wake_profiles.png`). This is the same excess flow that appears as the kinetic-energy drift. The deeper models have a lower centreline v' RMS ratio between steps 100 and 299 in all 3 seeds, which, given the absence of shedding, means slightly less spurious fluctuation, not better shedding.

### 3.5 Kinetic-energy drift and numerical instability

| Quantity | K = 5 | K = 10 | K = 15 |
|---|---|---|---|
| KE ratio at step 99 | 1.49 ± 0.42 | 1.19 ± 0.29 | 1.43 ± 0.62 |
| KE ratio at step 299 | 3.99 ± 1.89 | 1.96 ± 0.49 | 2.97 ± 2.49 |
| KE ratio at step 599, mean / median over trajectories | 7.19 / 5.45 | 3.56 / 2.79 | 8.71 / 4.15 |
| Growth of log KE ratio per 100 steps, 300-599 | 0.125 ± 0.053 | 0.105 ± 0.185 | 0.121 ± 0.117 |
| Largest peak-speed ratio (run median) | 3.1 ± 1.5 | 3.3 ± 1.9 | 5.3 ± 6.1 |
| Trajectories with KE ratio > 10 (of 30) | 5 | 3 | 4 |

No rollout produced non-finite values. The energy drift is not monotone early (72 % of the 90 pairs are above the reference at step 99) and is upward late (86 % at step 599). It continues during the extrapolation at about 11 to 13 % per 100 steps in log terms, at every depth. Single trajectories reach peak nodal speeds 3 to 5 times the reference's largest speed and energies above 10 times the reference (12 of 90 pairs). This is a slow, unbounded drift, not a blow-up: errors stay finite and growth slows after step 300.

### 3.6 Time to exceed error thresholds

| First step (run median over trajectories) | K = 5 | K = 10 | K = 15 | Slope per 5 layers |
|---|---|---|---|---|
| E_u > 0.1 | 17 ± 8 | 17 ± 7 | 19 ± 8 | +1 [-9, 12] (1/2) |
| E_u > 0.25 | 69 ± 40 | 84 ± 55 | 50 ± 15 | -9 [-47, 28] (2/1) |
| E_u above persistence | 108 ± 61 | 255 ± 282 | 89 ± 33 | -10 [-62, 43] (2/1) |
| E_u > 0.5 | 147 ± 88 | 285 ± 274 | 148 ± 58 | +1 [-88, 89] (2/1) |
| E_u > 1 | 342 ± 225 | 452 ± 146 | 462 ± 240 | +60 [-195, 315] (0/3) |
| E_p > 1 | 208 ± 99 | 312 ± 256 | 418 ± 232 | +105 [-150, 359] (0/3) |
| KE ratio outside [1/1.5, 1.5] | 166 ± 142 | 308 ± 229 | 203 ± 133 | +19 [-142, 179] (2/1) |
| In-band amplitude < 0.5 (shedding trajectories) | 53 ± 7 | 52 ± 6 | 47 ± 2 | -3 [-14, 8] (2/1) |

The ordering of events is the same at every depth: shedding is lost first (about step 50), the velocity error passes 0.1 at step 17 to 19 and 0.25 by step 50 to 84, the error passes the persistence baseline around step 90 to 110 (step 580 for one seed-0 run at K = 10), and energy and large-error thresholds follow over hundreds of steps. The seed spread of every time except the first and the shedding loss is comparable to its mean. The large-threshold times (E_u > 1, E_p > 1) are later for deeper models in all 3 seeds, but their intervals span several hundred steps either side of zero.

### 3.7 One-step accuracy against long-horizon behaviour

| One-step increment error against | Spearman rho (9 runs) | within-seed permutation p |
|---|---|---|
| E_u, steps 100-299 | +0.70 | 0.33 |
| E_u, steps 300-599 | +0.77 | 0.16 |
| first step above persistence | -0.63 | 0.50 |
| KE ratio at step 299 | +0.82 | 0.10 |
| growth of log E_u, steps 100-299 | +0.82 | 0.08 |
| band power ratio, steps 100-299 | +0.62 | 0.25 |
| first step with in-band amplitude < 0.5 | +0.10 | 0.80 |

Across the 9 runs, lower one-step error goes with lower long-horizon error and drift (rho 0.7 to 0.8). Almost all of that association is between seeds, not between depths: when depth labels are permuted within each seed, correlations as large arise in 8 to 50 % of relabellings, and the one-step pressure error, which has the clearest depth effect, shows weaker associations (rho 0.4 to 0.6, p >= 0.07). Seed 0, which has the lowest one-step errors at every depth, also has the lowest rollout errors. One-step accuracy does not predict the loss of shedding at all (rho 0.10). Read together: the seed (initial weights, noise realisation, sample order, and where the final checkpoint fell) moves one-step and rollout accuracy together; depth moves one-step accuracy without moving the rollout (`local_lh_onestep_vs_rollout.png`).

## 4. Depth effects

| Question | Evidence at this budget | Answer |
|---|---|---|
| Do deeper networks improve one-step prediction? | velocity increment error slope -0.016 [-0.042, 0.011] per 5 layers, negative in 3/3 seeds; pressure error slope -0.054 [-0.076, -0.031], 3/3 seeds, the one depth slope whose interval excludes zero | Yes, consistently in sign and clearly for pressure; small (about 5 % in velocity increment, 22 % in pressure from K = 5 to 15) |
| Improve long-horizon rollout accuracy? | E_u slope over steps 300-599 -0.05 [-1.16, 1.07], sign 2/1; K = 10 lowest in the mean (K10 - K5 paired -0.42 [-0.77, -0.08], 3/3 seeds) but K = 15 is not | No monotone effect. The K = 10 versus K = 5 difference is one of 10 intervals clear of zero among 252 computed (63 metrics, slopes and 3 pairwise differences each), fewer than the 13 expected by chance at the 95 % level; it is consistent with the earlier audit but not evidence of an optimum |
| Preserve vortex shedding? | 0 of 63 pairs preserve shedding at any depth; shedding lost at step 53, 52, 47 | No, at every depth |
| Delay instability? | no non-finite values at any depth; KE ratio > 10 in 5, 3, 4 trajectories; energy-drift rate 0.125, 0.105, 0.121 per 100 steps; E_u > 1 and E_p > 1 later for deeper models in 3/3 seeds with intervals of several hundred steps | No detectable effect; the late-threshold ordering is suggestive only |
| Change error accumulation? | log E_u growth over steps 100-299 lower for K = 10 and 15 than K = 5 in 3/3 seeds (slope -0.059 [-0.179, 0.061]); no difference over steps 300-599 | Possibly slower mid-rollout growth with depth; not resolved with 3 seeds |

No optimal depth can be declared. The seed effect on rollout metrics is larger than the depth effect for almost every quantity (for example window-mean error: seed effects ±0.20, depth effects ±0.08, `ROLLOUT_VERIFICATION.md` section 7), and the 95 % t-intervals over 3 seeds are wide (half-width 2.5 SD). The design has the power to detect the one-step pressure effect and nothing of the size seen in the rollout metrics.

## 5. Approximation accuracy and dynamical fidelity

The two properties separate cleanly in this study.

- **Approximation accuracy** (one step from the true state) improves with depth: the pressure error falls from 0.488 to 0.381 and the velocity-increment error from 0.581 to 0.550, consistently across seeds. This is what extra message-passing capacity should buy.
- **Dynamical fidelity** (the rollout as a dynamical system) does not. Every model at every depth fails in the same way, in the same order and at about the same time: the vortex street decays within one to two shedding periods (median step 53), the probe signals turn into slow drift at 0.1 to 0.3 times the shedding frequency, the time-mean wake loses its deficit and gains energy, and the field error passes the persistence baseline around step 100 and keeps growing slowly without blowing up. The learned map has the wrong attractor: a drifting, energy-gaining non-shedding flow instead of the limit cycle of the reference.

An increment error of 0.55 to 0.58 means more than half of the true one-step change is wrong. A periodic wake is sustained by a delicate balance of the increment's oscillatory part; an error of this size in that part damps the oscillation within a few periods regardless of how the remaining error is distributed. The 5 % to 20 % improvements that depth buys in one-step error are therefore too small to cross into a regime where shedding survives. Whether more training or more capacity would cross it is not answerable from these data (section 7).

## 6. Missing artefacts and incomplete evaluations

| Item | Status |
|---|---|
| Predicted fields of the 90 rollouts | not stored (about 1.1 GB as float32); regenerated on demand from the checkpoints. The compact series used here are tracked |
| Lift, drag, Strouhal number | not computed: viscosity and density are not in the files and the vorticity is first-order at the wall. The probe frequency in units of steps (f_ref = 1/26 to 1/57 per step) is used instead |
| Spectra in steps 1-99 | not computed (fewer than two periods for the slow trajectories); the demodulated envelope covers this window from one period on |
| Phase before one reference period (26 to 57 steps) | undefined (filter edge). In 48 % of shedding pairs shedding is already lost when the first value is available, so their loss time is left-censored at one period |
| Intermediate checkpoints | each run has one checkpoint (end of its single epoch); checkpoint noise cannot be separated from seed effects (gap G4) |
| Test set | first 10 test trajectories; 7 shedding. Spectral statements rest on 7 trajectories x 3 seeds per depth |
| CUDA evaluation | not run; every number is from MPS (stored) and CPU (regeneration) |

## 7. Is additional training scientifically necessary?

Yes, for any statement about depth beyond this budget. The present data establish that at 11,960 steps no depth reproduces shedding, and that depth improves one-step accuracy without changing the rollout. They cannot tell whether the failure is under-training (validation loss still falling, learning rate at 0.99 of its initial value, one-step increment error above 0.5) or a capacity or architecture limit. Only a budget-scaling experiment at fixed depth answers that (gap G1). Without it, more seeds or more test trajectories at the same budget would narrow intervals around a regime in which every model fails, which has little scientific value. More diagnostics on the existing nine runs are not needed: they agree with each other and with the earlier three-run check.

## 8. Protocol and resources for a future full-budget study

Priority P1 of `EXPERIMENT_GAPS.md`, updated with the diagnostics above.

**Design.** K = 5 and K = 15 (the extremes; K = 10 is added only if P1 shows a depth effect), seeds 0, 1, 2, trained on 400 trajectories for 3 epochs (358,800 optimiser steps of batch 1, close to NVIDIA's 373,750), with checkpoints at the end of each epoch (119,600, 239,200, 358,800 steps). Evaluate every checkpoint on the first 50 test trajectories with 599-step rollouts, and run `scripts/long_horizon.py` and `scripts/long_horizon_analysis.py` on each with the probes, windows, thresholds and statistics fixed above. Pre-registered primary outcomes: (1) share of shedding trajectories with shedding preserved over steps 300 to 599; (2) first step with in-band amplitude < 0.5; (3) window-mean E_u over steps 300 to 599; (4) KE ratio at step 599. Secondary: all other quantities in `tables.md`.

**Decision rules.** A depth effect is claimed only if the within-seed paired difference has the same sign in all seeds and its t-interval excludes zero for a primary outcome; with 3 seeds that requires a large, consistent effect. If the outcomes differ between depths but the intervals overlap zero, add seeds (P5, 5 to 10) before any claim. The budget effect is read from the three checkpoints of each run.

**Required code change.** `engine.load_trained_model` loads the latest checkpoint; evaluating the per-epoch checkpoints needs an `epoch` argument passed to `load_checkpoint` (not made here; the `local` study has one checkpoint per run). `scripts/long_horizon.py` should then take that argument and write to a per-checkpoint directory.

**Resources** (MPS-equivalent; CUDA speed-up f unmeasured, roughly 1 to 4 for these launch-bound batch-1 graphs; replace with `scripts/probe.py` on the target GPU before starting):

| Item | Estimate |
|---|---|
| Training | 358,800 steps x (37.7 + 104.0) ms x 3 seeds = 42.4 h MPS-equivalent, / f on CUDA |
| Standard evaluation | 3 checkpoints x 3 seeds x 50 trajectories x 599 steps x (17.5 + 44.5) ms = 4.6 h of rollouts, about 5.6 h with the one-step evaluation |
| Long-horizon regeneration | on the training GPU, the same order as the standard evaluation (about 3 h); on a laptop CPU about 1.4 min (K = 5) to 2.1 min (K = 15) per trajectory per process, about 9 h for the 900 rollouts with 3 processes. Better: store probe series inside `evaluate_run` to avoid a second rollout |
| Data | 400 training, 10 validation and 50 test trajectories, about 6.0 GB on disk at the 13 MB per trajectory of the current files (download needs approval: over 500 MB) |
| Outputs | checkpoints 18 x 10 MB; tracked probe series about 50 MB (900 rollouts at 0.06 MB each) |

**Commands** (on the CUDA machine, after the probe confirms the budget):

```bash
python scripts/download_data.py --train 400 --valid 10 --test 50
python scripts/probe.py --config full --budget-hours <H>
python scripts/train.py --config full num_training_samples=400 epochs=3 "study.processor_sizes=[5,15]"
python scripts/long_horizon.py --config full --threads <n>      # after the checkpoint-epoch change
python scripts/long_horizon_analysis.py --study results/full
```

The campaign has not been started. It needs explicit approval, a dedicated GPU, and the data download.
