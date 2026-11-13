# Training budget and long-horizon wake fidelity

Does increasing the number of Adam updates improve autoregressive accuracy and preserve
vortex shedding at fixed MeshGraphNet depth? The reduced depth study used 40 training
trajectories. Its 11,960-step runs cannot serve as the control for a 200-trajectory
experiment.

## Matched design

For each seed (0, 1, 2), train one MeshGraphNet with ten processor blocks and latent
width 128, and save its weights after exactly 11,960, 59,800 and 119,600 updates. The
first cutoff is 20% of the first shuffled epoch. Each longer cutoff continues the
same sample order and optimisation trajectory. The checkpoints are paired by seed.

Use the first 200 training, 10 validation and 50 test trajectories of DeepMind's
cylinder-flow dataset. Training and validation use the first 300 time levels. The
upstream batch size of 1, Adam with initial learning rate 0.0001 and decay multiplier
0.9999991 per update, normalisation, training noise and boundary conditions are
inherited from the existing PhysicsNeMo configuration. This estimates the effect of
additional updates under the fixed schedule, including the additional learning-rate
decay that accompanies them. It does not isolate the effect of changing the schedule.

## CUDA verification and costs

Use the [Kaggle notebook](../kaggle/training_budget.ipynb) with Internet access and a
T4 accelerator. It checks out a full Git SHA and installs the pinned PhysicsNeMo
revision in a virtual environment that excludes the base image’s optional RAPIDS
packages. It verifies PyTorch, PyG and CUDA sum aggregation through torch_scatter,
runs the tests and the existing CUDA smoke experiment, then times K=10 on one visible
GPU. The CPU-only torch_scatter build uses PyTorch's CUDA scatter-add for sum. The
probe reports synchronised throughput after 50 warm-up steps, over 300 measured
steps. It records CUDA peak allocated training memory, training and validation data
setup time, validation-pass time and 599-step inference time for one trajectory.

Gradient runtime at cutoff (s) is (s / r) seconds for measured throughput (r).
The notebook also projects validation, setup and evaluation overhead separately
from the gradient estimate and budgets for all three seed trajectories. These
projections are sensitive to the single probe mesh, machine load and CPU diagnostic
cost. A stage starts only if it fits the notebook's remaining time allowance.

Start with `STAGE = "probe"`. After successful verification, `STAGE = "seed0"`
trains through 119,600 and evaluates the extended and baseline snapshots. Evaluate
the intermediate cutoff and additional seeds with `STAGE = "followup"` if the seed-0
comparison warrants replication. Supply the Kaggle notebook version in the notebook
configuration for provenance.

## Checkpoints and outputs

Each seed writes step-numbered training states under `training_state/` and publishes
separate immutable model snapshots under `steps_*/mp10_seed*/`. Training states
store the model, optimiser, scheduler, AMP scaler, RNG state, data-loader generator
state and loss history. The state is written atomically. An interrupted invocation
reconstructs the same shuffled epoch and consumes its already completed prefix
before the next update. CUDA reduction kernels can introduce numerical variation
across otherwise equivalent executions. SHA-256 hashes of the actual dataset files are verified on
resume. Reuse the same Git SHA, device class, package versions and configuration.

For Kaggle resumption, attach the trusted prior notebook version's saved outputs and
set `RESUME_FROM` to its `training_budget` directory. Successful notebook versions
export that directory under `/kaggle/working`. A hard termination before Kaggle
publishes outputs can make that session's checkpoints inaccessible. The notebook
sets a wall-time limit for the training subprocess to give Kaggle time to publish.

The equivalent commands after CUDA verification and dataset setup are

```bash
python scripts/training_budget.py --mode train --seed 0
python scripts/training_budget.py --mode evaluate --seed 0 --cutoff 119600
python scripts/training_budget.py --mode evaluate --seed 0 --cutoff 11960
```

## Evaluation

Each model is evaluated over 599 autoregressive steps. The standard metric file
reports one-step velocity-increment and pressure errors, and velocity, pressure,
vorticity and kinetic-energy errors at steps 1, 10, 100, 299 and 599. The rollout
curves also contain the velocity persistence baseline, which holds the initial
velocity fixed. The wake analysis derives frequency, amplitude, phase, wake profile,
peak speed and energy drift from the same rollouts. The preserved-shedding count
uses the established criteria at the three-diameter probe. The predicted dominant
frequency must lie in the reference band and predicted band power must reach
25% of the reference. See [Long-horizon diagnostics](LONG_HORIZON_DIAGNOSTICS.md).

For the seed-0 comparison, report both endpoints on the same 50 trajectories and
separate gains in one-step error from changes in the shedding-preservation rate,
time to amplitude loss, extrapolation-window velocity error and step-599 energy
ratio. The evaluation command writes a per-seed comparison JSON with the endpoint
metrics and differences from the 11,960-step baseline when available. Treat any single-seed comparison as descriptive. Use seed-paired comparisons
for replication, with trajectories nested within seed.

## Evidence

The published depth-study results remain in [results/local](../results/local/).
The matched-data results below come from separate CUDA runs.

### Single-T4 benchmark

Kaggle notebook version 4 used commit `25cbdf0604e55f21729e7719fe8dd009f217bce7`
and one Tesla T4 (14.56 GiB visible). CUDA was available with PyTorch
2.15.0+cu132, PhysicsNeMo 2.3.0a0, PyG 2.8.1 and torch_scatter 2.1.2.
CUDA sum aggregation, the existing 38-step smoke training and checkpoint
reload, and smoke rollouts passed. Kaggle pytest reported 81 passed and
3 skipped. The isolated environment, headless plotting backend and explicit
CUDA allocator initialization resolved image and runtime compatibility failures.

| Measurement | Value |
|---|---:|
| Throughput after 50 warm-up steps, over 300 measured updates | 26.71 steps/s |
| Time per gradient step | 37.44 ms |
| Peak allocated GPU training memory | 468.76 MiB |
| Train and validation setup | 17.62 s |
| Validation pass | 4.09 s |
| One test trajectory setup | 0.14 s |
| 599-step probe rollout | 9.13 s |

Projected gradient time to 11,960, 59,800 and 119,600 steps was 7.46, 37.32
and 74.63 minutes, respectively. The notebook projected 1.89 hours for
seed-0 training and two evaluations and 6.35 hours for three seeds with
three evaluations each, excluding notebook setup. Seed-0 training actually
took 82.77 minutes through 119,600 steps, including 81.65 minutes of
optimisation at 24.41 steps/s. The two evaluations took about 12.75 minutes
each. The benchmark used one test trajectory, so extrapolations are approximate.

### Seed 0

Both checkpoints use the same 200 training trajectories and the same 50
test trajectories. Validation MSE fell from 0.1014 to 0.0421. The
intermediate training checkpoint has validation MSE 0.0762.

| Metric | 11,960 steps | 119,600 steps |
|---|---:|---:|
| One-step relative velocity-increment error | 0.480 | 0.291 |
| One-step relative pressure error | 0.571 | 0.251 |
| Mean relative velocity error, steps 300–599 | 1.712 | 0.366 |
| Mean predicted/reference kinetic-energy ratio, step 599 | 11.468 | 0.685 |
| Median step when in-band amplitude fell below half, censored at 600 | 76 | 362 |
| Shedding preserved, steps 100–299 | 0/36 | 25/36 |
| Shedding preserved, steps 300–599 | 0/36 | 19/36 |

Mean errors over the 50 trajectories. Each paired entry is baseline / extended.

| Step | Relative velocity | Relative pressure | Relative vorticity | Relative energy error | Velocity persistence |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.021 / 0.012 | 0.666 / 0.361 | 0.128 / 0.064 | 0.014 / 0.003 | 0.052 |
| 10 | 0.073 / 0.028 | 0.483 / 0.287 | 0.227 / 0.078 | 0.040 / 0.016 | 0.182 |
| 100 | 0.317 / 0.121 | 1.092 / 0.331 | 0.519 / 0.182 | 0.217 / 0.160 | 0.384 |
| 299 | 1.099 / 0.277 | 3.561 / 0.522 | 1.165 / 0.398 | 2.882 / 0.289 | 0.411 |
| 599 | 2.257 / 0.449 | 6.155 / 0.990 | 2.340 / 0.608 | 10.565 / 0.370 | 0.412 |

The longer-trained model improves one-step prediction and preserves shedding
in roughly half the reference shedding cases over steps 300–599. Its mean
velocity error at step 599 still exceeds persistence, and its mean energy
ratio indicates a deficit. All 100 evaluated rollouts remained finite.
These observations apply to seed 0 and the chosen 50 trajectories.

The [numerical outputs](../results/training_budget/) include per-trajectory
frequency and amplitude diagnostics, rollout curves, training histories,
dataset hashes and checkpoint checksums. The weights and training states
are available in the [Kaggle notebook output](https://www.kaggle.com/code/adebanjiadelowo/mgn-vortex-budget-t4-seed-0/output).
