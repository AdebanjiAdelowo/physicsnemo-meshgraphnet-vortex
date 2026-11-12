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
T4 accelerator. It checks out a full Git SHA, installs the pinned PhysicsNeMo
revision, verifies PyTorch, PyG and CUDA sum aggregation through torch_scatter,
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
before the next update. SHA-256 hashes of the actual dataset files are verified on
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
ratio. Treat any single-seed comparison as descriptive. Use seed-paired comparisons
for replication, with trajectories nested within seed.

## Evidence

The published depth-study results remain in [results/local](../results/local/).
The budget snapshots require separate CUDA measurements and evaluation.
