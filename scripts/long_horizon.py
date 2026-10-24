"""Regenerate the 599-step rollouts of a trained study and store compact wake diagnostics.

    python scripts/long_horizon.py --config local --threads 3 mp05_seed0 mp10_seed0 ...

No training. For every run the checkpoint in ``runs/<study>/<run>`` is rolled out on the CPU
with the same dataset, model loading, rollout and metric code as ``scripts/evaluate.py``. The
regenerated error and energy curves are compared with the stored ``rollout_curves.npz``; a run
whose curves disagree by more than ``--tolerance`` is not written. For each test trajectory the
script then keeps, for prediction and reference:

- (u, v, p) at the probes of ``diagnostics.PROBES`` (interior nodes nearest to points
  downstream of the cylinder, in diameters),
- the time-mean u and the RMS of the v fluctuation along the wake centreline for the
  windows of ``diagnostics.WINDOWS``,
- the unsteadiness (fluctuation ratio) of the wake region for the same windows,
- the largest nodal speed at every step.

Output: ``results/<study>/long_horizon/<run>.npz`` and ``reference.npz`` (the reference series,
identical for every run). Runs already written are skipped. Inference only; holds no GPU lock.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from _bootstrap import ROOT
from mgn_vortex import diagnostics, mesh_ops
from mgn_vortex.config import load_config
from mgn_vortex.data import build_dataset
from mgn_vortex.engine import load_trained_model
from mgn_vortex.evaluation import trajectory_metrics
from mgn_vortex.rollout import rollout

CHECKED = ("rel_l2_velocity", "rel_l2_pressure", "rel_l2_vorticity", "kinetic_energy_pred", "lagmin_rel_l2_velocity")


def wake_series(result: dict, geometry: dict) -> dict:
    """Compact series of one rollout for prediction (``pred_*``) and reference (``true_*``)."""
    pos, cells = result["mesh_pos"], result["cells"]
    weights = mesh_ops.nodal_areas(pos, cells)
    wake = pos[:, 0] > geometry["centre"][0] + 0.5 * geometry["diameter"]
    out = {}
    for name, field in (("pred", result["pred"]), ("true", result["exact"])):
        field = field.astype(np.float64)
        out[f"{name}_probe"] = field[:, geometry["probe_nodes"]].astype(np.float32)
        out[f"{name}_speed_max"] = np.linalg.norm(field[..., 0:2], axis=-1).max(axis=1)
        line = field[:, geometry["line_nodes"]]
        mean_u, rms_v, unsteady = [], [], []
        for lo, hi in diagnostics.WINDOWS.values():
            seg = line[lo - 1 : hi]
            mean_u.append(seg[..., 0].mean(axis=0))
            rms_v.append(seg[..., 1].std(axis=0))
            unsteady.append(diagnostics.window_unsteadiness(field[lo - 1 : hi, :, 0:2], weights, wake))
        out[f"{name}_line_mean_u"] = np.stack(mean_u)
        out[f"{name}_line_rms_v"] = np.stack(rms_v)
        out[f"{name}_wake_unsteadiness"] = np.asarray(unsteady)
    return out


def process_run(cfg, run: str, study_dir, out_dir, tolerance: float) -> None:
    run_dir = study_dir / run
    processor_size = int(run[2:4])
    device = torch.device("cpu")
    dataset = build_dataset(cfg, "test", run_dir)
    model = load_trained_model(cfg, processor_size, run_dir, device)
    stored = np.load(run_dir / "rollout_curves.npz")
    series, geometry_all, agreement = [], [], {name: 0.0 for name in CHECKED}
    tick = time.perf_counter()
    for traj in range(cfg.num_test_samples):
        result = rollout(model, dataset, traj, device, autoregressive=True)
        curves = trajectory_metrics(result, cfg.rollout.max_lag)
        for name in CHECKED:
            diff = np.abs(curves[name] - stored[name][traj].astype(np.float64))
            if name.startswith("kinetic"):
                diff = diff / np.abs(stored[name][traj].astype(np.float64))
            agreement[name] = max(agreement[name], float(np.nanmax(diff)))
        # one-hot node type of the graph: codes 0..3 are node types 0 (interior), 4, 5, 6 (wall)
        code = dataset[traj * (dataset.num_steps - 1)][0].x[:, 2:6].argmax(dim=1).numpy()
        geometry = diagnostics.wake_geometry(result["mesh_pos"], wall=code == 3, interior=code == 0)
        geometry_all.append(geometry)
        series.append(wake_series(result, geometry))
        print(f"{run} trajectory {traj}: {time.perf_counter() - tick:.0f} s, max |dE_u| so far {agreement['rel_l2_velocity']:.1e}", flush=True)
    worst = max(agreement.values())
    if worst > tolerance:
        raise RuntimeError(f"{run}: regenerated curves differ from rollout_curves.npz by {worst:.2e} > {tolerance:g}: {agreement}")
    pred = {key.removeprefix("pred_"): np.stack([s[key] for s in series]) for key in series[0] if key.startswith("pred_")}
    meta = {"run": run, "processor_size": processor_size, "device": "cpu", "torch_threads": torch.get_num_threads(),
            "seconds": time.perf_counter() - tick, "max_abs_difference_to_stored_curves": agreement}
    np.savez_compressed(out_dir / f"{run}.npz", **pred, meta=json.dumps(meta))
    reference = out_dir / "reference.npz"
    if not reference.exists():
        true = {key.removeprefix("true_"): np.stack([s[key] for s in series]) for key in series[0] if key.startswith("true_")}
        geo = {key: np.stack([g[key] for g in geometry_all]) for key in ("probe_nodes", "probe_pos", "line_nodes", "line_pos", "centre", "diameter")}
        tmp = out_dir / f"reference.{os.getpid()}.tmp.npz"  # several processes may finish together
        np.savez_compressed(tmp, **true, **geo, probe_names=np.asarray(list(diagnostics.PROBES)),
                            line_x_over_d=np.asarray(diagnostics.LINE_X_OVER_D), windows=json.dumps(diagnostics.WINDOWS), source_run=run)
        os.replace(tmp, reference)
    print(f"{run}: written, {meta['seconds']:.0f} s, agreement {agreement}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="local")
    parser.add_argument("--threads", type=int, default=3, help="CPU threads for torch")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="largest admissible difference to the stored curves")
    parser.add_argument("runs", nargs="*", help="run names; default: every run of the study")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    cfg = load_config(args.config, ["device=cpu"])
    study_dir = ROOT / "runs" / cfg.name
    out_dir = ROOT / "results" / cfg.name / "long_horizon"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = args.runs or sorted(p.name for p in study_dir.glob("mp*_seed*") if (p / "checkpoints").is_dir())
    for run in runs:
        if (out_dir / f"{run}.npz").exists():
            print(f"{run}: already written, skipped")
            continue
        process_run(cfg, run, study_dir, out_dir, args.tolerance)


if __name__ == "__main__":
    main()
