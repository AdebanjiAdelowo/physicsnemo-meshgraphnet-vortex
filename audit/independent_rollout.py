"""Independent rollout of saved checkpoints on CPU.

Reads the raw TFRecords (not VortexSheddingDataset), builds the bidirectional mesh-edge
graph and its normalised features from the run's own edge_stats/node_stats, rolls the
loaded MeshGraphNet out autoregressively (and teacher-forced), and computes area-weighted
errors with its own implementation. Compares with the stored rollout_curves.npz.

usage: [TRAJ=0,2,6] independent_rollout.py <repo> <out.npz> <run> [<run> ...]
Runs from <repo>/runs/local (checkpoints are not tracked in git).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from physicsnemo.models.meshgraphnet import MeshGraphNet
from physicsnemo.utils import load_checkpoint
from tfrecord.reader import tfrecord_loader
from torch_geometric.data import Data

torch.set_num_threads(4)
repo, out_path, runs = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
SUBSET = [int(x) for x in os.environ.get("TRAJ", "").split(",") if x]
root = repo / "data/cylinder_flow"
meta = json.loads((root / "meta.json").read_text())
N_TRAJ, T = 10, 600
PROBE_TRAJ = [0, 1, 2, 4, 5, 6, 9]  # unsteady test trajectories (see data_checks.py)


def load_test():
    desc = {k: "byte" for k in meta["field_names"]}
    out = []
    for i, rec in enumerate(tfrecord_loader(str(root / "test.tfrecord"), None, desc)):
        if i >= N_TRAJ:
            break
        out.append({k: np.frombuffer(rec[k], dtype=s["dtype"]).reshape(s["shape"]) for k, s in meta["features"].items()})
    return out


def edges(cells):
    e = np.concatenate([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    return np.concatenate([e, e[:, ::-1]]).T  # both directions


def weights(pos, cells):
    p = pos[cells].astype(np.float64)
    a = 0.5 * np.abs((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
    w = np.zeros(len(pos))
    for k in range(3):
        np.add.at(w, cells[:, k], a / 3)
    return w


def rel(a, b, w):
    return np.sqrt(((a - b) ** 2).sum(-1) @ w) / np.sqrt((b**2).sum(-1) @ w)


test = load_test()
results = {}
for run in runs:
    K = int(run[2:4])
    run_dir = repo / "runs/local" / run
    ns = {k: torch.tensor(v) for k, v in json.loads((run_dir / "node_stats.json").read_text()).items()}
    es = {k: torch.tensor(v) for k, v in json.loads((run_dir / "edge_stats.json").read_text()).items()}
    model = MeshGraphNet(6, 3, 3, processor_size=K, mlp_activation_fn="relu")
    load_checkpoint(str(run_dir / "checkpoints"), models=model, device="cpu")
    model.eval()
    stored = np.load(run_dir / "rollout_curves.npz")
    t0 = time.perf_counter()
    for i, r in enumerate(test):
        if SUBSET and i not in SUBSET:
            continue
        pos, cells, nt = r["mesh_pos"][0], r["cells"][0], r["node_type"][0, :, 0]
        u_ref, p_ref = r["velocity"].astype(np.float64), r["pressure"].astype(np.float64)
        ei = torch.tensor(edges(cells), dtype=torch.long)
        disp = torch.tensor(pos[ei[0]] - pos[ei[1]])
        ea = (torch.cat([disp, disp.norm(dim=1, keepdim=True)], 1) - es["edge_mean"]) / es["edge_std"]
        code = np.where(nt == 0, 0, nt - 3)
        onehot = torch.nn.functional.one_hot(torch.tensor(code, dtype=torch.long), 4).float()
        graph = Data(edge_index=ei, num_nodes=len(pos))
        free = torch.tensor((nt == 0) | (nt == 5))[:, None]
        w = weights(pos, cells)

        def step(u):
            x = torch.cat([(u - ns["velocity_mean"]) / ns["velocity_std"], onehot], 1)
            with torch.no_grad():
                y = model(x, ea, graph)
            du = y[:, :2] * ns["velocity_diff_std"] + ns["velocity_diff_mean"]
            p = y[:, 2:] * ns["pressure_std"] + ns["pressure_mean"]
            return u + torch.where(free, du, torch.zeros_like(du)), p

        u = torch.tensor(r["velocity"][0])
        pred_u, pred_p = [], []
        for t in range(T - 1):
            u, p = step(u)  # the predicted velocity is the next input
            pred_u.append(u.numpy().astype(np.float64))
            pred_p.append(p.numpy().astype(np.float64))
        pred_u, pred_p = np.stack(pred_u), np.stack(pred_p)
        # teacher-forced one-step predictions from the reference state, every 5th step
        tf_err, tf_inc = 0.0, 0.0
        for t in range(0, T - 1, 5):
            if t + 1 > 299:
                break
            un, _ = step(torch.tensor(r["velocity"][t]))
            e = un.numpy().astype(np.float64) - u_ref[t + 1]
            tf_err += ((e**2).sum(-1) @ w)
            tf_inc += (((u_ref[t + 1] - u_ref[t]) ** 2).sum(-1) @ w)
        ev = rel(pred_u, u_ref[1:], w)
        epres = rel(pred_p, p_ref[1:], w)
        ke_p = 0.5 * (pred_u**2).sum(-1) @ w
        ke_t = 0.5 * (u_ref[1:] ** 2).sum(-1) @ w
        sv = stored["rel_l2_velocity"][i].astype(np.float64)
        results[f"{run}/{i}"] = dict(ev=ev, ep=epres, ke_p=ke_p, ke_t=ke_t, tf=(tf_err, tf_inc),
                                     boundary_max=float(np.abs(pred_u[:, ~free[:, 0].numpy()] - u_ref[1:, ~free[:, 0].numpy()]).max()))
        if i in PROBE_TRAJ:
            wall = pos[nt == 6].astype(np.float64)
            cyl = wall[(wall[:, 1] > 1e-6) & (wall[:, 1] < 0.41 - 1e-6)]
            c, d = cyl.mean(0), 2 * np.linalg.norm(cyl - cyl.mean(0), axis=1).mean()
            interior = np.where(nt == 0)[0]
            probe = interior[np.argmin(np.linalg.norm(pos[interior] - (c + [3 * d, 0]), axis=1))]
            results[f"{run}/{i}"]["probe_pred"] = pred_u[:, probe, 1]
            results[f"{run}/{i}"]["probe_true"] = u_ref[1:, probe, 1]
        d = np.abs(ev - sv)
        print(f"{run} traj {i}: |E_cpu - E_stored(mps)| step1 {d[0]:.1e}, max<=10 {d[:10].max():.1e}, max<=50 {d[:50].max():.1e}, "
              f"max<=299 {d[:299].max():.1e}, max all {d.max():.1e}; E299 cpu {ev[298]:.3f} mps {sv[298]:.3f}", flush=True)
    print(f"{run}: {time.perf_counter() - t0:.0f} s", flush=True)

flat = {}
for k, v in results.items():
    for kk, vv in v.items():
        flat[f"{k}/{kk}"] = np.asarray(vv)
np.savez_compressed(out_path, **flat)
