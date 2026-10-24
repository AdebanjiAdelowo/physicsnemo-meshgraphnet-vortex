"""Independent checks on the raw cylinder_flow TFRecords (not through VortexSheddingDataset)."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from tfrecord.reader import tfrecord_loader

root = Path(sys.argv[1])
meta = json.loads((root / "meta.json").read_text())


def records(split, n):
    desc = {k: "byte" for k in meta["field_names"]}
    for i, rec in enumerate(tfrecord_loader(str(root / f"{split}.tfrecord"), None, desc)):
        if i >= n:
            return
        out = {}
        for k, spec in meta["features"].items():
            a = np.frombuffer(rec[k], dtype=spec["dtype"]).reshape(spec["shape"])
            out[k] = a
        yield out


def mesh_hash(r):
    return hashlib.sha256(r["mesh_pos"].tobytes() + r["cells"].tobytes()).hexdigest()[:16]


def areas(pos, cells):
    p = pos[cells].astype(np.float64)
    a = 0.5 * np.abs((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
    return a


hashes = {}
for split, n in (("train", 60), ("valid", 10), ("test", 20)):
    hashes[split] = []
    types, nodes, maxdrift, ke_steps = set(), [], 0.0, []
    for r in records(split, n):
        hashes[split].append(mesh_hash(r))
        nt = r["node_type"][0, :, 0]
        types |= set(np.unique(nt).tolist())
        nodes.append(len(nt))
        fixed = (nt == 4) | (nt == 6)
        u = r["velocity"]
        maxdrift = max(maxdrift, float(np.abs(u[:, fixed] - u[0, fixed]).max()))
    print(f"{split}: {n} records, node types {sorted(types)}, nodes {min(nodes)}-{max(nodes)}, "
          f"max |u(t)-u(0)| on inflow+wall nodes {maxdrift:.2e}, unique meshes {len(set(hashes[split]))}")
for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
    print(f"shared meshes {a}/{b}: {len(set(hashes[a]) & set(hashes[b]))}")

print("\nTest trajectories 0-9: geometry, unsteadiness and shedding period")
for i, r in enumerate(records("test", 10)):
    pos, cells, nt = r["mesh_pos"][0].astype(np.float64), r["cells"][0], r["node_type"][0, :, 0]
    u = r["velocity"].astype(np.float64)
    wall = pos[nt == 6]
    cyl = wall[(wall[:, 1] > 1e-6) & (wall[:, 1] < 0.41 - 1e-6)]
    centre = cyl.mean(axis=0)
    diam = 2 * np.linalg.norm(cyl - centre, axis=1).mean()
    area = areas(pos, cells).sum()
    expected = 1.6 * 0.41 - np.pi * diam**2 / 4
    umax = u[0][nt == 4, 0].max()
    # probe: interior node nearest to 3 diameters downstream on the centreline of the cylinder
    target = centre + np.array([3 * diam, 0.0])
    interior = np.where(nt == 0)[0]
    probe = interior[np.argmin(np.linalg.norm(pos[interior] - target, axis=1))]
    sig = u[300:, probe, 1] - u[300:, probe, 1].mean()
    amp = sig.std()
    spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
    freqs = np.fft.rfftfreq(len(sig), d=1.0)  # cycles per step
    k = np.argmax(spec[1:]) + 1
    period = 1.0 / freqs[k] if amp > 1e-3 else float("nan")
    print(f"traj {i}: D={diam:.3f} centre=({centre[0]:.2f},{centre[1]:.2f}) Umax_in={umax:.2f} "
          f"area/expected={area/expected:.4f} probe v' rms={amp:.3e} period={period:.1f} steps")
