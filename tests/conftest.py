"""A small synthetic dataset in the format of DeepMind's ``cylinder_flow`` files.

The tests never download data. The fixture writes ``meta.json`` and three TFRecord files
with a few trajectories on structured triangular meshes of different sizes, with the same
field names, dtypes, node-type codes and byte layout as the real files.
"""

import json

import numpy as np
import pytest
import tfrecord

from mgn_vortex.config import load_config

STEPS = 12
NODE_NORMAL, NODE_INFLOW, NODE_OUTFLOW, NODE_WALL = 0, 4, 5, 6


def make_mesh(nx: int, ny: int):
    x, y = np.meshgrid(np.linspace(0.0, 1.6, nx), np.linspace(0.0, 0.4, ny), indexing="ij")
    pos = np.stack([x.ravel(), y.ravel()], axis=1).astype(np.float32)
    idx = np.arange(nx * ny).reshape(nx, ny)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    cells = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)]).astype(np.int32)
    node_type = np.full((nx, ny), NODE_NORMAL, dtype=np.int32)
    node_type[:, 0] = node_type[:, -1] = NODE_WALL
    node_type[-1, 1:-1] = NODE_OUTFLOW
    node_type[0, 1:-1] = NODE_INFLOW
    return pos, cells, node_type.reshape(-1, 1)


def make_trajectory(nx: int, ny: int, phase: float):
    pos, cells, node_type = make_mesh(nx, ny)
    t = 0.01 * np.arange(STEPS)[:, None]
    x, y = pos[None, :, 0], pos[None, :, 1]
    profile = 6.0 * y * (0.4 - y) / 0.16
    u = profile * (1.0 + 0.2 * np.sin(8.0 * t + 3.0 * x + phase))
    v = 0.1 * profile * np.cos(8.0 * t + 3.0 * x + phase)
    fixed = (node_type[:, 0] == NODE_INFLOW) | (node_type[:, 0] == NODE_WALL)
    u[:, fixed] = u[0, fixed]  # boundary values constant in time, as in the real data
    v[:, fixed] = v[0, fixed]
    velocity = np.stack([u, v], axis=-1).astype(np.float32)
    pressure = ((1.6 - x) * (1.0 + 0.1 * np.sin(8.0 * t + phase)))[..., None].astype(np.float32)
    return {"cells": cells[None], "mesh_pos": pos[None], "node_type": node_type[None], "velocity": velocity, "pressure": pressure}


def write_split(path, trajectories):
    writer = tfrecord.TFRecordWriter(str(path))
    for traj in trajectories:
        writer.write({k: (v.tobytes(), "byte") for k, v in traj.items()})
    writer.close()


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("cylinder_flow")
    meta = {
        "simulator": "synthetic",
        "dt": 0.01,
        "features": {
            "cells": {"type": "static", "shape": [1, -1, 3], "dtype": "int32"},
            "mesh_pos": {"type": "static", "shape": [1, -1, 2], "dtype": "float32"},
            "node_type": {"type": "static", "shape": [1, -1, 1], "dtype": "int32"},
            "velocity": {"type": "dynamic", "shape": [STEPS, -1, 2], "dtype": "float32"},
            "pressure": {"type": "dynamic", "shape": [STEPS, -1, 1], "dtype": "float32"},
        },
        "field_names": ["cells", "mesh_pos", "node_type", "velocity", "pressure"],
        "trajectory_length": STEPS,
    }
    (root / "meta.json").write_text(json.dumps(meta))
    sizes = [(9, 5), (10, 5), (8, 6), (9, 6)]
    for i, split in enumerate(("train", "valid", "test")):
        write_split(root / f"{split}.tfrecord", [make_trajectory(nx, ny, 0.7 * j + i) for j, (nx, ny) in enumerate(sizes)])
    return root


@pytest.fixture()
def cfg(data_dir, tmp_path):
    return load_config(
        "smoke",
        [
            f"data_dir={data_dir}",
            f"output_dir={tmp_path / 'runs'}",
            "num_training_samples=3",
            f"num_training_time_steps={STEPS}",
            "num_test_samples=2",
            f"num_test_time_steps={STEPS}",
            "validation.num_samples=2",
            "validation.stride=3",
            "rollout.horizons=[1,5,11]",
            "rollout.field_times=[5,11]",
            "rollout.max_lag=3",
            "log_every=10",
        ],
    )
