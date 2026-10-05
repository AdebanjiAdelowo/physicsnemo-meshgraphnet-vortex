import io
import json

import numpy as np
import pytest
import torch
from conftest import NODE_INFLOW, NODE_NORMAL, NODE_OUTFLOW, NODE_WALL, STEPS, make_mesh
from download_data import copy_records
from omegaconf import OmegaConf

from mgn_vortex.data import build_dataset, check_data, count_records


def test_count_records(data_dir):
    assert count_records(data_dir / "train.tfrecord") == 4


def test_partial_download_stops_at_a_record_boundary(data_dir, tmp_path):
    raw = (data_dir / "train.tfrecord").read_bytes()
    out = io.BytesIO()
    records, written, sha = copy_records(io.BytesIO(raw), out, 2)
    assert records == 2 and written == len(out.getvalue())
    assert raw.startswith(out.getvalue())
    part = tmp_path / "part.tfrecord"
    part.write_bytes(out.getvalue())
    assert count_records(part) == 2
    everything = io.BytesIO()
    assert copy_records(io.BytesIO(raw), everything, None)[0] == 4 and everything.getvalue() == raw
    with pytest.raises(IOError):
        copy_records(io.BytesIO(raw[:-5]), io.BytesIO(), None)


def test_check_data_reports_the_subset(cfg):
    record = check_data(cfg)
    assert record["splits"]["train"]["trajectories_used"] == 3
    assert record["splits"]["train"]["trajectories_in_file"] == 4
    assert record["splits"]["test"]["time_steps_used"] == STEPS
    json.dumps(record)


def test_check_data_rejects_too_few_trajectories(cfg):
    too_many = OmegaConf.merge(cfg, {"num_training_samples": 5})
    with pytest.raises(ValueError, match="holds 4 trajectories"):
        check_data(too_many)


def test_training_graph(cfg, tmp_path):
    torch.manual_seed(0)
    dataset = build_dataset(cfg, "train", tmp_path)
    assert len(dataset) == 3 * (STEPS - 1)
    assert (tmp_path / "node_stats.json").exists() and (tmp_path / "edge_stats.json").exists()

    graph = dataset[0]
    pos, cells, node_type = make_mesh(9, 5)
    n = len(pos)
    assert graph.x.shape == (n, 6) and graph.y.shape == (n, 3)
    assert graph.edge_attr.shape == (graph.edge_index.shape[1], 3)
    assert graph.x.dtype == graph.y.dtype == graph.edge_attr.dtype == torch.float32

    # every mesh edge appears in both directions, once
    pairs = set(map(tuple, graph.edge_index.T.tolist()))
    assert len(pairs) == graph.edge_index.shape[1]
    assert all((j, i) in pairs for i, j in pairs)
    mesh_edges = {tuple(sorted((int(c[a]), int(c[b])))) for c in cells for a, b in ((0, 1), (1, 2), (2, 0))}
    assert len(pairs) == 2 * len(mesh_edges)

    # edge features: relative position and its length, normalised with the stored statistics
    row, col = graph.edge_index
    disp = torch.tensor(pos)[row] - torch.tensor(pos)[col]
    raw = torch.cat([disp, disp.norm(dim=1, keepdim=True)], dim=1)
    expected = (raw - dataset.edge_stats["edge_mean"]) / dataset.edge_stats["edge_std"]
    assert torch.allclose(graph.edge_attr, expected, atol=1e-5)

    # one-hot node type in the order interior, inflow, outflow, wall
    one_hot = graph.x[:, 2:]
    codes = torch.tensor([NODE_NORMAL, NODE_INFLOW, NODE_OUTFLOW, NODE_WALL])
    assert torch.equal(codes[one_hot.argmax(dim=1)], torch.tensor(node_type[:, 0]))
    assert torch.all(one_hot.sum(dim=1) == 1)


def test_different_meshes_per_trajectory(cfg, tmp_path):
    torch.manual_seed(0)
    dataset = build_dataset(cfg, "train", tmp_path)
    sizes = [dataset[i * (STEPS - 1)].x.shape[0] for i in range(3)]
    assert sizes == [45, 50, 48]


def test_evaluation_split_is_noise_free_and_carries_the_mesh(cfg, tmp_path):
    torch.manual_seed(0)
    train = build_dataset(cfg, "train", tmp_path)
    test = build_dataset(cfg, "test", tmp_path)
    graph, cells, mask = test[0]
    assert graph["mesh_pos"].shape == (45, 2) and cells.shape[1] == 3
    # velocity is advanced on interior and outflow nodes only
    _, _, node_type = make_mesh(9, 5)
    expected = np.isin(node_type[:, 0], [NODE_NORMAL, NODE_OUTFLOW])
    assert np.array_equal(mask.numpy()[:, 0], expected)
    # targets: velocity increment and next pressure, in the units of the training statistics
    stats = train.node_stats
    increment = graph.y[:, :2] * stats["velocity_diff_std"] + stats["velocity_diff_mean"]
    velocity = graph.x[:, :2] * stats["velocity_std"] + stats["velocity_mean"]
    nxt, _, _ = test[1]
    velocity_next = nxt.x[:, :2] * stats["velocity_std"] + stats["velocity_mean"]
    assert torch.allclose(velocity + increment, velocity_next, atol=1e-5)


def test_items_share_one_graph_object(cfg, tmp_path):
    """Why every config keeps the upstream batch size of 1: items of a trajectory alias one object."""
    torch.manual_seed(0)
    dataset = build_dataset(cfg, "train", tmp_path)
    first = dataset[0]
    x0 = first.x.clone()
    second = dataset[1]
    assert first is second and not torch.equal(first.x, x0)
