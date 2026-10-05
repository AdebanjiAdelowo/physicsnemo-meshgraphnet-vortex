import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import STEPS
from torch_geometric.loader import DataLoader as PyGDataLoader

from mgn_vortex.data import build_dataset
from mgn_vortex.model import build_model
from mgn_vortex.rollout import rollout

CPU = torch.device("cpu")


@pytest.fixture()
def test_set(cfg, tmp_path):
    torch.manual_seed(0)
    build_dataset(cfg, "train", tmp_path)  # writes the normalisation statistics
    return build_dataset(cfg, "test", tmp_path)


def oracle(node_features, edge_features, graph):
    """A 'model' that returns the normalised reference targets."""
    return graph.y


def zero(node_features, edge_features, graph):
    return torch.zeros_like(graph.y)


def test_shapes_and_time_levels(test_set):
    out = rollout(zero, test_set, 1, CPU)
    n = out["mesh_pos"].shape[0]
    assert n == 50  # second trajectory, its own mesh
    assert out["pred"].shape == out["exact"].shape == (STEPS - 1, n, 3)
    assert out["input_velocity"].shape == (STEPS - 1, n, 2)
    assert list(out["steps"]) == list(range(1, STEPS))
    assert out["cells"].shape[1] == 3 and out["mask"].shape == (n,)


def test_oracle_rollout_reproduces_the_reference(test_set):
    out = rollout(oracle, test_set, 0, CPU)
    assert np.allclose(out["pred"], out["exact"], atol=2e-5)
    # the reference of step k+1 is the input of step k+2
    assert np.allclose(out["exact"][:-1, :, :2], out["input_velocity"][1:], atol=1e-5)


def test_boundary_velocity_is_never_updated(test_set):
    out = rollout(zero, test_set, 0, CPU)
    fixed = ~out["mask"]
    assert fixed.any() and (~fixed).any()
    initial = out["input_velocity"][0]
    assert np.allclose(out["pred"][:, fixed, :2], initial[fixed], atol=1e-6)
    # a zero normalised increment is the mean increment in physical units, accumulated on free nodes
    step = test_set.node_stats["velocity_diff_mean"].numpy()
    assert np.allclose(out["pred"][-1][~fixed, :2], initial[~fixed] + (STEPS - 1) * step, atol=1e-4)


def test_one_step_uses_reference_inputs(test_set):
    torch.manual_seed(1)
    model = build_model_small()
    free = rollout(model, test_set, 0, CPU, autoregressive=True)
    forced = rollout(model, test_set, 0, CPU, autoregressive=False)
    assert np.allclose(free["pred"][0], forced["pred"][0], atol=1e-6)  # same first step
    assert not np.allclose(free["pred"][-1], forced["pred"][-1], atol=1e-6)
    strided = rollout(model, test_set, 0, CPU, autoregressive=False, stride=4)
    assert list(strided["steps"]) == [1, 5, 9]
    assert np.allclose(strided["pred"], forced["pred"][[0, 4, 8]], atol=1e-6)
    with pytest.raises(ValueError):
        rollout(model, test_set, 0, CPU, autoregressive=True, stride=2)


def build_model_small():
    from mgn_vortex.config import load_config

    return build_model(load_config("official"), 2).eval()


def test_matches_the_upstream_rollout_if_a_clone_is_available(test_set, cfg):
    """Run NVIDIA's MGNRollout.predict on the same model and data and compare the fields."""
    clone = os.environ.get("PHYSICSNEMO_UPSTREAM")
    if not clone:
        pytest.skip("set PHYSICSNEMO_UPSTREAM to a PhysicsNeMo checkout to compare with inference.py")
    spec = importlib.util.spec_from_file_location("upstream_inference", Path(clone) / "examples/cfd/vortex_shedding_mgn/inference.py")
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)

    torch.manual_seed(1)
    model = build_model_small()
    reference = object.__new__(upstream.MGNRollout)
    reference.num_test_time_steps = STEPS
    reference.device = "cpu"
    reference.dataset = test_set
    reference.dataloader = PyGDataLoader(test_set, batch_size=1, shuffle=False, drop_last=False)
    reference.model = model
    reference.predict()

    per_trajectory = STEPS - 1
    for traj in range(cfg.num_test_samples):
        ours = rollout(model, test_set, traj, CPU)
        chunk = slice(traj * per_trajectory, (traj + 1) * per_trajectory)
        theirs_pred = torch.stack(reference.pred[chunk]).numpy()
        theirs_exact = torch.stack(reference.exact[chunk]).numpy()
        assert np.allclose(ours["pred"], theirs_pred, atol=1e-5)
        assert np.allclose(ours["exact"], theirs_exact, atol=1e-5)
