import pytest
import torch
from torch_geometric.data import Data

from mgn_vortex.config import load_config
from mgn_vortex.model import build_model, count_parameters

# Trainable parameters of the upstream architecture (hidden size 128) per processor size.
PARAMETERS = {5: 845_059, 10: 1_588_739, 15: 2_332_419}


def _graph(n=30, e=120):
    g = torch.Generator().manual_seed(0)
    edge_index = torch.randint(0, n, (2, e), generator=g)
    return torch.randn(n, 6, generator=g), torch.randn(e, 3, generator=g), Data(edge_index=edge_index, num_nodes=n)


@pytest.mark.parametrize("size", [5, 10, 15])
def test_parameter_count(size):
    assert count_parameters(build_model(load_config("official"), size)) == PARAMETERS[size]


def test_default_processor_size_is_the_upstream_model():
    cfg = load_config("official")
    from physicsnemo.models.meshgraphnet import MeshGraphNet

    upstream = MeshGraphNet(cfg.num_input_features, cfg.num_edge_features, cfg.num_output_features)
    ours = build_model(cfg, cfg.processor_size)
    assert [(k, v.shape) for k, v in upstream.state_dict().items()] == [(k, v.shape) for k, v in ours.state_dict().items()]


def test_only_the_processor_depth_changes():
    cfg = load_config("official")
    small, large = build_model(cfg, 5), build_model(cfg, 15)
    # one edge block and one node block per message-passing step
    assert len(small.processor.processor_layers) == 10 and len(large.processor.processor_layers) == 30
    for part in ("node_encoder", "edge_encoder", "node_decoder"):
        assert count_parameters(getattr(small, part)) == count_parameters(getattr(large, part))
    per_step = (PARAMETERS[15] - PARAMETERS[5]) / 10
    assert count_parameters(large.processor) == 15 * per_step


def test_forward_shape_and_gradients():
    x, e, graph = _graph()
    model = build_model(load_config("official"), 3)
    out = model(x, e, graph)
    assert out.shape == (30, 3) and torch.isfinite(out).all()
    out.square().mean().backward()
    assert all(p.grad is not None for p in model.parameters())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_mps_matches_cpu():
    x, e, graph = _graph()
    torch.manual_seed(0)
    model = build_model(load_config("official"), 3).eval()
    with torch.no_grad():
        cpu = model(x, e, graph)
        mps = model.to("mps")(x.to("mps"), e.to("mps"), graph.to("mps")).cpu()
    assert torch.allclose(cpu, mps, atol=1e-4)
