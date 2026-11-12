import json
import sys

import pytest
import torch
from omegaconf import OmegaConf, open_dict

from mgn_vortex.budget import matched_data, train_budget
from mgn_vortex.config import steps_per_run


@pytest.mark.parametrize("workers", [
    0,
    pytest.param(2, marks=pytest.mark.skipif(sys.platform != "linux", reason="Kaggle Linux worker start method")),
])
def test_resume_matches_uninterrupted_training(cfg, tmp_path, workers):
    with open_dict(cfg):
        cfg = OmegaConf.merge(cfg, {
            "epochs": 2,
            "num_dataloader_workers": workers,
            "processor_size": 2,
            "budget": {"cutoffs": [11, 33, 66], "checkpoint_every": 9},
        })
    assert steps_per_run(cfg) == 66
    data = matched_data(cfg)
    device = torch.device("cpu")
    whole = tmp_path / "whole"
    resumed = tmp_path / "resumed"
    assert train_budget(cfg, whole, 0, device, dataset=data) == 66
    assert train_budget(cfg, resumed, 0, device, dataset=data, stop_at=14) == 14
    assert train_budget(cfg, resumed, 0, device, dataset=data, stop_at=33) == 33
    assert train_budget(cfg, resumed, 0, device, dataset=data) == 66
    for cutoff in cfg.budget.cutoffs:
        path = f"steps_{cutoff:06d}/mp02_seed0/budget_weights.pt"
        a = torch.load(whole / path, weights_only=True)
        b = torch.load(resumed / path, weights_only=True)
        assert a.keys() == b.keys()
        assert all(torch.equal(a[key], b[key]) for key in a)
        row = json.loads((resumed / path).with_name("train_metrics.json").read_text())
        assert row["steps"] == cutoff
    checkpoint = resumed / "training_state/mp02_seed0/step_000066.pt"
    original = checkpoint.read_bytes()
    assert train_budget(cfg, resumed, 0, device, dataset=data) == 66
    assert checkpoint.read_bytes() == original
    with pytest.raises(ValueError, match="differs"):
        train_budget(cfg, resumed, 0, device, dataset={**data, "meta_sha256": "changed"})


def test_dataset_manifest_hash_mismatch(cfg, tmp_path):
    root = tmp_path / "copy"
    root.mkdir()
    from pathlib import Path
    import shutil

    for entry in Path(cfg.data_dir).iterdir():
        shutil.copy(entry, root)
    (root / "manifest.json").write_text(json.dumps({"splits": {
        "train": {"bytes": (root / "train.tfrecord").stat().st_size, "sha256": "incorrect"},
    }}))
    cfg.data_dir = str(root)
    with pytest.raises(ValueError, match="Dataset hash mismatch"):
        matched_data(cfg)
