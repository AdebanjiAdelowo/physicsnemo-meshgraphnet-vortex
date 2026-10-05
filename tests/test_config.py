import os
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from mgn_vortex.config import CONFIG_DIR, load_config, steps_per_run

# Values of examples/cfd/vortex_shedding_mgn/conf/config.yaml at the pinned commit.
UPSTREAM = {
    "batch_size": 1,
    "epochs": 25,
    "num_training_samples": 400,
    "num_training_time_steps": 300,
    "lr": 0.0001,
    "lr_decay_rate": 0.9999991,
    "num_input_features": 6,
    "num_output_features": 3,
    "num_edge_features": 3,
    "amp": False,
    "jit": False,
    "num_dataloader_workers": 4,
    "do_concat_trick": False,
    "num_processor_checkpoint_segments": 0,
    "recompute_activation": False,
    "wandb_mode": "disabled",
    "watch_model": False,
    "ckpt_path": "./checkpoints",
    "num_test_samples": 10,
    "num_test_time_steps": 300,
    "viz_vars": ["u", "v", "p"],
    "frame_skip": 10,
    "frame_interval": 1,
}
# Keys a reduced study may change. Everything else must equal the official config.
BUDGET_KEYS = {
    "name",
    "device",
    "epochs",
    "num_training_samples",
    "num_training_time_steps",
    "num_dataloader_workers",
    "num_test_samples",
    "num_test_time_steps",
    "log_every",
    "validation",
    "rollout",
    "study",
    "output_dir",
}


@pytest.mark.parametrize("name", ["official", "smoke", "local", "full"])
def test_configs_compose(name):
    cfg = load_config(name)
    assert cfg.name == name
    assert cfg.output_dir == f"runs/{name}"
    assert steps_per_run(cfg) > 0
    assert max(cfg.rollout.horizons) <= cfg.num_test_time_steps - 1
    assert max(cfg.rollout.field_times) <= cfg.num_test_time_steps - 1


def test_official_keeps_the_upstream_values():
    cfg = OmegaConf.to_container(load_config("official"))
    for key, value in UPSTREAM.items():
        assert cfg[key] == value, key
    assert cfg["processor_size"] == 15 and cfg["study"]["processor_sizes"] == [15]
    assert steps_per_run(load_config("official")) == 2_990_000


def test_upstream_table_matches_a_clone_if_available():
    clone = os.environ.get("PHYSICSNEMO_UPSTREAM")
    if not clone:
        pytest.skip("set PHYSICSNEMO_UPSTREAM to a PhysicsNeMo checkout to compare with the upstream file")
    upstream = yaml.safe_load((Path(clone) / "examples/cfd/vortex_shedding_mgn/conf/config.yaml").read_text())
    upstream.pop("hydra")
    upstream.pop("data_dir")
    assert upstream == UPSTREAM


@pytest.mark.parametrize("name", ["smoke", "local", "full"])
def test_reduced_studies_change_only_budget_keys(name):
    official = OmegaConf.to_container(load_config("official"))
    reduced = OmegaConf.to_container(load_config(name))
    changed = {k for k in official if official[k] != reduced[k]}
    assert changed <= BUDGET_KEYS, changed - BUDGET_KEYS


def test_capacity_studies_vary_one_parameter_under_one_budget():
    for name, steps in (("local", 11_960), ("full", 119_600)):
        cfg = load_config(name)
        assert list(cfg.study.processor_sizes) == [5, 10, 15]
        assert list(cfg.study.seeds) == [0, 1, 2]
        assert steps_per_run(cfg) == steps  # independent of the processor size
        assert cfg.batch_size == 1  # the upstream dataset is only correct for batch size 1


def test_overrides():
    cfg = load_config("local", ["study.seeds=[0]", "epochs=3"])
    assert list(cfg.study.seeds) == [0] and cfg.epochs == 3


def test_config_files_present():
    assert {p.stem for p in CONFIG_DIR.glob("*.yaml")} == {"official", "smoke", "local", "full"}
