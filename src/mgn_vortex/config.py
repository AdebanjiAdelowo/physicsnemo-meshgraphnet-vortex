"""Configuration loading through Hydra's compose API."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def load_config(name: str, overrides: list[str] | None = None) -> DictConfig:
    """Compose ``configs/<name>.yaml`` with optional Hydra overrides.

    Parameters
    ----------
    name : str
        Config name without the ``.yaml`` suffix (``official``, ``smoke``, ``local``, ``full``).
    overrides : list[str], optional
        Hydra overrides such as ``epochs=1`` or ``study.seeds=[0]``.
    """
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(config_name=name, overrides=list(overrides or []))


def steps_per_run(cfg: DictConfig) -> int:
    """Gradient steps of one training run: one per (trajectory, time step) pair and epoch."""
    samples = cfg.num_training_samples * (cfg.num_training_time_steps - 1)
    return cfg.epochs * (samples // cfg.batch_size)
