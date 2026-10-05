import csv
import json

import numpy as np
import torch
from conftest import STEPS

from mgn_vortex import engine
from mgn_vortex.config import steps_per_run
from mgn_vortex.data import build_dataset
from mgn_vortex.study import evaluate_study, summary_fields, train_study

CPU = torch.device("cpu")


def test_run_order_is_seed_major(cfg):
    from omegaconf import OmegaConf

    two = OmegaConf.merge(cfg, {"study": {"processor_sizes": [2, 3], "seeds": [0, 1]}})
    assert engine.study_runs(two) == [(2, 0), (3, 0), (2, 1), (3, 1)]
    assert engine.run_name(5, 2) == "mp05_seed2"


def test_training_is_reproducible_and_seed_dependent(cfg, tmp_path):
    a = engine.train_run(cfg, 2, 0, tmp_path / "a", CPU)
    b = engine.train_run(cfg, 2, 0, tmp_path / "b", CPU)
    c = engine.train_run(cfg, 2, 1, tmp_path / "c", CPU)
    assert a["final_train_loss"] == b["final_train_loss"] and a["final_valid_loss"] == b["final_valid_loss"]
    assert a["final_train_loss"] != c["final_train_loss"]
    assert a["steps"] == steps_per_run(cfg) == 3 * (STEPS - 1)
    assert (tmp_path / "a" / "node_stats.json").read_text() == (tmp_path / "b" / "node_stats.json").read_text()


def test_checkpoint_round_trip(cfg, tmp_path):
    run_dir = tmp_path / "run"
    engine.train_run(cfg, 2, 0, run_dir, CPU)
    assert list((run_dir / "checkpoints").glob("MeshGraphNet.0.0.mdlus"))
    first = engine.load_trained_model(cfg, 2, run_dir, CPU)
    second = engine.load_trained_model(cfg, 2, run_dir, CPU)
    for (name, p), q in zip(first.state_dict().items(), second.state_dict().values()):
        assert torch.equal(p, q), name
    # the loaded weights are the trained ones: they reproduce the stored validation loss
    valid = build_dataset(cfg, "valid", run_dir)
    loss = engine.validation_loss(first, valid, range(0, len(valid), cfg.validation.stride), CPU)
    stored = json.loads((run_dir / "train_metrics.json").read_text())["final_valid_loss"]
    assert abs(loss - stored) < 1e-6 * max(1.0, abs(stored))


def test_study_outputs(cfg, tmp_path):
    out_dir = train_study(cfg, tmp_path)
    summary = evaluate_study(cfg, tmp_path)

    meta = json.loads((out_dir / "study_metadata.json").read_text())
    for key in (
        "timestamp_utc",
        "git_commit",
        "git_dirty",
        "device",
        "gpu_name",
        "python_version",
        "torch_version",
        "physicsnemo",
        "torch_geometric_version",
        "torch_scatter_version",
        "tfrecord_version",
        "dataset",
        "steps_per_run",
        "config",
    ):
        assert key in meta, key
    assert meta["physicsnemo"]["example"] == "examples/cfd/vortex_shedding_mgn"
    assert meta["dataset"]["splits"]["train"]["trajectories_used"] == 3

    with open(out_dir / "summary.csv") as f:
        rows = list(csv.DictReader(f))
    assert [r["run"] for r in rows] == ["mp02_seed0", "mp03_seed0"]
    assert list(rows[0]) == summary_fields(cfg)
    assert int(rows[0]["parameters"]) < int(rows[1]["parameters"])
    assert summary["missing_runs"] == []
    assert [e["processor_size"] for e in summary["by_processor_size"]] == [2, 3]

    for run in ("mp02_seed0", "mp03_seed0"):
        run_dir = out_dir / run
        train = json.loads((run_dir / "train_metrics.json").read_text())
        assert train["steps"] == steps_per_run(cfg) and train["seed"] == 0
        with open(run_dir / "history.csv") as f:
            history = list(csv.DictReader(f))
        assert int(history[-1]["step"]) == train["steps"] and history[-1]["valid_loss"] != ""
        ev = json.loads((run_dir / "eval_metrics.json").read_text())
        assert set(ev["rollout_horizons"]) == {"1", "5", "11"}
        assert ev["rollout_steps"] == STEPS - 1
        curves = np.load(run_dir / "rollout_curves.npz")
        assert curves["rel_l2_velocity"].shape == (2, STEPS - 1)
        # the first rollout step is a one-step prediction from the reference state
        assert np.isfinite(curves["rel_l2_velocity"]).all()
        assert (curves["lagmin_rel_l2_velocity"] <= curves["rel_l2_velocity"] + 1e-6).all()
        assert np.allclose(curves["kinetic_energy_true"], np.load(out_dir / "mp02_seed0" / "rollout_curves.npz")["kinetic_energy_true"])

    fields = np.load(out_dir / "fields.npz")
    n = fields["mesh_pos"].shape[0]
    assert fields["truth"].shape == fields["prediction_mp02"].shape == fields["prediction_mp03"].shape == (2, n, 3)
    assert int(fields["trajectory"]) == summary["field_trajectory"]


def test_resume_skips_finished_runs(cfg, tmp_path, capsys):
    train_study(cfg, tmp_path)
    capsys.readouterr()
    train_study(cfg, tmp_path, resume=True)
    out = capsys.readouterr().out
    assert out.count("[skip]") == 2 and "[train]" not in out


def test_only_complete_seeds_are_evaluated(cfg, tmp_path):
    from omegaconf import OmegaConf

    out_dir = train_study(cfg, tmp_path)
    two_seeds = OmegaConf.merge(cfg, {"study": {"seeds": [0, 1]}})
    engine.train_run(two_seeds, 2, 1, out_dir / "mp02_seed1", CPU)  # seed 1 lacks processor size 3
    summary = evaluate_study(two_seeds, tmp_path)
    assert summary["seeds"] == [0]
    assert summary["missing_runs"] == ["mp02_seed1", "mp03_seed1"]
    assert all(e["n_seeds"] == 1 for e in summary["by_processor_size"])
