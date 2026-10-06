"""Figures of a study, drawn only from the files written by training and evaluation."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402

from . import mesh_ops  # noqa: E402

# Categorical slots for the processor sizes, in fixed order, with a marker and a line style
# as second encoding. Text and axes stay in neutral ink.
SERIES = [("#2a78d6", "o", "-"), ("#eb6834", "s", "--"), ("#1baf7a", "^", "-.")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQUENTIAL, DIVERGING = "Blues", "RdBu_r"

plt.rcParams.update(
    {
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "font.size": 9,
    }
)


class Study:
    """Reader of a study directory (a run directory or its copy under ``results/``)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.summary = json.loads((self.path / "summary.json").read_text())
        self.meta = json.loads((self.path / "study_metadata.json").read_text())
        self.cfg = self.meta["config"]
        self.rows = self.summary["runs"]
        self.sizes = sorted({r["processor_size"] for r in self.rows})
        self.seeds = sorted({r["seed"] for r in self.rows})
        self.window = self.cfg["num_training_time_steps"] - 1
        self.style = {k: SERIES[i] for i, k in enumerate(self.sizes)}

    def run_dir(self, size: int, seed: int) -> Path:
        return self.path / f"mp{size:02d}_seed{seed}"

    def curves(self, size: int, name: str) -> np.ndarray:
        """Shape (seeds, trajectories, steps)."""
        return np.stack([np.load(self.run_dir(size, s) / "rollout_curves.npz")[name] for s in self.seeds]).astype(np.float64)

    def history(self, size: int, seed: int) -> list[dict]:
        with open(self.run_dir(size, seed) / "history.csv") as f:
            return list(csv.DictReader(f))

    def values(self, size: int, key: str) -> np.ndarray:
        return np.array([r[key] for r in self.rows if r["processor_size"] == size], dtype=float)

    def label(self, size: int) -> str:
        return f"{size} message-passing steps"

    def subtitle(self) -> str:
        device = self.meta["gpu_name"] or self.meta["device"]
        return (
            f"{self.summary['name']} study: {self.summary['steps_per_run']:,} gradient steps per model, "
            f"{len(self.seeds)} seed(s), {self.summary['test_trajectories']} test trajectories, {device}"
        )


def _finish(fig, study: Study, path: Path, title: str) -> Path:
    footer = 0.18 / fig.get_figheight()
    fig.get_layout_engine().set(rect=(0, footer, 1, 1 - footer))
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, 0.005, study.subtitle(), ha="left", va="bottom", fontsize=7.5, color=MUTED)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _seed_band(ax, x, per_seed, color, marker, line, label, markevery=None):
    """Mean over seeds as a line; the range over seeds as a band."""
    mean = per_seed.mean(axis=0)
    if per_seed.shape[0] > 1:
        ax.fill_between(x, per_seed.min(axis=0), per_seed.max(axis=0), color=color, alpha=0.18, linewidth=0)
    ax.plot(x, mean, color=color, linestyle=line, marker=marker, markevery=markevery, label=label)
    return mean


def _window_line(ax, study: Study, steps: int, label: bool = True) -> None:
    if steps > study.window:
        ax.axvline(study.window, color=MUTED, linewidth=1.0, linestyle=":")
        if label:
            ax.annotate("end of training\ntime window", (study.window, 0.0), xycoords=("data", "axes fraction"),
                        xytext=(4, 4), textcoords="offset points", va="bottom", fontsize=7.5, color=MUTED)


def rollout_error(study: Study, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    panels = (("rel_l2_velocity", "Velocity"), ("rel_l2_pressure", "Pressure"))
    for ax, (name, title) in zip(axes, panels):
        for size in study.sizes:
            color, marker, line = study.style[size]
            per_seed = study.curves(size, name).mean(axis=1)
            x = np.arange(1, per_seed.shape[1] + 1)
            _seed_band(ax, x, per_seed, color, marker, line, study.label(size), markevery=max(len(x) // 8, 1))
        if name == "rel_l2_velocity":
            persistence = study.curves(study.sizes[0], "persistence_rel_l2_velocity")[0].mean(axis=0)
            ax.plot(x, persistence, color=MUTED, linewidth=1.2, linestyle=(0, (1, 1)), label="initial state held fixed")
        _window_line(ax, study, len(x), label=name == "rel_l2_velocity")
        ax.set_yscale("log")
        ax.set_xlabel("rollout step (0.01 s each)")
        ax.set_ylabel("relative $L^2$ error")
        ax.set_title(title)
    axes[0].legend(loc="lower right")
    return _finish(fig, study, out, "Rollout error against time, mean over test trajectories")


def error_vs_depth(study: Study, out: Path) -> Path:
    horizons = list(study.cfg["rollout"]["horizons"])
    if len(horizons) > 5:  # a readable subset: first step, short and medium range, end of the window, end of the rollout
        keep = {horizons[0], 10, 100, study.window, horizons[-1]}
        horizons = [h for h in horizons if h in keep]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), constrained_layout=True)
    x = np.array(study.sizes)

    def scatter_mean(ax, key, color, label=None, marker="o"):
        values = np.stack([study.values(k, key) for k in study.sizes])
        for j in range(values.shape[1]):
            ax.plot(x, values[:, j], linestyle="none", marker=marker, markersize=4, color=color, alpha=0.45)
        ax.plot(x, values.mean(axis=1), color=color, marker=marker, label=label)
        return values.mean(axis=1)

    scatter_mean(axes[0], "one_step_rel_l2_velocity_increment", SERIES[0][0])
    axes[0].set_title("One step: velocity increment")
    axes[0].set_ylabel("error / norm of the true increment")

    shades = plt.get_cmap(SEQUENTIAL)(np.linspace(0.35, 0.95, len(horizons)))
    for metric, ax, title in (("rel_l2_velocity", axes[1], "Rollout: velocity"), ("rel_l2_pressure", axes[2], "Rollout: pressure")):
        for shade, h in zip(shades, horizons):
            mean = scatter_mean(ax, f"rollout_{metric}_h{h}", shade)
            ax.annotate(f"step {h}", (x[-1], mean[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7.5, color=MUTED)
        ax.set_title(title)
        ax.set_ylabel("relative $L^2$ error")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xlabel("message-passing steps")
        span = x[-1] - x[0]
        ax.set_xlim(x[0] - 0.15 * span, x[-1] + 0.35 * span)
    return _finish(fig, study, out, "Error against processor depth (small markers: seeds, line: mean)")


def training_curves(study: Study, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True, sharex=True)
    for size in study.sizes:
        color, marker, line = study.style[size]
        train, valid, steps_t, steps_v = [], [], None, None
        for seed in study.seeds:
            history = study.history(size, seed)
            steps_t = np.array([int(r["step"]) for r in history])
            train.append([float(r["train_loss"]) for r in history])
            rows_v = [r for r in history if r["valid_loss"]]
            steps_v = np.array([int(r["step"]) for r in rows_v])
            valid.append([float(r["valid_loss"]) for r in rows_v])
        _seed_band(axes[0], steps_t, np.array(train), color, None, line, study.label(size))
        _seed_band(axes[1], steps_v, np.array(valid), color, marker, line, study.label(size))
    axes[0].set_title("Training loss (noisy inputs, mean over each logging interval)")
    axes[1].set_title("Validation loss (noise-free inputs, held-out trajectories)")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("gradient step")
        ax.set_ylabel("MSE of normalised targets")
    axes[1].legend(loc="upper right")
    return _finish(fig, study, out, "Training curves under the same budget")


def cost(study: Study, out: Path) -> Path:
    """Parameters against error, and the cost per step from the timing probe stored with the study.

    Run durations are not plotted: they include validation and depend on what else the machine
    was doing during a long run. ``probe.json`` holds a short dedicated measurement per size.
    """
    probe_path = study.path / "probe.json"
    probe = json.loads(probe_path.read_text()) if probe_path.exists() else None
    fig, axes = plt.subplots(1, 3 if probe else 1, figsize=(13 if probe else 5, 4.3), constrained_layout=True, squeeze=False)
    axes = axes[0]
    params = np.array([study.values(k, "parameters")[0] for k in study.sizes]) / 1e6
    x = np.array(study.sizes)
    errors = np.stack([study.values(k, "rollout_mean_rel_l2_velocity_window") for k in study.sizes])
    axes[0].plot(params, errors.mean(axis=1), color=MUTED, linewidth=1.2, zorder=1)
    for i, size in enumerate(study.sizes):
        color, marker, _ = study.style[size]
        axes[0].plot(np.full(errors.shape[1], params[i]), errors[i], linestyle="none", marker=marker, color=color, markersize=7,
                     markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=study.label(size), zorder=2)
    axes[0].set_yscale("log")
    axes[0].set_title("Parameters and rollout error (one marker per seed)")
    axes[0].set_xlabel("trainable parameters (millions)")
    axes[0].set_ylabel("mean velocity error over the training window")
    axes[0].legend(loc="best")
    if probe:
        rows = {r["processor_size"]: r for r in probe["per_processor_size"]}
        steps = study.cfg["num_test_time_steps"] - 1
        train_ms = [1000.0 / rows[k]["steps_per_second"] for k in study.sizes]
        rollout_ms = [1000.0 * rows[k]["rollout_seconds_per_trajectory"] / steps for k in study.sizes]
        for ax, values, title in ((axes[1], train_ms, "Training: time per gradient step"), (axes[2], rollout_ms, "Rollout: time per step")):
            ax.plot(x, values, color=MUTED, linewidth=1.2, zorder=1)
            for i, size in enumerate(study.sizes):
                color, marker, _ = study.style[size]
                ax.plot(x[i], values[i], linestyle="none", marker=marker, color=color, markersize=8, markeredgecolor="#fcfcfb", markeredgewidth=1.5, zorder=2)
                ax.annotate(f"{values[i]:.0f} ms", (x[i], values[i]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8, color=MUTED)
            ax.set_title(title)
            ax.set_xlabel("message-passing steps")
            ax.set_ylabel("milliseconds")
            ax.set_xticks(x)
            ax.set_ylim(0, 1.2 * max(values))
    return _finish(fig, study, out, "Cost of depth")


def error_accumulation(study: Study, out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)
    for size in study.sizes:
        color, marker, line = study.style[size]
        plain = study.curves(size, "rel_l2_velocity").mean(axis=(0, 1))
        aligned = study.curves(size, "lagmin_rel_l2_velocity").mean(axis=(0, 1))
        x = np.arange(1, len(plain) + 1)
        axes[0].plot(x, aligned / plain, color=color, linestyle=line, label=study.label(size))
        ke_pred, ke_true = study.curves(size, "kinetic_energy_pred"), study.curves(size, "kinetic_energy_true")
        axes[1].plot(x, ((ke_pred - ke_true) / ke_true).mean(axis=(0, 1)), color=color, linestyle=line, label=study.label(size))
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Share of the error left after the best time shift")
    axes[0].set_ylabel(f"lag-minimised error / error (shift up to {study.cfg['rollout']['max_lag']} steps)")
    axes[1].axhline(0, color=MUTED, linewidth=0.8)
    axes[1].set_title("Kinetic energy drift")
    axes[1].set_ylabel("(predicted - reference) / reference")
    axes[1].legend(loc="best")

    largest = study.sizes[-1]
    per_traj = study.curves(largest, "rel_l2_velocity")[0]
    fluct = np.array(study.summary["reference_fluctuation_ratio"])
    shades = plt.get_cmap(SEQUENTIAL)(0.3 + 0.65 * (fluct - fluct.min()) / max(np.ptp(fluct), 1e-12))
    for curve, shade in zip(per_traj, shades):
        axes[2].plot(x, curve, color=shade, linewidth=1.0)
    axes[2].plot(x, np.median(per_traj, axis=0), color=INK, linewidth=2.0, label="median")
    axes[2].set_yscale("log")
    axes[2].set_title(f"Every test trajectory, {largest} steps, seed {study.seeds[0]}")
    axes[2].set_ylabel("relative $L^2$ velocity error (darker: more unsteady reference)")
    axes[2].legend(loc="lower right")
    for i, ax in enumerate(axes):
        ax.set_xlabel("rollout step (0.01 s each)")
        _window_line(ax, study, len(x), label=i == 0)
    return _finish(fig, study, out, "How the rollout error accumulates")


def _tri(fields):
    pos = fields["mesh_pos"]
    return mtri.Triangulation(pos[:, 0], pos[:, 1], fields["cells"])


def _panel(ax, tri, values, cmap, vmin, vmax):
    ax.set_facecolor("#8a8983")  # the cylinder, which is not meshed
    image = ax.tripcolor(tri, values, cmap=cmap, vmin=vmin, vmax=vmax, shading="gouraud", rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlim(tri.x.min(), tri.x.max())
    ax.set_ylim(tri.y.min(), tri.y.max())
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def fields_vorticity(study: Study, out: Path) -> Path:
    """Reference, prediction of the deepest model and absolute error of the vorticity at the stored times."""
    fields = np.load(study.path / "fields.npz")
    size = study.sizes[-1]
    tri = _tri(fields)
    dx, dy = mesh_ops.gradient_operators(fields["mesh_pos"], fields["cells"])
    truth = mesh_ops.vorticity(fields["truth"][..., 0:2], dx, dy)
    pred = mesh_ops.vorticity(fields[f"prediction_mp{size:02d}"][..., 0:2], dx, dy)
    times = fields["times"]
    limit = np.percentile(np.abs(truth), 98)
    error_max = np.percentile(np.abs(pred - truth), 98)
    fig, axes = plt.subplots(len(times), 3, figsize=(13, 1.05 * len(times) + 1.0), constrained_layout=True, squeeze=False)
    for i, t in enumerate(times):
        left = _panel(axes[i, 0], tri, truth[i], DIVERGING, -limit, limit)
        _panel(axes[i, 1], tri, pred[i], DIVERGING, -limit, limit)
        right = _panel(axes[i, 2], tri, np.abs(pred[i] - truth[i]), SEQUENTIAL, 0, error_max)
        axes[i, 0].set_ylabel(f"step {t}", fontsize=9)
    axes[0, 0].set_title("Reference (COMSOL)")
    axes[0, 1].set_title(f"MeshGraphNet rollout, {size} message-passing steps")
    axes[0, 2].set_title("Absolute error")
    fig.colorbar(left, ax=axes[:, :2], shrink=0.7, pad=0.01, label="vorticity (1/s), same scale for both columns", extend="both")
    fig.colorbar(right, ax=axes[:, 2], shrink=0.7, pad=0.02, label="|error| (1/s), clipped at the 98th percentile", extend="max")
    title = f"Vorticity of test trajectory {int(fields['trajectory'])}, seed {int(fields['plot_seed'])}"
    return _finish(fig, study, out, title)


def fields_by_depth(study: Study, out: Path) -> Path:
    """Horizontal velocity at the end of the training window for every processor size."""
    fields = np.load(study.path / "fields.npz")
    tri = _tri(fields)
    times = list(fields["times"])
    i = int(np.argmin(np.abs(np.array(times) - study.window)))
    truth = fields["truth"][i, :, 0]
    vmin, vmax = float(truth.min()), float(truth.max())
    preds = {k: fields[f"prediction_mp{k:02d}"][i, :, 0] for k in study.sizes}
    error_max = max(np.percentile(np.abs(p - truth), 99.5) for p in preds.values())
    fig, axes = plt.subplots(len(study.sizes) + 1, 2, figsize=(11, 1.3 * (len(study.sizes) + 1) + 1.0), constrained_layout=True)
    left = _panel(axes[0, 0], tri, truth, "cividis", vmin, vmax)
    axes[0, 0].set_ylabel("reference")
    axes[0, 1].axis("off")
    for row, size in enumerate(study.sizes, start=1):
        _panel(axes[row, 0], tri, preds[size], "cividis", vmin, vmax)
        right = _panel(axes[row, 1], tri, np.abs(preds[size] - truth), SEQUENTIAL, 0, error_max)
        axes[row, 0].set_ylabel(f"{size} steps")
    axes[0, 0].set_title("Horizontal velocity u")
    axes[1, 1].set_title("Absolute error of u")
    fig.colorbar(left, ax=axes[:, 0], shrink=0.7, pad=0.01, label="u (m/s), same scale for all rows")
    fig.colorbar(right, ax=axes[1:, 1], shrink=0.7, pad=0.02, label="|error| (m/s), same scale for all rows")
    title = f"Rollout step {times[i]} of test trajectory {int(fields['trajectory'])}, seed {int(fields['plot_seed'])}"
    return _finish(fig, study, out, title)


FIGURES = {
    "rollout_error": rollout_error,
    "error_vs_depth": error_vs_depth,
    "error_accumulation": error_accumulation,
    "training_curves": training_curves,
    "cost": cost,
    "fields_vorticity": fields_vorticity,
    "fields_by_depth": fields_by_depth,
}


def plot_study(study_dir: Path, out_dir: Path) -> list[Path]:
    study = Study(study_dir)
    return [draw(study, Path(out_dir) / f"{study.summary['name']}_{name}.png") for name, draw in FIGURES.items()]
