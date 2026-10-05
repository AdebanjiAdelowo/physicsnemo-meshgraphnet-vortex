"""Error metrics for one-step predictions and rollouts.

All arrays are NumPy. Fields have shape (time, nodes, components) and weights are the
lumped nodal areas of ``mesh_ops.nodal_areas``, shape (nodes,).
"""

import numpy as np


def weighted_norm(field: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Discrete L2 norm over nodes and components for every time, shape (time,)."""
    return np.sqrt(np.einsum("tnc,n->t", field.astype(np.float64) ** 2, weights))


def relative_l2(pred: np.ndarray, true: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """E(t) = ||pred(t) - true(t)|| / ||true(t)|| in the area-weighted norm."""
    return weighted_norm(pred - true, weights) / weighted_norm(true, weights)


def nodal_rmse(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Root mean square over nodes of the Euclidean error per node, shape (time,). Physical units."""
    return np.sqrt(np.mean(np.sum((pred - true).astype(np.float64) ** 2, axis=-1), axis=-1))


def kinetic_energy(velocity: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """0.5 * integral of |u|^2 over the mesh, per unit density and depth, shape (time,)."""
    return 0.5 * np.einsum("tnc,n->t", velocity.astype(np.float64) ** 2, weights)


def lag_minimised_error(pred: np.ndarray, true: np.ndarray, weights: np.ndarray, max_lag: int):
    """Relative error after the best time shift of the reference.

    E_lag(t) = min over |s| <= max_lag of ||pred(t) - true(t + s)|| / ||true(t)||.
    The shift s = 0 is always admissible, so E_lag(t) <= E(t). A prediction with the
    right flow pattern at the wrong phase has E_lag much smaller than E; a prediction
    with the wrong amplitude or structure does not. Returns (E_lag, best shift).
    """
    steps = pred.shape[0]
    norm = weighted_norm(true, weights)
    best = np.full(steps, np.inf)
    best_lag = np.zeros(steps, dtype=np.int64)
    t = np.arange(steps)
    for lag in range(-max_lag, max_lag + 1):
        valid = (t + lag >= 0) & (t + lag < steps)
        err = np.full(steps, np.inf)
        err[valid] = weighted_norm(pred[valid] - true[t[valid] + lag], weights)
        err = np.where(np.isfinite(err), err, np.inf)
        better = err < best
        best[better], best_lag[better] = err[better], lag
    return best / norm, best_lag


def fluctuation_ratio(velocity: np.ndarray, weights: np.ndarray) -> float:
    """Unsteadiness of a reference trajectory over its second half.

    Ratio of the area-weighted RMS of the velocity fluctuation about the time mean to
    the RMS of the velocity. Close to zero for a steady wake, larger with vortex shedding.
    """
    late = velocity[velocity.shape[0] // 2 :].astype(np.float64)
    fluct = late - late.mean(axis=0, keepdims=True)
    return float(np.sqrt(np.einsum("tnc,n->", fluct**2, weights) / np.einsum("tnc,n->", late**2, weights)))


def first_exceedance(error: np.ndarray, threshold: float) -> int:
    """First index at which the error exceeds the threshold or is not finite; -1 if never."""
    bad = ~np.isfinite(error) | (error > threshold)
    return int(np.argmax(bad)) if bad.any() else -1
