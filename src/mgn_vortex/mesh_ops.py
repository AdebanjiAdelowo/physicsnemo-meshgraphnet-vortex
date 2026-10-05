"""Discrete operators on the triangular meshes of the dataset.

The nodal fields are treated as continuous piecewise-linear (P1) functions on the
supplied triangulation. Nothing here uses information beyond node positions and cells.
"""

import numpy as np
import scipy.sparse as sp


def triangle_areas(pos: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Signed areas of the triangles, shape (n_cells,). ``pos`` is (n_nodes, 2), ``cells`` (n_cells, 3)."""
    p0, p1, p2 = (pos[cells[:, k]].astype(np.float64) for k in range(3))
    return 0.5 * ((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))


def nodal_areas(pos: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Lumped-mass weights: one third of the area of every triangle adjacent to a node.

    The weights sum to the mesh area and define the discrete L2 norm used for the
    error metrics, so that the fine mesh near the cylinder is not over-counted.
    """
    area = np.abs(triangle_areas(pos, cells))
    weights = np.zeros(len(pos))
    for k in range(3):
        np.add.at(weights, cells[:, k], area / 3.0)
    return weights


def gradient_operators(pos: np.ndarray, cells: np.ndarray) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Sparse nodal d/dx and d/dy operators, shape (n_nodes, n_nodes).

    The gradient of a P1 function is constant on each triangle; the nodal value is
    the area-weighted average over the adjacent triangles. The operators are exact
    for linear fields at every node and first-order accurate otherwise.
    """
    pos = pos.astype(np.float64)
    n_nodes, n_cells = len(pos), len(cells)
    signed = triangle_areas(pos, cells)
    rows = np.repeat(np.arange(n_cells), 3)
    cols = cells.reshape(-1)
    gx = np.empty((n_cells, 3))
    gy = np.empty((n_cells, 3))
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        gx[:, i] = (pos[cells[:, j], 1] - pos[cells[:, k], 1]) / (2.0 * signed)
        gy[:, i] = (pos[cells[:, k], 0] - pos[cells[:, j], 0]) / (2.0 * signed)
    cell_dx = sp.csr_matrix((gx.reshape(-1), (rows, cols)), shape=(n_cells, n_nodes))
    cell_dy = sp.csr_matrix((gy.reshape(-1), (rows, cols)), shape=(n_cells, n_nodes))
    area = np.abs(signed)
    average = sp.csr_matrix((np.repeat(area, 3), (cols, rows)), shape=(n_nodes, n_cells))
    average = sp.diags(1.0 / np.asarray(average.sum(axis=1)).ravel()) @ average
    return (average @ cell_dx).tocsr(), (average @ cell_dy).tocsr()


def vorticity(velocity: np.ndarray, dx: sp.csr_matrix, dy: sp.csr_matrix) -> np.ndarray:
    """Nodal vorticity dv/dx - du/dy of velocity fields of shape (..., n_nodes, 2)."""
    flat = velocity.reshape(-1, velocity.shape[-2], 2).astype(np.float64)
    omega = (dx @ flat[:, :, 1].T - dy @ flat[:, :, 0].T).T
    return omega.reshape(velocity.shape[:-1])
