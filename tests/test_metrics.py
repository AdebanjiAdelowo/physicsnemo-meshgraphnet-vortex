import numpy as np
from conftest import make_mesh

from mgn_vortex import mesh_ops, metrics


def _mesh():
    pos, cells, _ = make_mesh(13, 7)
    return pos.astype(np.float64), cells


def test_nodal_areas_sum_to_the_domain_area():
    pos, cells = _mesh()
    weights = mesh_ops.nodal_areas(pos, cells)
    assert np.isclose(weights.sum(), 1.6 * 0.4)
    assert (weights > 0).all()
    # independent of the orientation of the cells
    assert np.allclose(weights, mesh_ops.nodal_areas(pos, cells[:, ::-1]))


def test_gradients_are_exact_for_linear_fields():
    pos, cells = _mesh()
    dx, dy = mesh_ops.gradient_operators(pos, cells)
    field = 2.0 * pos[:, 0] - 3.0 * pos[:, 1] + 0.5
    assert np.allclose(dx @ field, 2.0) and np.allclose(dy @ field, -3.0)
    dx_r, dy_r = mesh_ops.gradient_operators(pos, cells[:, ::-1])
    assert np.allclose(dx_r @ field, 2.0) and np.allclose(dy_r @ field, -3.0)


def test_vorticity_of_rigid_rotation_and_of_shear():
    pos, cells = _mesh()
    dx, dy = mesh_ops.gradient_operators(pos, cells)
    rotation = np.stack([-pos[:, 1], pos[:, 0]], axis=1)  # vorticity 2
    shear = np.stack([4.0 * pos[:, 1], np.zeros(len(pos))], axis=1)  # vorticity -4
    fields = np.stack([rotation, shear])
    omega = mesh_ops.vorticity(fields, dx, dy)
    assert omega.shape == (2, len(pos))
    assert np.allclose(omega[0], 2.0) and np.allclose(omega[1], -4.0)


def test_relative_l2_and_rmse():
    pos, cells = _mesh()
    w = mesh_ops.nodal_areas(pos, cells)
    rng = np.random.default_rng(0)
    true = rng.normal(size=(4, len(pos), 2))
    assert np.allclose(metrics.relative_l2(true, true, w), 0.0)
    assert np.allclose(metrics.relative_l2(1.1 * true, true, w), 0.1)
    assert np.allclose(metrics.relative_l2(np.zeros_like(true), true, w), 1.0)
    shifted = true + np.array([3.0, 4.0])
    assert np.allclose(metrics.nodal_rmse(shifted, true), 5.0)


def test_kinetic_energy_of_uniform_flow():
    pos, cells = _mesh()
    w = mesh_ops.nodal_areas(pos, cells)
    velocity = np.tile(np.array([2.0, 1.0]), (3, len(pos), 1))
    assert np.allclose(metrics.kinetic_energy(velocity, w), 0.5 * 5.0 * 0.64)


def test_lag_minimised_error_separates_phase_from_amplitude():
    pos, cells = _mesh()
    w = mesh_ops.nodal_areas(pos, cells)
    t = np.arange(60)[:, None, None]
    x = pos[None, :, 0:1]
    true = np.concatenate([np.sin(0.3 * t + 4.0 * x), np.cos(0.3 * t + 4.0 * x)], axis=-1)
    delayed = np.concatenate([np.sin(0.3 * (t - 3) + 4.0 * x), np.cos(0.3 * (t - 3) + 4.0 * x)], axis=-1)
    plain = metrics.relative_l2(delayed, true, w)
    aligned, lag = metrics.lag_minimised_error(delayed, true, w, max_lag=5)
    interior = slice(5, 55)
    assert (plain[interior] > 0.5).all()
    assert np.allclose(aligned[interior], 0.0, atol=1e-12) and (lag[interior] == -3).all()
    assert (aligned <= plain + 1e-12).all()  # zero shift is always admissible
    # a wrong amplitude is not removed by any shift
    damped, _ = metrics.lag_minimised_error(0.5 * true, true, w, max_lag=5)
    assert np.allclose(damped[interior], 0.5)
    zero_lag, _ = metrics.lag_minimised_error(delayed, true, w, max_lag=0)
    assert np.allclose(zero_lag, plain)


def test_fluctuation_ratio_and_first_exceedance():
    pos, cells = _mesh()
    w = mesh_ops.nodal_areas(pos, cells)
    steady = np.ones((40, len(pos), 2))
    assert metrics.fluctuation_ratio(steady, w) == 0.0
    t = np.arange(40)[:, None, None]
    unsteady = steady + 0.5 * np.sin(0.7 * t)
    assert metrics.fluctuation_ratio(unsteady, w) > 0.2
    assert metrics.first_exceedance(np.array([0.1, 0.5, 1.2, 0.3]), 1.0) == 2
    assert metrics.first_exceedance(np.array([0.1, np.nan]), 1.0) == 1
    assert metrics.first_exceedance(np.array([0.1, 0.2]), 1.0) == -1
