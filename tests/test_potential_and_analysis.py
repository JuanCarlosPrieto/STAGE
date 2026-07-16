import numpy as np
import pytest

from stage_escape.abp_abf import (
    Potential,
    extract_coordinate,
    theoretical_density_1d,
    weighted_histogram_density,
)


def quadratic_potential():
    return Potential(
        1,
        lambda x: 0.5 * x[0] ** 2,
        lambda x: np.array([x[0]]),
        lambda _x: np.array([[1.0]]),
    )


def test_add_gaussian_does_not_recurse():
    potential = quadratic_potential()
    potential.add_gaussian(center=[0.0], height=2.0, width=0.5)
    assert potential.potential_at([0.0]) == pytest.approx(2.0)
    np.testing.assert_allclose(potential.potential_prime_at([0.0]), [0.0])
    assert potential.potential_at([1.0]) > 0.5


def test_with_gaussian_does_not_mutate_base():
    base = quadratic_potential()
    combined = base.with_gaussian(center=[0.0], height=1.0, width=1.0)
    assert base.potential_at([0.0]) == 0.0
    assert combined.potential_at([0.0]) == 1.0


def test_numerical_derivatives():
    potential = Potential(1, lambda x: x[0] ** 4)
    np.testing.assert_allclose(potential.potential_prime_at([2.0]), [32.0], rtol=1e-5)
    np.testing.assert_allclose(potential.potential_biprime_at([2.0]), [[48.0]], rtol=1e-4)


def test_weighted_density_is_normalized():
    centers, density, counts, edges = weighted_histogram_density(
        values=np.array([-0.75, -0.25, 0.25, 0.75]),
        weights=np.ones(4),
        bins=4,
        value_range=(-1.0, 1.0),
    )
    assert len(centers) == len(density) == len(counts) == 4
    assert np.sum(density * np.diff(edges)) == pytest.approx(1.0)


def test_theoretical_density_is_normalized():
    x = np.linspace(-5, 5, 1001)
    density = theoretical_density_1d(quadratic_potential(), x, D=1.0)
    assert np.trapezoid(density, x) == pytest.approx(1.0, rel=1e-6)


def test_extract_coordinate():
    trajectory = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(extract_coordinate(trajectory, axis=1), [2.0, 4.0])
