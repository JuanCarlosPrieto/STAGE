import numpy as np

from stage_escape.abp_abf import Potential


def test_centered_finite_difference_gradient_and_hessian():
    potential = Potential(
        dimension=2,
        function=lambda x: x[0] ** 2 + 3.0 * x[0] * x[1] + 2.0 * x[1] ** 2,
    )
    point = np.array([0.4, -0.2])
    expected_gradient = np.array(
        [2.0 * point[0] + 3.0 * point[1], 3.0 * point[0] + 4.0 * point[1]]
    )
    expected_hessian = np.array([[2.0, 3.0], [3.0, 4.0]])
    np.testing.assert_allclose(
        potential.potential_prime_at(point), expected_gradient, rtol=1e-5, atol=1e-7
    )
    np.testing.assert_allclose(
        potential.potential_biprime_at(point), expected_hessian, rtol=1e-5, atol=1e-6
    )


def test_with_gaussian_does_not_mutate_original():
    original = Potential.zero(1)
    modified = original.with_gaussian(center=[0.0], height=2.0, width=0.5)
    assert original.potential_at([0.0]) == 0.0
    assert modified.potential_at([0.0]) == 2.0
