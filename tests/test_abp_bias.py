import numpy as np

from stage_escape.abp_abf.abp_bias import (
    gaussian_bias_gradient,
    gaussian_bias_value,
)


def test_gaussian_value_and_gradient_at_center():
    center = np.array([[0.5, -0.25]])
    value = gaussian_bias_value(center[0], center, height=2.0, sigma=0.3)
    gradient = gaussian_bias_gradient(center[0], center, height=2.0, sigma=0.3)
    assert np.isclose(value, 2.0)
    np.testing.assert_allclose(gradient, np.zeros(2))


def test_gaussian_gradient_matches_finite_difference():
    position = np.array([0.2])
    centers = np.array([[0.0], [0.7]])
    epsilon = 1e-6
    numerical = (
        gaussian_bias_value(position + epsilon, centers, 0.4, 0.2)
        - gaussian_bias_value(position - epsilon, centers, 0.4, 0.2)
    ) / (2.0 * epsilon)
    analytic = gaussian_bias_gradient(position, centers, 0.4, 0.2)[0]
    assert np.isclose(analytic, numerical, rtol=1e-5, atol=1e-7)
