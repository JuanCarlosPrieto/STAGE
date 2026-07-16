from __future__ import annotations

import matplotlib
import numpy as np
import pytest

from stage_escape.abp_abf import Potential

# Tests must be safe in headless environments such as CI and remote servers.
matplotlib.use("Agg")


@pytest.fixture
def zero_potential_1d() -> Potential:
    """One-dimensional zero potential with analytic derivatives."""
    return Potential(
        dimension=1,
        function=lambda _x: 0.0,
        first_derivative=lambda _x: np.zeros(1, dtype=float),
        second_derivative=lambda _x: np.zeros((1, 1), dtype=float),
    )


@pytest.fixture
def quadratic_potential_1d() -> Potential:
    """V(x) = x² / 2 with exact gradient and Hessian."""
    return Potential(
        dimension=1,
        function=lambda x: 0.5 * float(x[0]) ** 2,
        first_derivative=lambda x: np.array([float(x[0])], dtype=float),
        second_derivative=lambda _x: np.ones((1, 1), dtype=float),
    )
