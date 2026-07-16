import numpy as np
import pytest

from stage_escape.abp_abf import Potential, TransitionDetector


@pytest.fixture
def zero_potential_1d():
    return Potential.zero(1)


@pytest.fixture
def quadratic_potential_1d():
    return Potential(
        dimension=1,
        function=lambda x: 0.5 * float(x[0]) ** 2,
        first_derivative=lambda x: np.array([float(x[0])]),
        second_derivative=lambda x: np.array([[1.0]]),
    )


@pytest.fixture
def inactive_detector():
    return TransitionDetector(
        collective_variable=lambda x: float(x[0]),
        threshold=np.inf,
    )
