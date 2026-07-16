import numpy as np

from stage_escape.abp_abf.abf_real_time import ABFRealTime
from stage_escape.abp_abf.potential import Potential
from stage_escape.abp_abf.transition_detector import TransitionDetector


def zero_potential():
    return Potential(
        dimension=1,
        function=lambda x: 0.0,
        first_derivative=lambda x: np.zeros_like(x, dtype=float),
        second_derivative=lambda x: np.zeros((1, 1), dtype=float),
    )


def inactive_detector():
    return TransitionDetector(
        collective_variable=lambda x: float(x[0]),
        threshold=np.inf,
    )

def quadratic_potential():
    """One-dimensional potential V(x) = x² / 2."""
    return Potential(
        dimension=1,
        function=lambda x: (
            0.5 * float(np.asarray(x)[0]) ** 2
        ),
        first_derivative=lambda x: np.array(
            [float(np.asarray(x)[0])],
            dtype=float,
        ),
        second_derivative=lambda x: np.array(
            [[1.0]],
            dtype=float,
        ),
    )


def build_quadratic_simulation(seed):
    return ABFRealTime(
        deposition_stride=10,
        td=inactive_detector(),
        delta_t=1e-3,
        D=0.2,
        initial_position=np.array([0.25]),
        b=quadratic_potential(),
        bins=40,
        range=(-10.0, 10.0),
        seed=seed,
    )


def build_simulation(seed):
    return ABFRealTime(
        deposition_stride=10,
        td=inactive_detector(),
        delta_t=1e-3,
        D=0.2,
        initial_position=np.array([0.0]),
        b=zero_potential(),
        bins=40,
        range=(-10.0, 10.0),
        seed=seed,
    )


def test_same_seed_reproduces_abf():
    simulation_1 = build_simulation(seed=12345)
    simulation_2 = build_simulation(seed=12345)

    simulation_1.simulate(max_iters=40)
    simulation_2.simulate(max_iters=40)

    np.testing.assert_allclose(
        np.asarray(simulation_1.positions),
        np.asarray(simulation_2.positions),
    )

    np.testing.assert_allclose(
        simulation_1.force_bias,
        simulation_2.force_bias,
    )

    np.testing.assert_allclose(
        simulation_1.number_of_copies,
        simulation_2.number_of_copies,
    )

    assert np.isclose(
        simulation_1.real_time,
        simulation_2.real_time,
    )

def test_abf_keeps_one_dimensional_state():
    simulation = build_simulation(seed=123)
    simulation.simulate(max_iters=40)

    positions = np.asarray(simulation.positions)

    assert positions.ndim == 2
    assert positions.shape[1] == 1
    assert simulation.force_bias.shape == (simulation.bins,)
    assert np.all(np.isfinite(positions))
    assert np.all(np.isfinite(simulation.force_bias))

def test_abf_updates_force_at_each_step():
    simulation = build_simulation(seed=123)
    simulation.simulate(max_iters=40)

    assert len(simulation.positions) == 41
    assert np.sum(simulation.number_of_copies) == 40

def quadratic_potential():
    return Potential(
        dimension=1,
        function=lambda x: 0.5 * float(x[0]) ** 2,
        first_derivative=lambda x: np.array([float(x[0])]),
        second_derivative=lambda x: np.array([[1.0]]),
    )

def test_abf_updates_bias_potential():
    simulation = build_quadratic_simulation(seed=123)
    simulation.simulate(max_iters=100)

    assert np.all(np.isfinite(simulation.bias_potential))
    assert not np.allclose(
        simulation.bias_potential,
        0.0,
    )

def test_abf_real_time_is_positive_and_finite():
    simulation = build_simulation(seed=123)

    simulation.simulate(max_iters=100)

    assert np.ndim(simulation.real_time) == 0
    assert np.isfinite(simulation.real_time), (
        f"real_time must be finite, got {simulation.real_time}"
    )
    assert simulation.real_time > 0.0, (
        f"real_time must be positive, got {simulation.real_time}"
    )
