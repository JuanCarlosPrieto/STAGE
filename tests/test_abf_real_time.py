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
        positions=np.empty((0, 1)),
        cv=lambda x: float(np.asarray(x)[0]),
        threshold=np.inf,
    )


def build_simulation(seed):
    return ABFRealTime(
        num_steps=10,
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