import numpy as np

from stage_escape.abp_abf.abp_metadynamics import ABPMetaDynamics
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


def build_simulation(seed):
    return ABPMetaDynamics(
        deposition_stride=20,
        td=inactive_detector(),
        delta_t=1e-3,
        dimension=1,
        D=0.2,
        initial_position=np.array([0.0]),
        b=zero_potential(),
        W=0.05,
        sigma=0.2,
        seed=seed,
    )


def test_same_seed_reproduces_metadynamics():
    simulation_1 = build_simulation(seed=12345)
    simulation_2 = build_simulation(seed=12345)

    simulation_1.simulate(max_iters=4)
    simulation_2.simulate(max_iters=4)

    np.testing.assert_allclose(
        np.asarray(simulation_1.positions),
        np.asarray(simulation_2.positions),
    )

    np.testing.assert_allclose(
        np.asarray(simulation_1.weights),
        np.asarray(simulation_2.weights),
    )

    np.testing.assert_allclose(
        np.asarray(simulation_1.centers),
        np.asarray(simulation_2.centers),
    )

    assert np.isclose(
        simulation_1.real_time,
        simulation_2.real_time,
    )


def test_different_seeds_change_metadynamics_path():
    simulation_1 = build_simulation(seed=12345)
    simulation_2 = build_simulation(seed=67890)

    simulation_1.simulate(max_iters=4)
    simulation_2.simulate(max_iters=4)

    assert not np.allclose(
        np.asarray(simulation_1.positions),
        np.asarray(simulation_2.positions),
    )