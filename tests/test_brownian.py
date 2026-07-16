import numpy as np
import pytest

from stage_escape import BrownianMotion


def test_same_seed_produces_same_brownian_path():
    simulation_1 = BrownianMotion(
        deposition_stride=100,
        delta_t=1e-3,
        dimension=2,
        D=0.5,
        seed=12345,
    )

    simulation_2 = BrownianMotion(
        deposition_stride=100,
        delta_t=1e-3,
        dimension=2,
        D=0.5,
        seed=12345,
    )

    path_1 = simulation_1.simulate()
    path_2 = simulation_2.simulate()

    np.testing.assert_allclose(path_1, path_2)


def test_different_seeds_produce_different_paths():
    simulation_1 = BrownianMotion(
        deposition_stride=100,
        delta_t=1e-3,
        dimension=1,
        D=0.5,
        seed=12345,
    )

    simulation_2 = BrownianMotion(
        deposition_stride=100,
        delta_t=1e-3,
        dimension=1,
        D=0.5,
        seed=67890,
    )

    path_1 = simulation_1.simulate()
    path_2 = simulation_2.simulate()

    assert not np.allclose(path_1, path_2)


def test_reset_reproduces_path():
    simulation = BrownianMotion(
        deposition_stride=100,
        delta_t=1e-3,
        dimension=1,
        D=0.5,
        seed=12345,
    )

    path_1 = simulation.simulate().copy()
    path_2 = simulation.simulate(reset=True).copy()

    np.testing.assert_allclose(path_1, path_2)


def test_seed_and_rng_cannot_be_given_together():
    rng = np.random.default_rng(12345)

    with pytest.raises(ValueError):
        BrownianMotion(
            deposition_stride=10,
            delta_t=1e-3,
            seed=12345,
            rng=rng,
        )