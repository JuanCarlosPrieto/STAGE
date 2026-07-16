import numpy as np
import pytest

from stage_escape.abp_abf import ABFRealTime, TransitionDetector


def build_simulation(seed, potential, detector=None, **kwargs):
    return ABFRealTime(
        transition_detector=detector,
        delta_t=1e-3,
        D=0.2,
        initial_position=np.array([0.25]),
        b=potential,
        bins=40,
        value_range=(-10.0, 10.0),
        seed=seed,
        profile_update_stride=1,
        **kwargs,
    )


def test_same_seed_reproduces_abf(quadratic_potential_1d):
    first = build_simulation(12345, quadratic_potential_1d)
    second = build_simulation(12345, quadratic_potential_1d)
    result_1 = first.run(max_steps=40)
    result_2 = second.run(max_steps=40)
    np.testing.assert_allclose(result_1.positions, result_2.positions)
    np.testing.assert_allclose(result_1.force_bias, result_2.force_bias)
    np.testing.assert_array_equal(result_1.visit_counts, result_2.visit_counts)
    assert result_1.physical_time == result_2.physical_time


def test_abf_updates_force_at_every_step(quadratic_potential_1d):
    result = build_simulation(123, quadratic_potential_1d).run(max_steps=40)
    assert result.n_steps == 40
    assert np.sum(result.visit_counts) == 40
    assert result.positions.shape == (41, 1)


def test_abf_reconstructs_nonzero_profiles(quadratic_potential_1d):
    result = build_simulation(123, quadratic_potential_1d).run(max_steps=100)
    assert np.all(np.isfinite(result.bias_potential))
    assert not np.allclose(result.force_bias, 0.0)
    assert not np.allclose(result.bias_potential, 0.0)


def test_initial_transition_stops_without_step(quadratic_potential_1d):
    detector = TransitionDetector(lambda x: x[0], threshold=0.25)
    result = build_simulation(123, quadratic_potential_1d, detector).run(
        max_steps=10
    )
    assert result.transition_index == 0
    assert result.n_steps == 0


def test_out_of_range_policy_is_explicit(quadratic_potential_1d):
    simulation = ABFRealTime(
        transition_detector=None,
        delta_t=1e-3,
        D=0.2,
        initial_position=[2.0],
        b=quadratic_potential_1d,
        bins=10,
        value_range=(-1.0, 1.0),
        seed=123,
        out_of_range="clip",
    )
    assert simulation.position_to_bin([2.0]) == 9

    with pytest.raises(ValueError):
        ABFRealTime(
            transition_detector=None,
            delta_t=1e-3,
            D=0.2,
            initial_position=[2.0],
            b=quadratic_potential_1d,
            bins=10,
            value_range=(-1.0, 1.0),
            seed=123,
            out_of_range="raise",
        )
