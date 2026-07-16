import numpy as np
import pytest

from stage_escape.abp_abf import (
    ABFRealTime,
    ABPMetaDynamics,
    Potential,
    TransitionDetector,
)


def quadratic_potential():
    return Potential(
        1,
        lambda x: 0.5 * x[0] ** 2,
        lambda x: np.array([x[0]]),
        lambda _x: np.array([[1.0]]),
    )


def never_transition():
    return TransitionDetector(lambda x: x[0], threshold=np.inf)


def test_abp_same_seed_reproduces_complete_result():
    kwargs = dict(
        deposition_stride=5,
        transition_detector=never_transition(),
        delta_t=1e-3,
        D=0.2,
        b=quadratic_potential(),
        W=0.05,
        sigma=0.2,
        seed=123,
    )
    first = ABPMetaDynamics(**kwargs).run(max_steps=40)
    second = ABPMetaDynamics(**kwargs).run(max_steps=40)
    np.testing.assert_allclose(first.positions, second.positions)
    np.testing.assert_allclose(first.weights, second.weights)
    np.testing.assert_allclose(first.centers, second.centers)
    assert first.n_steps == 40


def test_abp_continuation_uses_total_step_cap():
    simulation = ABPMetaDynamics(
        4, never_transition(), 1e-3, D=0.1, b=quadratic_potential(), seed=1
    )
    assert simulation.run(max_steps=10).n_steps == 10
    assert simulation.run(max_steps=25).n_steps == 25
    assert simulation.run(max_steps=20).n_steps == 25


def test_abf_updates_once_per_step_and_reproduces():
    kwargs = dict(
        transition_detector=never_transition(),
        delta_t=1e-3,
        D=0.1,
        b=quadratic_potential(),
        bins=40,
        value_range=(-4.0, 4.0),
        seed=123,
    )
    first = ABFRealTime(**kwargs).run(max_steps=40)
    second = ABFRealTime(**kwargs).run(max_steps=40)
    assert np.sum(first.visit_counts) == 40
    np.testing.assert_allclose(first.positions, second.positions)
    np.testing.assert_allclose(first.force_bias, second.force_bias)


def test_transition_at_initial_position_is_detected_without_step():
    detector = TransitionDetector(lambda x: x[0], threshold=0.0)
    result = ABPMetaDynamics(5, detector, 0.01, seed=1).run(max_steps=10)
    assert result.transition_index == 0
    assert result.n_steps == 0
    assert result.termination_reason == "transition"


def test_abf_rejects_noninteger_visit_counts_in_result_contract():
    simulation = ABFRealTime(
        never_transition(), 0.01, bins=4, value_range=(-2, 2), seed=1
    )
    simulation.number_of_copies = np.array([0.0, 1.5, 0.0, 0.0])
    with pytest.raises(ValueError, match="integer"):
        simulation.result()
