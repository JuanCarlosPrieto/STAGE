import numpy as np

from stage_escape.abp_abf import ABPMetaDynamics, TransitionDetector


def build_simulation(seed, potential, detector=None):
    return ABPMetaDynamics(
        deposition_stride=5,
        transition_detector=detector,
        delta_t=1e-3,
        dimension=1,
        D=0.2,
        initial_position=np.array([0.0]),
        b=potential,
        W=0.05,
        sigma=0.2,
        seed=seed,
    )


def test_same_seed_reproduces_metadynamics(zero_potential_1d):
    first = build_simulation(12345, zero_potential_1d)
    second = build_simulation(12345, zero_potential_1d)
    result_1 = first.run(max_steps=20)
    result_2 = second.run(max_steps=20)
    np.testing.assert_allclose(result_1.positions, result_2.positions)
    np.testing.assert_allclose(result_1.weights, result_2.weights)
    np.testing.assert_allclose(result_1.centers, result_2.centers)
    assert result_1.physical_time == result_2.physical_time


def test_deposition_stride_and_result_lengths(zero_potential_1d):
    result = build_simulation(123, zero_potential_1d).run(max_steps=12)
    assert result.n_steps == 12
    assert len(result.positions) == len(result.weights) == 13
    assert len(result.centers) == 2
    assert result.termination_reason == "max_steps"


def test_initial_transition_stops_without_step(zero_potential_1d):
    detector = TransitionDetector(lambda x: x[0], threshold=0.0)
    result = build_simulation(123, zero_potential_1d, detector).run(max_steps=10)
    assert result.transition_index == 0
    assert result.n_steps == 0
    assert result.physical_time == 0.0


def test_online_and_posthoc_transition_indices_agree(zero_potential_1d):
    detector = TransitionDetector(lambda x: x[0], threshold=-10.0)
    result = build_simulation(123, zero_potential_1d, detector).run(max_steps=10)
    assert detector.first_transition_index(result.positions) == result.transition_index
