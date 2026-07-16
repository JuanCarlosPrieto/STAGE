import numpy as np
import pytest

from stage_escape.abp_abf import TransitionDetector


def test_above_threshold_is_inclusive():
    detector = TransitionDetector(lambda x: x[0], threshold=1.0)
    assert detector.is_transition(np.array([1.0]))


def test_below_threshold_is_inclusive():
    detector = TransitionDetector(
        lambda x: x[0], threshold=-1.0, direction="below"
    )
    assert detector.is_transition(np.array([-1.0]))


def test_first_transition_index_and_mask_agree():
    detector = TransitionDetector(lambda x: x[0], threshold=0.5)
    positions = np.array([[0.0], [0.2], [0.5], [0.9]])
    np.testing.assert_array_equal(
        detector.transition_mask(positions),
        np.array([False, False, True, True]),
    )
    assert detector.first_transition_index(positions) == 2


def test_no_transition_returns_none():
    detector = TransitionDetector(lambda x: x[0], threshold=2.0)
    assert detector.first_transition_index([[0.0], [1.0]]) is None


def test_collective_variable_must_be_scalar():
    detector = TransitionDetector(lambda x: x, threshold=0.0)
    with pytest.raises(ValueError):
        detector.is_transition(np.array([0.0, 1.0]))
