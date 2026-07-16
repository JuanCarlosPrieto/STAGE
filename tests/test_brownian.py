import numpy as np
import pytest

from stage_escape import BrownianMotion


def test_same_seed_reproduces_path():
    first = BrownianMotion(5, 0.01, dimension=2, seed=123)
    second = BrownianMotion(5, 0.01, dimension=2, seed=123)
    np.testing.assert_allclose(first.simulate(), second.simulate())


def test_step_advances_exactly_once():
    motion = BrownianMotion(5, 0.01, dimension=2, seed=1)
    motion.step()
    assert motion.n_steps == 1
    assert np.asarray(motion.positions).shape == (2, 2)


def test_reset_replays_internal_rng():
    motion = BrownianMotion(4, 0.01, dimension=1, seed=7)
    first = motion.simulate().copy()
    second = motion.simulate(reset=True).copy()
    np.testing.assert_allclose(first, second)


def test_drift_shape_is_validated():
    motion = BrownianMotion(3, 0.01, dimension=2, b=lambda _x: 1.0, seed=1)
    proposal = motion.propose()
    assert proposal.shape == (2,)

    bad = BrownianMotion(3, 0.01, dimension=2, b=lambda _x: [1.0], seed=1)
    with pytest.raises(ValueError, match="Drift"):
        bad.propose()
