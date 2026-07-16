import numpy as np
import pytest

from stage_escape import EquivalentNarrowEscape, Escape, NaiveNarrowEscape, Surface


class ScriptedMotion:
    def __init__(self, initial_position, proposals, delta_t=0.1):
        self.initial_position = np.asarray(initial_position, dtype=float)
        self.dimension = self.initial_position.size
        self.delta_t = float(delta_t)
        self._proposals = [np.asarray(p, dtype=float) for p in proposals]
        self.positions = [self.initial_position.copy()]
        self._cursor = 0

    @property
    def n_steps(self):
        return len(self.positions) - 1

    def step(self):
        point = self._proposals[self._cursor].copy()
        self._cursor += 1
        self.positions.append(point)
        return point.copy()

    def reset(self):
        self.positions = [self.initial_position.copy()]
        self._cursor = 0


def square_surface():
    return Surface(
        "square",
        [
            lambda p: p[0] - 1.0,
            lambda p: -p[0] - 1.0,
            lambda p: p[1] - 1.0,
            lambda p: -p[1] - 1.0,
        ],
    )


def top_window():
    return Escape(
        [
            lambda p: np.isclose(p[1], 1.0, atol=1e-8),
            lambda p: -0.6 <= p[0] <= 0.6,
        ]
    )


def endpoint_region():
    return Escape([lambda p: p[1] >= 1.0, lambda p: -0.6 <= p[0] <= 0.6])


def test_surface_selects_earliest_crossing_at_corner_proposal():
    crossing = square_surface().first_boundary_intersection([0, 0], [2, 4])
    assert crossing.surface_index == 2
    assert crossing.theta == pytest.approx(0.25)
    np.testing.assert_allclose(crossing.point, [0.5, 1.0])


def test_direct_escape_uses_substep_crossing_time():
    motion = ScriptedMotion([0, 0], [[0.5, 2.0]], delta_t=0.2)
    result = NaiveNarrowEscape(motion, square_surface(), [top_window()]).run(1)
    assert result.escaped
    assert result.escape_index == 0
    np.testing.assert_allclose(result.escape_point, [0.25, 1.0])
    assert result.escape_time == pytest.approx(0.1)
    assert result.attempted_steps == 1


def test_direct_reflection_preserves_exact_step_cap_and_counts_rejections():
    motion = ScriptedMotion([0, 0], [[2, 0], [0.2, 0.2], [0.3, 0.3]])
    result = NaiveNarrowEscape(motion, square_surface(), [top_window()]).run(3)
    assert not result.escaped
    assert result.attempted_steps == 3
    assert result.rejected_steps == 1
    assert len(result.positions) == 4
    np.testing.assert_allclose(result.positions[1], [0, 0])


def test_equivalent_result_uses_same_contract():
    motion = ScriptedMotion([0, 0], [[0.1, 0.2], [0.2, 1.2]], delta_t=0.1)
    result = EquivalentNarrowEscape(
        motion, square_surface(), [endpoint_region()]
    ).run(2)
    assert result.escaped
    assert result.escape_index == 0
    assert result.escape_time == pytest.approx(0.2)
    assert result.attempted_steps == 2


def test_result_arrays_are_read_only():
    motion = ScriptedMotion([0, 0], [[0.1, 0.1]])
    result = NaiveNarrowEscape(motion, square_surface(), [top_window()]).run(1)
    with pytest.raises(ValueError):
        result.positions[0, 0] = 1.0
