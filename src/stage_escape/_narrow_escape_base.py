from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .narrow_escape_result import NarrowEscapeResult


class NarrowEscapeBase(ABC):
    """Shared validation, state handling and result construction."""

    def __init__(self, brownian_motion, surface, escapes) -> None:
        required = ("positions", "dimension", "delta_t", "n_steps", "step", "reset")
        missing = [name for name in required if not hasattr(brownian_motion, name)]
        if missing:
            raise TypeError(
                "brownian_motion is missing required attributes: "
                + ", ".join(missing)
            )
        if not callable(getattr(surface, "is_inside", None)):
            raise TypeError("surface must provide is_inside(point).")
        self.brownian_motion = brownian_motion
        self.surface = surface
        self.escapes = tuple(escapes)
        if not self.escapes:
            raise ValueError("At least one escape condition must be provided.")
        if not all(
            callable(getattr(escape, "is_valid_escape", None))
            for escape in self.escapes
        ):
            raise TypeError("Each escape must provide is_valid_escape(point).")
        self._rejected_steps = 0
        self._validate_initial_state()

    def _validate_initial_state(self) -> None:
        position = np.asarray(self.brownian_motion.positions[0], dtype=float)
        expected = (self.brownian_motion.dimension,)
        if position.shape != expected:
            raise ValueError("The initial position has an inconsistent dimension.")
        if not self.surface.is_inside(position):
            raise ValueError("The initial position must be inside the domain.")

    def check_escape(self, point) -> bool:
        return self.escape_index(point) is not None

    def escape_index(self, point) -> int | None:
        for index, escape in enumerate(self.escapes):
            if escape.is_valid_escape(point):
                return index
        return None

    def _replace_last_position(self, position) -> None:
        position = np.asarray(position, dtype=float).reshape(
            self.brownian_motion.dimension
        )
        self.brownian_motion.positions[-1] = position.copy()

    def _reset_state(self) -> None:
        self.brownian_motion.reset()
        self._rejected_steps = 0
        self._validate_initial_state()

    def _result(
        self,
        *,
        attempted_steps: int,
        rejected_steps: int,
        escape_index: int | None = None,
        escape_point=None,
        escape_time: float | None = None,
    ) -> NarrowEscapeResult:
        escaped = escape_index is not None
        return NarrowEscapeResult(
            positions=np.asarray(self.brownian_motion.positions, dtype=float),
            delta_t=self.brownian_motion.delta_t,
            attempted_steps=attempted_steps,
            rejected_steps=rejected_steps,
            termination_reason="escape" if escaped else "max_steps",
            escape_index=escape_index,
            escape_point=escape_point,
            escape_time=escape_time,
        )

    @abstractmethod
    def run(self, max_steps=1_000_000, *, reset: bool = False) -> NarrowEscapeResult:
        raise NotImplementedError
