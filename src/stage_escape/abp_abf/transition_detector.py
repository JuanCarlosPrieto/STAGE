from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ._validation import as_scalar

TransitionDirection = Literal["above", "below"]


@dataclass(frozen=True, slots=True)
class TransitionDetector:
    """Detect threshold crossings of a scalar collective variable.

    Thresholds are inclusive. A one-dimensional array passed to
    ``transition_mask`` is interpreted as a trajectory of scalar positions;
    multidimensional trajectories must have shape ``(n_points, dimension)``.
    """

    collective_variable: Callable[[np.ndarray], float]
    threshold: float
    direction: TransitionDirection = "above"

    def __post_init__(self) -> None:
        if not callable(self.collective_variable):
            raise TypeError("collective_variable must be callable.")
        if self.direction not in {"above", "below"}:
            raise ValueError("direction must be either 'above' or 'below'.")
        threshold = float(self.threshold)
        if np.isnan(threshold):
            raise ValueError("threshold must not be NaN.")
        object.__setattr__(self, "threshold", threshold)

    def value_at(self, position) -> float:
        position = np.atleast_1d(np.asarray(position, dtype=float))
        if not np.all(np.isfinite(position)):
            raise ValueError("position must contain only finite values.")
        return as_scalar(
            self.collective_variable(position),
            name="collective-variable value",
        )

    def is_transition(self, position) -> bool:
        value = self.value_at(position)
        if self.direction == "above":
            return value >= self.threshold
        return value <= self.threshold

    def transition_mask(self, positions) -> np.ndarray:
        positions = np.asarray(positions, dtype=float)
        if positions.ndim == 0:
            trajectory = positions.reshape(1, 1)
        elif positions.ndim == 1:
            trajectory = positions.reshape(-1, 1)
        elif positions.ndim == 2:
            trajectory = positions
        else:
            raise ValueError(
                "positions must have shape (N,) or (N, dimension)."
            )
        if not np.all(np.isfinite(trajectory)):
            raise ValueError("positions must contain only finite values.")
        return np.fromiter(
            (self.is_transition(position) for position in trajectory),
            dtype=bool,
            count=len(trajectory),
        )

    def first_transition_index(self, positions) -> int | None:
        indices = np.flatnonzero(self.transition_mask(positions))
        return int(indices[0]) if indices.size else None
