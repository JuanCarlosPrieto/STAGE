from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from ._validation import as_scalar


TransitionDirection = Literal["above", "below"]


@dataclass(frozen=True, slots=True)
class TransitionDetector:
    """Detect a transition from a scalar collective variable.

    The threshold is inclusive: ``above`` means ``value >= threshold`` and
    ``below`` means ``value <= threshold``. The detector is deliberately
    stateless; trajectory storage belongs to the simulator or result object.
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
            positions = positions.reshape(1)
        return np.fromiter(
            (self.is_transition(position) for position in positions),
            dtype=bool,
            count=len(positions),
        )

    def first_transition_index(self, positions) -> int | None:
        mask = self.transition_mask(positions)
        indices = np.flatnonzero(mask)
        return int(indices[0]) if indices.size else None
