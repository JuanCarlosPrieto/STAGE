from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np


TransitionDirection = Literal["above", "below"]


@dataclass(frozen=True, slots=True)
class TransitionDetector:
    """Detect transitions through a scalar collective variable.

    Parameters
    ----------
    collective_variable
        Function mapping one simulation position to one scalar.
    threshold
        Transition threshold.
    direction
        ``"above"`` detects values greater than or equal to the threshold.
        ``"below"`` detects values less than or equal to the threshold.
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
        """Evaluate the collective variable at one position."""
        position = np.atleast_1d(np.asarray(position, dtype=float))
        value = np.asarray(
            self.collective_variable(position),
            dtype=float,
        )

        if value.size != 1:
            raise ValueError(
                "collective_variable must return exactly one scalar value."
            )

        scalar_value = float(value.reshape(-1)[0])
        if np.isnan(scalar_value):
            raise ValueError("collective_variable returned NaN.")

        return scalar_value

    def is_transition(self, position) -> bool:
        """Return whether one position satisfies the transition criterion."""
        value = self.value_at(position)

        if self.direction == "above":
            return value >= self.threshold

        return value <= self.threshold

    def first_transition_index(self, positions) -> int | None:
        """Return the first transition index, or ``None`` if absent."""
        positions = np.asarray(positions, dtype=float)

        if positions.ndim == 0:
            positions = positions.reshape(1)

        for index, position in enumerate(positions):
            if self.is_transition(position):
                return index

        return None
