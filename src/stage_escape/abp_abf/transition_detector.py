from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np


TransitionDirection = Literal["above", "below"]


@dataclass(frozen=True, slots=True)
class TransitionDetector:
    collective_variable: Callable[[np.ndarray], float]
    threshold: float
    direction: TransitionDirection = "above"

    def __post_init__(self):
        if self.direction not in {"above", "below"}:
            raise ValueError(
                "direction must be 'above' or 'below'."
            )

    def is_transition(self, position) -> bool:
        value = float(
            self.collective_variable(
                np.asarray(position, dtype=float)
            )
        )

        if self.direction == "above":
            return value > self.threshold

        return value < self.threshold

    def first_transition_index(
        self,
        positions,
    ) -> int | None:
        positions = np.asarray(
            positions,
            dtype=float,
        )

        for index, position in enumerate(positions):
            if self.is_transition(position):
                return index

        return None