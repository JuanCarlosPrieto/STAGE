from __future__ import annotations

from typing import Any

import numpy as np

from ._validation import (
    as_position,
    prepare_rng,
    validate_positive_float,
    validate_positive_int,
    validate_transition_detector,
)


class AdaptiveSimulationBase:
    """Internal state shared by ABP and ABF integrators."""

    def __init__(
        self,
        *,
        transition_detector: Any,
        delta_t: float,
        dimension: int,
        D: float,
        initial_position,
        seed: int | None,
        rng: np.random.Generator | None,
    ) -> None:
        self.delta_t = validate_positive_float("delta_t", delta_t)
        self.dimension = validate_positive_int("dimension", dimension)
        self.D = validate_positive_float("D", D)
        validate_transition_detector(transition_detector)
        self.transition_detector = transition_detector
        self.seed = seed
        self.rng, self._external_rng = prepare_rng(seed=seed, rng=rng)
        default_position = np.zeros(self.dimension, dtype=float)
        self.initial_position = as_position(
            default_position if initial_position is None else initial_position,
            self.dimension,
            name="initial_position",
        )

    def _reset_common(self, *, reset_rng: bool) -> None:
        if reset_rng and not self._external_rng:
            self.rng = np.random.default_rng(self.seed)
        self.positions = [self.initial_position.copy()]
        self.real_time = 0.0
        self.steps_completed = 0
        self.transition_index: int | None = None
        self.termination_reason: str | None = None

    def _last_position(self) -> np.ndarray:
        return np.asarray(self.positions[-1], dtype=float)

    def _noise(self) -> np.ndarray:
        return (
            np.sqrt(2.0 * self.D * self.delta_t)
            * self.rng.standard_normal(self.dimension)
        )

    def _validated_new_position(self, position) -> np.ndarray:
        result = as_position(position, self.dimension, name="new position")
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("The new position is not finite.")
        return result

    def _transition_detected_at(self, position) -> bool:
        return (
            self.transition_detector is not None
            and self.transition_detector.is_transition(position)
        )

    def _mark_transition(self) -> None:
        self.transition_index = len(self.positions) - 1
        self.termination_reason = "transition"

    def _remaining_steps(self, max_steps: int) -> int:
        """Return steps remaining before the total trajectory cap."""
        max_steps = validate_positive_int("max_steps", max_steps)
        return max(max_steps - self.steps_completed, 0)

    @property
    def n_steps(self) -> int:
        return self.steps_completed
