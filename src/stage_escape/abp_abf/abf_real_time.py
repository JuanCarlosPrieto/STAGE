from __future__ import annotations

from typing import Literal

import numpy as np

from ._adaptive_base import AdaptiveSimulationBase
from ._validation import (
    as_position,
    safe_exponential,
    validate_positive_int,
)
from .abf_profiles import reconstruct_abf_profiles
from .potential import Potential
from .results import ABFResult, TerminationReason
from .transition_detector import TransitionDetector

OutOfRangePolicy = Literal["raise", "clip"]


class ABFRealTime(AdaptiveSimulationBase):
    """One-dimensional adaptive biasing force simulation.

    ``force_bias`` stores the running estimate of the physical potential
    derivative in each bin. The resulting positive force contribution cancels
    the physical force ``-V'`` as the estimator converges.
    """

    def __init__(
        self,
        transition_detector,
        delta_t,
        D=1.0,
        initial_position=None,
        b=None,
        bins=100,
        value_range=(-3.0, 3.0),
        seed=None,
        rng=None,
        profile_update_stride=1,
        out_of_range: OutOfRangePolicy = "raise",
    ) -> None:
        self.bins = validate_positive_int("bins", bins, minimum=2)
        self.profile_update_stride = validate_positive_int(
            "profile_update_stride", profile_update_stride
        )
        if out_of_range not in {"raise", "clip"}:
            raise ValueError("out_of_range must be either 'raise' or 'clip'.")
        self.out_of_range: OutOfRangePolicy = out_of_range
        try:
            lower, upper = value_range
        except (TypeError, ValueError) as error:
            raise ValueError(
                "value_range must contain exactly two values."
            ) from error
        lower = float(lower)
        upper = float(upper)
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            raise ValueError("value_range must contain finite increasing bounds.")
        self.value_range = (lower, upper)

        super().__init__(
            transition_detector=transition_detector,
            delta_t=delta_t,
            dimension=1,
            D=D,
            initial_position=initial_position,
            seed=seed,
            rng=rng,
        )
        self.transition_detector: TransitionDetector | None
        self._range_adjusted_position(self.initial_position)
        self.b = Potential.zero(1) if b is None else b
        if not callable(getattr(self.b, "potential_prime_at", None)):
            raise TypeError("b must provide potential_prime_at(position).")
        if getattr(self.b, "dimension", 1) != 1:
            raise ValueError("ABFRealTime requires a one-dimensional potential.")
        self.reset(reset_rng=False)

    def reset(self, *, reset_rng: bool = True) -> None:
        self._reset_common(reset_rng=reset_rng)
        self.force_bias = np.zeros(self.bins, dtype=float)
        self.number_of_copies = np.zeros(self.bins, dtype=np.int64)
        self.bin_edges = np.linspace(
            self.value_range[0], self.value_range[1], self.bins + 1
        )
        self.free_energy_profile = np.zeros(self.bins, dtype=float)
        self.bias_potential = np.zeros(self.bins, dtype=float)

    @property
    def visit_counts(self) -> np.ndarray:
        """Alias using the result-object terminology."""
        return self.number_of_copies

    def _range_adjusted_position(self, position) -> float:
        x = float(as_position(position, 1)[0])
        lower, upper = self.value_range
        if lower <= x <= upper:
            return x
        if self.out_of_range == "clip":
            return float(np.clip(x, lower, upper))
        raise ValueError(
            f"ABF position x={x} lies outside value_range={self.value_range}."
        )

    def position_to_bin(self, position) -> int:
        x = self._range_adjusted_position(position)
        lower, upper = self.value_range
        normalized = (x - lower) / (upper - lower)
        return int(np.clip(int(normalized * self.bins), 0, self.bins - 1))

    def _current_bin(self) -> int:
        return self.position_to_bin(self._last_position())

    def _potential_gradient_scalar(self, position) -> float:
        gradient = np.asarray(
            self.b.potential_prime_at(position), dtype=float
        ).reshape(-1)
        if gradient.size != 1:
            raise ValueError("ABFRealTime requires a scalar potential gradient.")
        value = float(gradient[0])
        if not np.isfinite(value):
            raise FloatingPointError("The physical gradient is not finite.")
        return value

    def _update_force_estimator(self, position) -> int:
        current_bin = self.position_to_bin(position)
        sample = self._potential_gradient_scalar(position)
        count = int(self.number_of_copies[current_bin])
        self.force_bias[current_bin] += (
            sample - self.force_bias[current_bin]
        ) / (count + 1)
        self.number_of_copies[current_bin] = count + 1
        return current_bin

    def update_profiles(self) -> tuple[np.ndarray, np.ndarray]:
        (
            self.bin_edges,
            self.free_energy_profile,
            self.bias_potential,
        ) = reconstruct_abf_profiles(
            force_bias=self.force_bias,
            value_range=self.value_range,
        )
        return self.free_energy_profile, self.bias_potential

    def step(self) -> bool:
        """Advance the ABF dynamics by exactly one integration step."""
        if self.transition_index is not None:
            return True

        current_position = self._last_position()
        current_bin = self._current_bin()
        physical_gradient = self._potential_gradient_scalar(current_position)
        bias_energy = float(self.bias_potential[current_bin])
        drift = np.array(
            [
                (-physical_gradient + self.force_bias[current_bin])
                * self.delta_t
            ],
            dtype=float,
        )
        new_position = self._validated_new_position(
            current_position + drift + self._noise()
        )
        # In clip mode, only the bin lookup is clipped; the physical trajectory
        # and physical force evaluation remain at the actual position.
        self._range_adjusted_position(new_position)
        time_weight = safe_exponential(
            bias_energy / self.D,
            name="ABF reweighting factor",
        )

        self.positions.append(new_position.copy())
        self.real_time += time_weight * self.delta_t
        self.steps_completed += 1
        self._update_force_estimator(new_position)
        if self.steps_completed % self.profile_update_stride == 0:
            self.update_profiles()

        if self._transition_detected_at(new_position):
            self._mark_transition()
            return True
        return False

    def result(
        self,
        termination_reason: TerminationReason | None = None,
    ) -> ABFResult:
        if termination_reason is None:
            termination_reason = (
                "transition" if self.transition_index is not None else "max_steps"
            )
        self.update_profiles()
        return ABFResult(
            method="abf",
            positions=np.asarray(self.positions, dtype=float).reshape(-1, 1),
            delta_t=self.delta_t,
            diffusion=self.D,
            seed=self.seed,
            transition_index=self.transition_index,
            physical_time=float(self.real_time),
            termination_reason=termination_reason,
            force_bias=self.force_bias.copy(),
            visit_counts=self.number_of_copies.copy(),
            bin_edges=self.bin_edges.copy(),
            free_energy=self.free_energy_profile.copy(),
            bias_potential=self.bias_potential.copy(),
            metadata={
                "dimension": 1,
                "bins": self.bins,
                "value_range": tuple(self.value_range),
                "profile_update_stride": self.profile_update_stride,
                "out_of_range": self.out_of_range,
            },
        )

    def run(self, max_steps=1_000_000, *, reset: bool = False) -> ABFResult:
        """Run until transition or until the total step cap is reached."""
        if reset:
            self.reset()
        if self._transition_detected_at(self._last_position()):
            self._mark_transition()
            return self.result("transition")
        for _ in range(self._remaining_steps(max_steps)):
            if self.step():
                return self.result("transition")
        self.termination_reason = "max_steps"
        return self.result("max_steps")

    def simulate(self, max_iters=1_000_000, *, reset: bool = False) -> ABFResult:
        """Backward-compatible alias for :meth:`run`."""
        return self.run(max_steps=max_iters, reset=reset)
