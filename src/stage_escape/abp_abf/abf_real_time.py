from __future__ import annotations

import numpy as np

from .potential import Potential
from .results import ABFResult, TerminationReason
from .transition_detector import TransitionDetector


class ABFRealTime:
    """One-dimensional adaptive biasing force simulation."""

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
        profile_update_stride=100,
    ):
        if seed is not None and rng is not None:
            raise ValueError("Provide either seed or rng, not both.")

        delta_t = float(delta_t)
        D = float(D)

        if not np.isfinite(delta_t) or delta_t <= 0.0:
            raise ValueError("delta_t must be finite and strictly positive.")
        if not np.isfinite(D) or D <= 0.0:
            raise ValueError("D must be finite and strictly positive.")

        if (
            isinstance(bins, bool)
            or not isinstance(bins, (int, np.integer))
        ):
            raise TypeError("bins must be an integer.")
        if bins < 2:
            raise ValueError("bins must be greater than or equal to 2.")

        if (
            isinstance(profile_update_stride, bool)
            or not isinstance(
                profile_update_stride,
                (int, np.integer),
            )
        ):
            raise TypeError(
                "profile_update_stride must be an integer."
            )
        if profile_update_stride <= 0:
            raise ValueError(
                "profile_update_stride must be strictly positive."
            )

        try:
            lower, upper = value_range
        except (TypeError, ValueError) as error:
            raise ValueError(
                "value_range must contain exactly two values."
            ) from error

        lower = float(lower)
        upper = float(upper)

        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError("value_range bounds must be finite.")
        if upper <= lower:
            raise ValueError(
                "The upper value_range bound must be greater "
                "than the lower bound."
            )

        if (
            transition_detector is not None
            and not callable(
                getattr(transition_detector, "is_transition", None)
            )
        ):
            raise TypeError(
                "transition_detector must provide is_transition(position)."
            )

        self.transition_detector: TransitionDetector | None = (
            transition_detector
        )
        self.delta_t = delta_t
        self.dimension = 1
        self.D = D
        self.bins = int(bins)
        self.value_range = (lower, upper)
        self.profile_update_stride = int(profile_update_stride)
        self.seed = seed
        self.rng = rng if rng is not None else np.random.default_rng(seed)

        if initial_position is None:
            initial_position = np.zeros(1, dtype=float)
        else:
            initial_position = np.asarray(
                initial_position,
                dtype=float,
            ).reshape(-1)

        if initial_position.size != 1:
            raise ValueError(
                "ABFRealTime only supports one-dimensional positions."
            )
        if not np.all(np.isfinite(initial_position)):
            raise ValueError(
                "initial_position must contain only finite values."
            )

        self._validate_position_in_range(initial_position)

        if b is None:
            b = Potential(
                dimension=1,
                function=lambda x: 0.0,
                first_derivative=lambda x: np.zeros(1, dtype=float),
                second_derivative=lambda x: np.zeros((1, 1), dtype=float),
            )

        if not callable(getattr(b, "potential_prime_at", None)):
            raise TypeError(
                "b must provide potential_prime_at(position)."
            )
        if hasattr(b, "dimension") and b.dimension != 1:
            raise ValueError(
                "ABFRealTime requires a one-dimensional potential."
            )

        self.initial_position = initial_position.copy()
        self.b = b

        self.positions = [self.initial_position.copy()]
        self.force_bias = np.zeros(self.bins, dtype=float)
        self.number_of_copies = np.zeros(self.bins, dtype=float)

        self.bin_edges = np.linspace(lower, upper, self.bins + 1)
        self.free_energy_profile = np.zeros(self.bins, dtype=float)
        self.bias_potential = np.zeros(self.bins, dtype=float)

        self.real_time = 0.0
        self.steps_completed = 0
        self.transition_index = None
        self.termination_reason: TerminationReason | None = None

    def _last_position(self) -> np.ndarray:
        return np.asarray(self.positions[-1], dtype=float)

    @staticmethod
    def _position_scalar(position) -> float:
        position = np.asarray(position, dtype=float).reshape(-1)

        if position.size != 1:
            raise ValueError(
                "ABFRealTime only supports one-dimensional positions."
            )

        x = float(position[0])
        if not np.isfinite(x):
            raise FloatingPointError("The ABF position is not finite.")

        return x

    def _validate_position_in_range(self, position) -> float:
        x = self._position_scalar(position)
        lower, upper = self.value_range

        if x < lower or x > upper:
            raise ValueError(
                "ABF position outside value_range: "
                f"x={x}, value_range={self.value_range}."
            )

        return x

    def position_to_bin(self, position) -> int:
        """Map one position to its ABF bin."""
        x = self._validate_position_in_range(position)
        lower, upper = self.value_range

        normalized_position = (x - lower) / (upper - lower)
        bin_index = int(normalized_position * self.bins)

        return int(np.clip(bin_index, 0, self.bins - 1))

    def _current_bin(self) -> int:
        return self.position_to_bin(self._last_position())

    def _potential_gradient_scalar(self, position) -> float:
        gradient = np.asarray(
            self.b.potential_prime_at(position),
            dtype=float,
        ).reshape(-1)

        if gradient.size != 1:
            raise ValueError(
                "ABFRealTime requires a scalar potential gradient; "
                f"received {gradient.size} components."
            )

        scalar_gradient = float(gradient[0])
        if not np.isfinite(scalar_gradient):
            raise FloatingPointError(
                "The physical potential gradient is not finite."
            )

        return scalar_gradient

    def _update_force_estimator(self, position) -> int:
        """Update the running mean-force estimator in one bin."""
        current_bin = self.position_to_bin(position)
        sample_force = self._potential_gradient_scalar(position)

        count = self.number_of_copies[current_bin]
        self.force_bias[current_bin] = (
            count * self.force_bias[current_bin] + sample_force
        ) / (count + 1.0)
        self.number_of_copies[current_bin] = count + 1.0

        return current_bin

    def _transition_detected_at(self, position) -> bool:
        return (
            self.transition_detector is not None
            and self.transition_detector.is_transition(position)
        )

    def update_profiles(self):
        """Reconstruct free-energy and bias-potential profiles."""
        from .abf_profiles import reconstruct_abf_profiles

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
        """Advance the ABF dynamics by one integration step."""
        if self.transition_index is not None:
            return True

        current_position = self._last_position()
        current_bin = self._current_bin()
        physical_gradient = self._potential_gradient_scalar(
            current_position
        )

        applied_bias = float(self.bias_potential[current_bin])
        if not np.isfinite(applied_bias):
            raise FloatingPointError(
                "The ABF bias potential is not finite."
            )

        drift = np.array(
            [
                (
                    -physical_gradient
                    + self.force_bias[current_bin]
                )
                * self.delta_t
            ],
            dtype=float,
        )
        noise = (
            np.sqrt(2.0 * self.D * self.delta_t)
            * self.rng.standard_normal(1)
        )
        new_position = current_position + drift + noise

        self._validate_position_in_range(new_position)

        with np.errstate(over="ignore", invalid="ignore"):
            time_weight = float(np.exp(applied_bias / self.D))

        if not np.isfinite(time_weight):
            raise FloatingPointError(
                "The ABF reweighting factor is not finite."
            )

        self.positions.append(np.asarray(new_position, dtype=float))
        self.real_time += time_weight * self.delta_t
        self.steps_completed += 1

        self._update_force_estimator(new_position)

        if self.steps_completed % self.profile_update_stride == 0:
            self.update_profiles()

        if self._transition_detected_at(new_position):
            self.transition_index = len(self.positions) - 1
            self.termination_reason = "transition"
            return True

        return False

    def result(
        self,
        termination_reason: TerminationReason | None = None,
    ) -> ABFResult:
        """Build an immutable result from the current simulation state."""
        if termination_reason is None:
            termination_reason = (
                "transition"
                if self.transition_index is not None
                else "max_steps"
            )

        self.update_profiles()

        return ABFResult(
            method="abf",
            positions=np.asarray(
                self.positions,
                dtype=float,
            ).reshape(-1, 1),
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
                "dimension": self.dimension,
                "bins": self.bins,
                "value_range": tuple(self.value_range),
                "profile_update_stride": self.profile_update_stride,
            },
        )

    def run(self, max_steps=1_000_000) -> ABFResult:
        """Run at most ``max_steps`` additional integration steps."""
        if (
            isinstance(max_steps, bool)
            or not isinstance(max_steps, (int, np.integer))
        ):
            raise TypeError("max_steps must be an integer.")
        if max_steps <= 0:
            raise ValueError("max_steps must be strictly positive.")

        if self._transition_detected_at(self._last_position()):
            self.transition_index = len(self.positions) - 1
            self.termination_reason = "transition"
            return self.result("transition")

        for _ in range(int(max_steps)):
            if self.step():
                return self.result("transition")

        self.termination_reason = "max_steps"
        return self.result("max_steps")

    def simulate(self, max_iters=1_000_000):
        """Compatibility wrapper around :meth:`run`."""
        result = self.run(max_steps=max_iters)
        return result.physical_time, result.biased_transition_time
