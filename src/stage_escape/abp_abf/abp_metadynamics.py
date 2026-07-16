from __future__ import annotations

import numpy as np

from .potential import Potential
from .results import ABPResult, TerminationReason
from .transition_detector import TransitionDetector


class ABPMetaDynamics:
    """Adaptive biasing potential simulation based on metadynamics."""

    def __init__(
        self,
        deposition_stride,
        transition_detector,
        delta_t,
        dimension=1,
        D=1.0,
        initial_position=None,
        b=None,
        W=0.1,
        sigma=0.001,
        seed=None,
        rng=None,
    ):
        if seed is not None and rng is not None:
            raise ValueError("Provide either seed or rng, not both.")

        if (
            isinstance(deposition_stride, bool)
            or not isinstance(deposition_stride, (int, np.integer))
        ):
            raise TypeError("deposition_stride must be an integer.")
        if deposition_stride <= 0:
            raise ValueError("deposition_stride must be strictly positive.")

        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, (int, np.integer))
        ):
            raise TypeError("dimension must be an integer.")
        if dimension <= 0:
            raise ValueError("dimension must be strictly positive.")

        delta_t = float(delta_t)
        D = float(D)
        W = float(W)
        sigma = float(sigma)

        if not np.isfinite(delta_t) or delta_t <= 0.0:
            raise ValueError("delta_t must be finite and strictly positive.")
        if not np.isfinite(D) or D <= 0.0:
            raise ValueError("D must be finite and strictly positive.")
        if not np.isfinite(W) or W < 0.0:
            raise ValueError("W must be finite and non-negative.")
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("sigma must be finite and strictly positive.")

        if (
            transition_detector is not None
            and not callable(
                getattr(transition_detector, "is_transition", None)
            )
        ):
            raise TypeError(
                "transition_detector must provide is_transition(position)."
            )

        self.deposition_stride = int(deposition_stride)
        self.transition_detector: TransitionDetector | None = (
            transition_detector
        )
        self.delta_t = delta_t
        self.dimension = int(dimension)
        self.D = D
        self.W = W
        self.sigma = sigma
        self.seed = seed
        self.rng = rng if rng is not None else np.random.default_rng(seed)

        if initial_position is None:
            initial_position = np.zeros(self.dimension, dtype=float)
        else:
            initial_position = np.asarray(
                initial_position,
                dtype=float,
            ).reshape(-1)

        if initial_position.size != self.dimension:
            raise ValueError(
                "initial_position must contain exactly "
                f"{self.dimension} components."
            )
        if not np.all(np.isfinite(initial_position)):
            raise ValueError(
                "initial_position must contain only finite values."
            )

        if b is None:
            b = Potential(
                dimension=self.dimension,
                function=lambda x: 0.0,
                first_derivative=lambda x: np.zeros(
                    self.dimension,
                    dtype=float,
                ),
                second_derivative=lambda x: np.zeros(
                    (self.dimension, self.dimension),
                    dtype=float,
                ),
            )

        if not callable(getattr(b, "potential_prime_at", None)):
            raise TypeError(
                "b must provide potential_prime_at(position)."
            )
        if (
            hasattr(b, "dimension")
            and b.dimension != self.dimension
        ):
            raise ValueError(
                "The potential dimension does not match "
                "the simulation dimension."
            )

        self.initial_position = initial_position.copy()
        self.b = b

        self.positions = [self.initial_position.copy()]
        self.centers = []
        self.weights = [1.0]

        self.real_time = 0.0
        self.steps_completed = 0
        self.transition_index = None
        self.termination_reason: TerminationReason | None = None

    def _last_position(self) -> np.ndarray:
        return np.asarray(self.positions[-1], dtype=float)

    def _physical_gradient_at(self, position) -> np.ndarray:
        gradient = np.asarray(
            self.b.potential_prime_at(position),
            dtype=float,
        ).reshape(-1)

        if gradient.size != self.dimension:
            raise ValueError(
                "The physical potential gradient must contain "
                f"{self.dimension} components; received {gradient.size}."
            )
        if not np.all(np.isfinite(gradient)):
            raise FloatingPointError(
                "The physical potential gradient is not finite."
            )

        return gradient

    def bias_potential_at(self, position) -> float:
        from .abp_bias import gaussian_bias_value

        return gaussian_bias_value(
            position=position,
            centers=self.centers,
            height=self.W,
            sigma=self.sigma,
        )

    def bias_potential_prime_at(self, position) -> np.ndarray:
        from .abp_bias import gaussian_bias_gradient

        return gaussian_bias_gradient(
            position=position,
            centers=self.centers,
            height=self.W,
            sigma=self.sigma,
        )

    def _transition_detected_at(self, position) -> bool:
        return (
            self.transition_detector is not None
            and self.transition_detector.is_transition(position)
        )

    def step(self) -> bool:
        """Advance the biased dynamics by one integration step."""
        if self.transition_index is not None:
            return True

        current_position = self._last_position()
        physical_gradient = self._physical_gradient_at(current_position)
        bias_gradient = np.asarray(
            self.bias_potential_prime_at(current_position),
            dtype=float,
        ).reshape(-1)

        if bias_gradient.size != self.dimension:
            raise ValueError(
                "The bias gradient must contain "
                f"{self.dimension} components; received {bias_gradient.size}."
            )
        if not np.all(np.isfinite(bias_gradient)):
            raise FloatingPointError(
                "The bias potential gradient is not finite."
            )

        drift = -(physical_gradient + bias_gradient) * self.delta_t
        noise = (
            np.sqrt(2.0 * self.D * self.delta_t)
            * self.rng.standard_normal(self.dimension)
        )
        new_position = current_position + drift + noise

        if not np.all(np.isfinite(new_position)):
            raise FloatingPointError("The new ABP position is not finite.")

        bias_value = self.bias_potential_at(new_position)
        with np.errstate(over="ignore", invalid="ignore"):
            weight = float(np.exp(bias_value / self.D))

        if not np.isfinite(weight):
            raise FloatingPointError(
                "The ABP reweighting factor is not finite."
            )

        self.positions.append(np.asarray(new_position, dtype=float))
        self.weights.append(weight)
        self.real_time += weight * self.delta_t
        self.steps_completed += 1

        if self._transition_detected_at(new_position):
            self.transition_index = len(self.positions) - 1
            self.termination_reason = "transition"
            return True

        if self.steps_completed % self.deposition_stride == 0:
            self.centers.append(new_position.copy())

        return False

    def result(
        self,
        termination_reason: TerminationReason | None = None,
    ) -> ABPResult:
        """Build an immutable result from the current simulation state."""
        if termination_reason is None:
            termination_reason = (
                "transition"
                if self.transition_index is not None
                else "max_steps"
            )

        positions = np.asarray(
            self.positions,
            dtype=float,
        ).reshape(-1, self.dimension)

        centers = np.asarray(self.centers, dtype=float)
        if centers.size == 0:
            centers = np.empty((0, self.dimension), dtype=float)
        else:
            centers = centers.reshape(-1, self.dimension)

        weights = np.asarray(self.weights, dtype=float)
        if len(weights) != len(positions):
            raise RuntimeError(
                "ABP produced an inconsistent result: "
                f"{len(positions)} positions and {len(weights)} weights."
            )

        return ABPResult(
            method="abp",
            positions=positions,
            delta_t=self.delta_t,
            diffusion=self.D,
            seed=self.seed,
            transition_index=self.transition_index,
            physical_time=float(self.real_time),
            termination_reason=termination_reason,
            weights=weights.copy(),
            centers=centers.copy(),
            bias_height=self.W,
            bias_width=self.sigma,
            metadata={
                "dimension": self.dimension,
                "deposition_stride": self.deposition_stride,
            },
        )

    def run(self, max_steps=1_000_000) -> ABPResult:
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
