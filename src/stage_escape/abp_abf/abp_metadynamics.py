from __future__ import annotations

import numpy as np

from ._validation import (
    as_position,
    prepare_rng,
    validate_positive_float,
    validate_positive_int,
    validate_transition_detector,
)
from .abp_bias import gaussian_bias_gradient, gaussian_bias_value
from .potential import Potential
from .results import ABPResult, TerminationReason
from .transition_detector import TransitionDetector


class ABPMetaDynamics:
    """Adaptive biasing potential simulation using Gaussian metadynamics.

    The class owns only stochastic integration and bias deposition. Transition
    logic is injected through a stateless :class:`TransitionDetector`; plotting
    and distribution analysis are deliberately kept outside the simulator.
    """

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
        cutoff=5.0,
    ) -> None:
        self.deposition_stride = validate_positive_int(
            "deposition_stride", deposition_stride
        )
        self.delta_t = validate_positive_float("delta_t", delta_t)
        self.dimension = validate_positive_int("dimension", dimension)
        self.D = validate_positive_float("D", D)
        self.W = validate_positive_float("W", W, allow_zero=True)
        self.sigma = validate_positive_float("sigma", sigma)
        self.cutoff = (
            None
            if cutoff is None
            else validate_positive_float("cutoff", cutoff)
        )
        validate_transition_detector(transition_detector)
        self.transition_detector: TransitionDetector | None = transition_detector

        self.seed = seed
        self.rng, self._external_rng = prepare_rng(seed=seed, rng=rng)
        self.initial_position = as_position(
            np.zeros(self.dimension) if initial_position is None else initial_position,
            self.dimension,
            name="initial_position",
        )

        self.b = Potential.zero(self.dimension) if b is None else b
        if not callable(getattr(self.b, "potential_prime_at", None)):
            raise TypeError("b must provide potential_prime_at(position).")
        if getattr(self.b, "dimension", self.dimension) != self.dimension:
            raise ValueError("The potential dimension does not match dimension.")

        self.reset(reset_rng=False)

    def reset(self, *, reset_rng: bool = True) -> None:
        """Reset trajectory, bias and clocks.

        Internally seeded generators are recreated by default. Externally
        supplied generators are never rewound because their state is owned by
        the caller.
        """
        if reset_rng and not self._external_rng:
            self.rng = np.random.default_rng(self.seed)
        self.positions = [self.initial_position.copy()]
        self.centers: list[np.ndarray] = []
        self.weights = [1.0]
        self.real_time = 0.0
        self.steps_completed = 0
        self.transition_index: int | None = None
        self.termination_reason: TerminationReason | None = None

    def _last_position(self) -> np.ndarray:
        return np.asarray(self.positions[-1], dtype=float)

    def _physical_gradient_at(self, position) -> np.ndarray:
        gradient = np.asarray(
            self.b.potential_prime_at(position), dtype=float
        ).reshape(-1)
        if gradient.size != self.dimension:
            raise ValueError(
                "The physical gradient must contain "
                f"{self.dimension} components."
            )
        if not np.all(np.isfinite(gradient)):
            raise FloatingPointError("The physical gradient is not finite.")
        return gradient

    def bias_potential_at(self, position) -> float:
        return gaussian_bias_value(
            position=position,
            centers=self.centers,
            height=self.W,
            sigma=self.sigma,
            cutoff=self.cutoff,
        )

    def bias_potential_prime_at(self, position) -> np.ndarray:
        return gaussian_bias_gradient(
            position=position,
            centers=self.centers,
            height=self.W,
            sigma=self.sigma,
            cutoff=self.cutoff,
        )

    def effective_potential_at(self, position) -> float:
        return self.b.potential_at(position) + self.bias_potential_at(position)

    def _transition_detected_at(self, position) -> bool:
        return (
            self.transition_detector is not None
            and self.transition_detector.is_transition(position)
        )

    def step(self) -> bool:
        """Advance the biased dynamics by exactly one integration step."""
        if self.transition_index is not None:
            return True

        current_position = self._last_position()
        physical_gradient = self._physical_gradient_at(current_position)
        bias_gradient = self.bias_potential_prime_at(current_position)
        if bias_gradient.shape != (self.dimension,):
            raise ValueError("The bias gradient has an invalid shape.")

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
        if not np.isfinite(weight) or weight <= 0.0:
            raise FloatingPointError("The ABP reweighting factor is invalid.")

        self.positions.append(new_position.copy())
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
        if termination_reason is None:
            termination_reason = (
                "transition" if self.transition_index is not None else "max_steps"
            )

        centers = np.asarray(self.centers, dtype=float)
        if centers.size == 0:
            centers = np.empty((0, self.dimension), dtype=float)
        else:
            centers = centers.reshape(-1, self.dimension)

        return ABPResult(
            method="abp",
            positions=np.asarray(self.positions, dtype=float).reshape(
                -1, self.dimension
            ),
            delta_t=self.delta_t,
            diffusion=self.D,
            seed=self.seed,
            transition_index=self.transition_index,
            physical_time=float(self.real_time),
            termination_reason=termination_reason,
            weights=np.asarray(self.weights, dtype=float),
            centers=centers,
            bias_height=self.W,
            bias_width=self.sigma,
            metadata={
                "dimension": self.dimension,
                "deposition_stride": self.deposition_stride,
                "cutoff": self.cutoff,
            },
        )

    def run(self, max_steps=1_000_000, *, reset: bool = False) -> ABPResult:
        max_steps = validate_positive_int("max_steps", max_steps)
        if reset:
            self.reset()

        if self._transition_detected_at(self._last_position()):
            self.transition_index = len(self.positions) - 1
            self.termination_reason = "transition"
            return self.result("transition")

        for _ in range(max_steps):
            if self.step():
                return self.result("transition")

        self.termination_reason = "max_steps"
        return self.result("max_steps")

