from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ._validation import (
    as_position,
    as_scalar,
    validate_positive_float,
    validate_positive_int,
)


ScalarFunction = Callable[[np.ndarray], float]
GradientFunction = Callable[[np.ndarray], np.ndarray]
HessianFunction = Callable[[np.ndarray], np.ndarray]


class Potential:
    """Scalar potential and its first two derivatives.

    Analytic derivatives are optional. Missing derivatives are evaluated with
    centered finite differences, which are more accurate and less biased than
    the forward differences previously used by the project.
    """

    def __init__(
        self,
        dimension: int,
        function: ScalarFunction,
        first_derivative: GradientFunction | None = None,
        second_derivative: HessianFunction | None = None,
    ) -> None:
        self.dimension = validate_positive_int("dimension", dimension)
        if not callable(function):
            raise TypeError("function must be callable.")
        if first_derivative is not None and not callable(first_derivative):
            raise TypeError("first_derivative must be callable or None.")
        if second_derivative is not None and not callable(second_derivative):
            raise TypeError("second_derivative must be callable or None.")

        # Preserve the historical public attribute names for notebook
        # compatibility while giving them a rigorously validated interface.
        self.potential = function
        self.potential_prime = first_derivative
        self.potential_biprime = second_derivative

    @classmethod
    def zero(cls, dimension: int) -> "Potential":
        dimension = validate_positive_int("dimension", dimension)
        return cls(
            dimension=dimension,
            function=lambda x: 0.0,
            first_derivative=lambda x: np.zeros(dimension, dtype=float),
            second_derivative=lambda x: np.zeros(
                (dimension, dimension), dtype=float
            ),
        )

    def _point(self, point) -> np.ndarray:
        return as_position(point, self.dimension, name="point")

    def potential_at(self, point) -> float:
        point = self._point(point)
        return as_scalar(self.potential(point), name="potential value")

    def potential_prime_at(
        self,
        point,
        epsilon: float = 1e-5,
    ) -> np.ndarray:
        point = self._point(point)
        if self.potential_prime is not None:
            gradient = np.asarray(
                self.potential_prime(point), dtype=float
            ).reshape(-1)
            if gradient.size != self.dimension:
                raise ValueError(
                    "first_derivative must return exactly "
                    f"{self.dimension} components."
                )
            if not np.all(np.isfinite(gradient)):
                raise ValueError("first_derivative returned non-finite values.")
            return gradient.copy()

        epsilon = validate_positive_float("epsilon", epsilon)
        gradient = np.empty(self.dimension, dtype=float)
        for axis in range(self.dimension):
            offset = np.zeros(self.dimension, dtype=float)
            offset[axis] = epsilon
            gradient[axis] = (
                self.potential_at(point + offset)
                - self.potential_at(point - offset)
            ) / (2.0 * epsilon)
        return gradient

    def potential_biprime_at(
        self,
        point,
        epsilon: float = 1e-4,
    ) -> np.ndarray:
        point = self._point(point)
        if self.potential_biprime is not None:
            hessian = np.asarray(
                self.potential_biprime(point), dtype=float
            )
            expected_shape = (self.dimension, self.dimension)
            if hessian.shape != expected_shape:
                raise ValueError(
                    "second_derivative must return shape "
                    f"{expected_shape}; received {hessian.shape}."
                )
            if not np.all(np.isfinite(hessian)):
                raise ValueError("second_derivative returned non-finite values.")
            return hessian.copy()

        epsilon = validate_positive_float("epsilon", epsilon)
        hessian = np.empty((self.dimension, self.dimension), dtype=float)
        f0 = self.potential_at(point)

        for i in range(self.dimension):
            ei = np.zeros(self.dimension, dtype=float)
            ei[i] = epsilon
            hessian[i, i] = (
                self.potential_at(point + ei)
                - 2.0 * f0
                + self.potential_at(point - ei)
            ) / epsilon**2

            for j in range(i + 1, self.dimension):
                ej = np.zeros(self.dimension, dtype=float)
                ej[j] = epsilon
                mixed = (
                    self.potential_at(point + ei + ej)
                    - self.potential_at(point + ei - ej)
                    - self.potential_at(point - ei + ej)
                    + self.potential_at(point - ei - ej)
                ) / (4.0 * epsilon**2)
                hessian[i, j] = mixed
                hessian[j, i] = mixed

        return hessian

    def with_gaussian(
        self,
        center,
        height: float,
        width: float,
    ) -> "Potential":
        """Return a new potential with an added isotropic Gaussian."""
        center = as_position(center, self.dimension, name="center")
        height = float(height)
        width = validate_positive_float("width", width)
        if not np.isfinite(height):
            raise ValueError("height must be finite.")

        def gaussian(x: np.ndarray) -> float:
            difference = x - center
            exponent = -np.dot(difference, difference) / (2.0 * width**2)
            return float(height * np.exp(exponent))

        def gaussian_gradient(x: np.ndarray) -> np.ndarray:
            difference = x - center
            return -gaussian(x) * difference / width**2

        def gaussian_hessian(x: np.ndarray) -> np.ndarray:
            difference = x - center
            return gaussian(x) * (
                np.outer(difference, difference) / width**4
                - np.eye(self.dimension) / width**2
            )

        return Potential(
            dimension=self.dimension,
            function=lambda x: self.potential_at(x) + gaussian(x),
            first_derivative=lambda x: (
                self.potential_prime_at(x) + gaussian_gradient(x)
            ),
            second_derivative=lambda x: (
                self.potential_biprime_at(x) + gaussian_hessian(x)
            ),
        )

    def add_gaussian(self, center, height: float, width: float) -> "Potential":
        """Mutate this potential for backward compatibility.

        New code should prefer :meth:`with_gaussian`, which avoids hidden
        mutation and is easier to reason about in reproducible experiments.
        """
        combined = self.with_gaussian(center, height, width)
        self.potential = combined.potential
        self.potential_prime = combined.potential_prime
        self.potential_biprime = combined.potential_biprime
        return self
