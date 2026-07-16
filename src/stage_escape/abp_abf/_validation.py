from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def validate_positive_float(
    name: str,
    value: Any,
    *,
    allow_zero: bool = False,
) -> float:
    """Return a validated finite floating-point parameter."""
    result = float(value)
    lower_bound_satisfied = result >= 0.0 if allow_zero else result > 0.0
    if not np.isfinite(result) or not lower_bound_satisfied:
        qualifier = "non-negative" if allow_zero else "strictly positive"
        raise ValueError(f"{name} must be finite and {qualifier}.")
    return result


def validate_positive_int(
    name: str,
    value: Any,
    *,
    minimum: int = 1,
) -> int:
    """Return a validated integer parameter."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be greater than or equal to {minimum}.")
    return result


def as_position(
    position: Any,
    dimension: int,
    *,
    name: str = "position",
) -> np.ndarray:
    """Normalize one position to a finite array of shape ``(dimension,)``."""
    array = np.asarray(position, dtype=float)
    if array.ndim == 0 and dimension == 1:
        array = array.reshape(1)
    else:
        array = array.reshape(-1)

    if array.size != dimension:
        raise ValueError(
            f"{name} must contain exactly {dimension} components; "
            f"received {array.size}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array.copy()


def as_scalar(value: Any, *, name: str) -> float:
    """Normalize a scalar-like value and reject vector-valued outputs."""
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        raise ValueError(f"{name} must contain exactly one scalar value.")
    result = float(array.reshape(-1)[0])
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def prepare_rng(
    *,
    seed: int | None,
    rng: np.random.Generator | None,
) -> tuple[np.random.Generator, bool]:
    """Create or validate a NumPy random generator.

    Returns the generator and a flag indicating whether it was supplied by
    the caller. Internally created generators can be reset reproducibly from
    ``seed``; external generators retain ownership of their state.
    """
    if seed is not None and rng is not None:
        raise ValueError("Provide either seed or rng, not both.")
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an instance of numpy.random.Generator.")
    return (
        rng if rng is not None else np.random.default_rng(seed),
        rng is not None,
    )


def validate_transition_detector(detector: Any) -> None:
    """Validate the structural protocol used by the simulators."""
    if detector is None:
        return
    method = getattr(detector, "is_transition", None)
    if not callable(method):
        raise TypeError(
            "transition_detector must be None or provide "
            "is_transition(position)."
        )


def validate_callable(name: str, value: Any) -> Callable[..., Any]:
    if not callable(value):
        raise TypeError(f"{name} must be callable.")
    return value
