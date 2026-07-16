from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def find_intersection(phi, a, b) -> tuple[np.ndarray, float]:
    """Return a bracketed boundary intersection on the segment ``[a, b]``."""
    if not callable(phi):
        raise TypeError("phi must be callable.")
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be one-dimensional with the same shape.")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("a and b must contain finite values.")

    phi_a = float(phi(a))
    phi_b = float(phi(b))
    if not np.isfinite(phi_a) or not np.isfinite(phi_b):
        raise ValueError("phi must return finite scalar values.")
    if phi_a == 0.0:
        return a.copy(), 0.0
    if phi_b == 0.0:
        return b.copy(), 1.0
    if phi_a * phi_b > 0.0:
        raise ValueError(
            "The segment endpoints do not bracket a boundary intersection."
        )

    def segment_function(t: float) -> float:
        return float(phi(a + t * (b - a)))

    theta = float(brentq(segment_function, 0.0, 1.0))
    return a + theta * (b - a), theta
