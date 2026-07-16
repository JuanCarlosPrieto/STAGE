import numpy as np
from scipy.optimize import brentq


def find_intersection(phi, a, b):
    """Return the intersection point and its relative position on [a, b]."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape.")

    phi_a = float(phi(a))
    phi_b = float(phi(b))

    if phi_a == 0.0:
        return a.copy(), 0.0

    if phi_b == 0.0:
        return b.copy(), 1.0

    if phi_a * phi_b > 0:
        raise ValueError(
            "The segment endpoints do not bracket a boundary intersection."
        )

    def segment_function(t):
        point = a + t * (b - a)
        return float(phi(point))

    theta = brentq(segment_function, 0.0, 1.0)
    intersection_point = a + theta * (b - a)

    return intersection_point, theta