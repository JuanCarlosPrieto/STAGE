from __future__ import annotations

import numpy as np


def reconstruct_abf_profiles(
    force_bias,
    value_range,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the estimated mean force and construct the ABF bias.

    ``force_bias`` stores an estimate of the physical potential derivative.
    The free energy is reconstructed by trapezoidal integration. The bias is
    ``max(free_energy) - free_energy`` so that its minimum is zero and the
    associated reweighting factor is not altered by an arbitrary constant.
    """
    force_bias = np.asarray(force_bias, dtype=float)
    if force_bias.ndim != 1:
        raise ValueError("force_bias must be one-dimensional.")
    if len(force_bias) < 2:
        raise ValueError("At least two ABF bins are required.")
    if not np.all(np.isfinite(force_bias)):
        raise ValueError("force_bias must contain only finite values.")

    try:
        lower, upper = value_range
    except (TypeError, ValueError) as error:
        raise ValueError("value_range must contain exactly two values.") from error
    lower = float(lower)
    upper = float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("value_range must contain finite increasing bounds.")

    bin_edges = np.linspace(lower, upper, len(force_bias) + 1)
    bin_width = float(bin_edges[1] - bin_edges[0])

    free_energy = np.zeros_like(force_bias)
    free_energy[1:] = np.cumsum(
        0.5 * (force_bias[:-1] + force_bias[1:]) * bin_width
    )
    free_energy -= np.min(free_energy)

    bias_potential = np.max(free_energy) - free_energy
    return bin_edges, free_energy, bias_potential
