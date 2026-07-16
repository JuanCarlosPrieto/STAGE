import numpy as np


def reconstruct_abf_profiles(
    force_bias,
    value_range,
):
    force_bias = np.asarray(
        force_bias,
        dtype=float,
    )

    if force_bias.ndim != 1:
        raise ValueError(
            "force_bias must be one-dimensional."
        )

    if len(force_bias) < 2:
        raise ValueError(
            "At least two ABF bins are required."
        )

    lower, upper = value_range

    if upper <= lower:
        raise ValueError(
            "The upper bound must be greater "
            "than the lower bound."
        )

    bin_edges = np.linspace(
        lower,
        upper,
        len(force_bias) + 1,
    )

    bin_width = bin_edges[1] - bin_edges[0]

    free_energy = np.zeros_like(
        force_bias,
        dtype=float,
    )

    free_energy[1:] = np.cumsum(
        0.5
        * (
            force_bias[:-1]
            + force_bias[1:]
        )
        * bin_width
    )

    free_energy -= np.min(free_energy)

    bias_potential = -free_energy
    bias_potential -= np.min(bias_potential)

    return (
        bin_edges,
        free_energy,
        bias_potential,
    )