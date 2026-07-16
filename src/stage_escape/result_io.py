from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .abp_abf.results import (
    ABFResult,
    ABPResult,
    SimulationResult,
)


def save_result(
    result: SimulationResult,
    output_stem,
):
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    arrays = {
        "positions": result.positions,
    }

    metadata = {
        "method": result.method,
        "delta_t": result.delta_t,
        "diffusion": result.diffusion,
        "seed": result.seed,
        "transition_index": (
            result.transition_index
        ),
        "physical_time": result.physical_time,
        "simulated_time": result.simulated_time,
        "transition_detected": (
            result.transition_detected
        ),
        "parameters": result.metadata,
    }

    if isinstance(result, ABPResult):
        arrays.update(
            {
                "weights": result.weights,
                "centers": result.centers,
            }
        )

        metadata.update(
            {
                "bias_height": result.bias_height,
                "bias_width": result.bias_width,
            }
        )

    elif isinstance(result, ABFResult):
        arrays.update(
            {
                "force_bias": result.force_bias,
                "visit_counts": result.visit_counts,
                "bin_edges": result.bin_edges,
                "free_energy": result.free_energy,
                "bias_potential": (
                    result.bias_potential
                ),
            }
        )

    np.savez_compressed(
        output_stem.with_suffix(".npz"),
        **arrays,
    )

    output_stem.with_suffix(".json").write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )