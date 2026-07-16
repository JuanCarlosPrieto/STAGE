from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .abp_abf.results import (
    ABFResult,
    ABPResult,
    SimulationResult,
)

SCHEMA_VERSION = 1


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
        "schema_version": SCHEMA_VERSION,
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


def load_result(input_stem):
    input_stem = Path(input_stem)

    metadata = json.loads(
        input_stem
        .with_suffix(".json")
        .read_text(encoding="utf-8")
    )

    with np.load(
        input_stem.with_suffix(".npz"),
        allow_pickle=False,
    ) as data:
        arrays = {
            name: data[name].copy()
            for name in data.files
        }

    common = {
        "method": metadata["method"],
        "positions": arrays["positions"],
        "delta_t": metadata["delta_t"],
        "diffusion": metadata["diffusion"],
        "seed": metadata["seed"],
        "transition_index": metadata[
            "transition_index"
        ],
        "physical_time": metadata[
            "physical_time"
        ],
        "termination_reason": metadata[
            "termination_reason"
        ],
        "metadata": metadata.get(
            "parameters",
            {},
        ),
    }

    if metadata["method"] == "abp":
        return ABPResult(
            **common,
            weights=arrays["weights"],
            centers=arrays["centers"],
            bias_height=metadata["bias_height"],
            bias_width=metadata["bias_width"],
        )

    if metadata["method"] == "abf":
        return ABFResult(
            **common,
            force_bias=arrays["force_bias"],
            visit_counts=arrays["visit_counts"],
            bin_edges=arrays["bin_edges"],
            free_energy=arrays["free_energy"],
            bias_potential=arrays[
                "bias_potential"
            ],
        )

    raise ValueError(
        f"Unknown method: {metadata['method']}"
    )