from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .abp_abf.results import ABFResult, ABPResult, SimulationResult


SCHEMA_VERSION = 2


def _stem(path_like) -> Path:
    path = Path(path_like)
    if path.suffix in {".json", ".npz"}:
        return path.with_suffix("")
    return path


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_result(result: SimulationResult, output_stem) -> tuple[Path, Path]:
    """Save one simulation result as JSON metadata and compressed arrays."""
    output_stem = _stem(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    npz_path = output_stem.with_suffix(".npz")
    json_path = output_stem.with_suffix(".json")

    arrays: dict[str, np.ndarray] = {"positions": result.positions}
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method": result.method,
        "delta_t": result.delta_t,
        "diffusion": result.diffusion,
        "seed": result.seed,
        "transition_index": result.transition_index,
        "physical_time": result.physical_time,
        "termination_reason": result.termination_reason,
        "parameters": result.metadata,
    }

    if isinstance(result, ABPResult):
        arrays.update(weights=result.weights, centers=result.centers)
        metadata.update(
            bias_height=result.bias_height,
            bias_width=result.bias_width,
        )
    elif isinstance(result, ABFResult):
        arrays.update(
            force_bias=result.force_bias,
            visit_counts=result.visit_counts,
            bin_edges=result.bin_edges,
            free_energy=result.free_energy,
            bias_potential=result.bias_potential,
        )
    else:
        raise TypeError("save_result only supports ABPResult and ABFResult.")

    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(
        json.dumps(_jsonable(metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return json_path, npz_path


def load_result(input_stem) -> ABPResult | ABFResult:
    """Load a result, including backward-compatible schema-v1 metadata."""
    input_stem = _stem(input_stem)
    metadata = json.loads(
        input_stem.with_suffix(".json").read_text(encoding="utf-8")
    )
    with np.load(input_stem.with_suffix(".npz"), allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}

    transition_index = metadata.get("transition_index")
    termination_reason = metadata.get("termination_reason")
    if termination_reason is None:
        termination_reason = (
            "transition" if transition_index is not None else "max_steps"
        )

    common = {
        "method": metadata["method"],
        "positions": arrays["positions"],
        "delta_t": metadata["delta_t"],
        "diffusion": metadata["diffusion"],
        "seed": metadata.get("seed"),
        "transition_index": transition_index,
        "physical_time": metadata["physical_time"],
        "termination_reason": termination_reason,
        "metadata": metadata.get("parameters", {}),
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
            bias_potential=arrays["bias_potential"],
        )
    raise ValueError(f"Unknown simulation method: {metadata['method']!r}.")
