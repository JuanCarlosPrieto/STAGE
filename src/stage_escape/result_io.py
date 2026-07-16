from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .abp_abf.results import ABFResult, ABPResult, SimulationResult

SCHEMA_VERSION = 3
SUPPORTED_SCHEMA_VERSIONS = {1, 2, 3}


def _stem(path_like) -> Path:
    path = Path(path_like)
    return path.with_suffix("") if path.suffix in {".json", ".npz"} else path


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable.")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".npz",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def save_result(result: SimulationResult, output_stem) -> tuple[Path, Path]:
    """Persist an ABP or ABF result as JSON metadata plus compressed arrays."""
    if not isinstance(result, (ABPResult, ABFResult)):
        raise TypeError("save_result only supports ABPResult and ABFResult.")

    output_stem = _stem(output_stem)
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
    else:
        arrays.update(
            force_bias=result.force_bias,
            visit_counts=result.visit_counts,
            bin_edges=result.bin_edges,
            free_energy=result.free_energy,
            bias_potential=result.bias_potential,
        )

    # Write the numerical payload first. A reader will never observe metadata
    # pointing to an incomplete NPZ file.
    _atomic_write_npz(npz_path, arrays)
    try:
        _atomic_write_text(
            json_path,
            json.dumps(_jsonable(metadata), indent=2, sort_keys=True),
        )
    except Exception:
        npz_path.unlink(missing_ok=True)
        raise
    return json_path, npz_path


def _required_array(arrays: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    try:
        return arrays[name]
    except KeyError as error:
        raise ValueError(f"Result archive is missing array {name!r}.") from error


def load_result(input_stem) -> ABPResult | ABFResult:
    """Load and validate an ABP/ABF result from supported schema versions."""
    input_stem = _stem(input_stem)
    json_path = input_stem.with_suffix(".json")
    npz_path = input_stem.with_suffix(".npz")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Result metadata must be a JSON object.")

    schema_version = metadata.get("schema_version", 1)
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported result schema version {schema_version!r}; "
            f"supported versions are {sorted(SUPPORTED_SCHEMA_VERSIONS)}."
        )

    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}

    try:
        method = metadata["method"]
        delta_t = metadata["delta_t"]
        diffusion = metadata["diffusion"]
        physical_time = metadata["physical_time"]
    except KeyError as error:
        raise ValueError(f"Result metadata is missing key {error.args[0]!r}.") from error

    transition_index = metadata.get("transition_index")
    termination_reason = metadata.get("termination_reason")
    if termination_reason is None:
        termination_reason = (
            "transition" if transition_index is not None else "max_steps"
        )

    common = {
        "method": method,
        "positions": _required_array(arrays, "positions"),
        "delta_t": delta_t,
        "diffusion": diffusion,
        "seed": metadata.get("seed"),
        "transition_index": transition_index,
        "physical_time": physical_time,
        "termination_reason": termination_reason,
        "metadata": metadata.get("parameters", {}),
    }
    if method == "abp":
        return ABPResult(
            **common,
            weights=_required_array(arrays, "weights"),
            centers=_required_array(arrays, "centers"),
            bias_height=metadata["bias_height"],
            bias_width=metadata["bias_width"],
        )
    if method == "abf":
        return ABFResult(
            **common,
            force_bias=_required_array(arrays, "force_bias"),
            visit_counts=_required_array(arrays, "visit_counts"),
            bin_edges=_required_array(arrays, "bin_edges"),
            free_energy=_required_array(arrays, "free_energy"),
            bias_potential=_required_array(arrays, "bias_potential"),
        )
    raise ValueError(f"Unknown simulation method: {method!r}.")
