from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from stage_escape.abp_abf import ABPMetaDynamics, Potential, TransitionDetector
from stage_escape.result_io import save_result


def double_well() -> Potential:
    return Potential(
        dimension=1,
        function=lambda x: 0.25 * (x[0] ** 2 - 1.0) ** 2,
        first_derivative=lambda x: np.array([x[0] * (x[0] ** 2 - 1.0)]),
        second_derivative=lambda x: np.array([[3.0 * x[0] ** 2 - 1.0]]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reproducible 1D ABP reference case.")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("artifacts/reference_1d"))
    args = parser.parse_args()
    detector = TransitionDetector(lambda x: x[0], threshold=0.8)
    result = ABPMetaDynamics(
        deposition_stride=100,
        transition_detector=detector,
        delta_t=1e-3,
        D=0.1,
        initial_position=np.array([-1.0]),
        b=double_well(),
        W=0.01,
        sigma=0.15,
        seed=args.seed,
    ).run(args.steps)
    save_result(result, args.output)
    print(
        f"termination={result.termination_reason}, steps={result.n_steps}, "
        f"physical_time={result.physical_time:.6g}"
    )


if __name__ == "__main__":
    main()
