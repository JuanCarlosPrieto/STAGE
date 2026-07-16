from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from stage_escape.abp_abf import (
    ABFRealTime,
    ABPMetaDynamics,
    Potential,
    TransitionDetector,
)
from stage_escape.result_io import save_result


def harmonic_potential() -> Potential:
    return Potential(
        dimension=1,
        function=lambda x: 0.5 * x[0] ** 2,
        first_derivative=lambda x: np.array([x[0]]),
        second_derivative=lambda _x: np.array([[1.0]]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic ABP/ABF smoke cases.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, default=Path("artifacts/smoke"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    detector = TransitionDetector(lambda x: x[0], threshold=np.inf)
    potential = harmonic_potential()
    abp = ABPMetaDynamics(
        deposition_stride=20,
        transition_detector=detector,
        delta_t=1e-3,
        D=0.2,
        b=potential,
        W=0.02,
        sigma=0.25,
        seed=args.seed,
    ).run(args.steps)
    abf = ABFRealTime(
        transition_detector=detector,
        delta_t=1e-3,
        D=0.2,
        b=potential,
        bins=80,
        value_range=(-5.0, 5.0),
        seed=args.seed,
    ).run(args.steps)

    args.output.mkdir(parents=True, exist_ok=True)
    save_result(abp, args.output / "abp")
    save_result(abf, args.output / "abf")
    print(f"ABP: {abp.n_steps} steps, {len(abp.centers)} deposited centers")
    print(f"ABF: {abf.n_steps} steps, {int(abf.visit_counts.sum())} visits")


if __name__ == "__main__":
    main()
