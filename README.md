# STAGE — refactored simulation core

This tree is a reviewed refactor of commit `6f50424` of the STAGE repository. It keeps the research scope—Brownian dynamics, narrow escape, ABP and ABF—but gives the simulation core explicit contracts, exact step semantics, reproducible random-number handling, immutable result objects and tested persistence.

## Architecture

```text
src/stage_escape/
├── brownian.py                    # Euler–Maruyama proposals and one-step integration
├── surface.py                     # implicit domains and earliest boundary crossing
├── escape.py                      # absorbing-window predicates
├── narrow_escape_result.py        # shared immutable result contract
├── _narrow_escape_base.py         # shared narrow-escape state and validation
├── naive_narrow_escape.py         # direct geometric narrow escape
├── equivalent_narrow_escape.py    # equivalent/end-point escape formulation
├── result_io.py                   # versioned JSON + NPZ persistence
├── statistics.py                  # estimators and opt-in plotting helpers
├── visualization.py               # plotting without implicit plt.show()
└── abp_abf/
    ├── _adaptive_base.py          # shared ABP/ABF RNG, clock and step-cap logic
    ├── potential.py               # scalar potential and derivatives
    ├── transition_detector.py     # stateless threshold detector
    ├── abp_metadynamics.py        # Gaussian metadynamics/ABP
    ├── abf_real_time.py           # one-dimensional online ABF
    ├── abp_bias.py                # vectorized Gaussian bias primitives
    ├── abf_profiles.py            # force-profile integration
    ├── distribution_analysis.py   # weighted densities and theoretical laws
    └── results.py                 # immutable ABP/ABF result contracts
```

Simulation, post-processing, plotting and persistence are separate. Simulation methods return result objects; plotting functions return Matplotlib `(figure, axes)` objects and only display when `show=True`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Notebook dependencies are optional:

```bash
python -m pip install -e ".[notebook]"
```

## Verification

```bash
python -m pytest -q
python -m pytest -q --cov=stage_escape --cov-report=term-missing
ruff check .
```

The supplied suite verifies Brownian reproducibility, ABP/ABF reproducibility, exact total-step caps, online ABF visit counting, transition detection, Gaussian-potential composition, geometric boundary intersections, direct and equivalent narrow escape, result immutability, versioned persistence, numerical distributions and non-blocking plotting.

## Core usage

### Brownian motion

```python
from stage_escape import BrownianMotion

motion = BrownianMotion(
    deposition_stride=100,
    delta_t=1e-3,
    dimension=2,
    D=0.2,
    initial_position=[0.0, 0.0],
    seed=123,
)

motion.step()       # exactly one Euler–Maruyama step
path = motion.simulate()  # historical block API retained
```

### ABP

```python
import numpy as np
from stage_escape.abp_abf import ABPMetaDynamics, Potential, TransitionDetector

potential = Potential(
    dimension=1,
    function=lambda x: 0.25 * (x[0] ** 2 - 1.0) ** 2,
    first_derivative=lambda x: np.array([x[0] * (x[0] ** 2 - 1.0)]),
)
detector = TransitionDetector(lambda x: x[0], threshold=0.8)

simulation = ABPMetaDynamics(
    deposition_stride=100,
    transition_detector=detector,
    delta_t=1e-3,
    D=0.1,
    initial_position=[-1.0],
    b=potential,
    W=0.01,
    sigma=0.15,
    seed=123,
)
result = simulation.run(max_steps=100_000)
```

`max_steps` is a total trajectory cap. A later call with a larger value continues the same trajectory. `simulate(max_iters=...)` remains as a compatibility alias.

### Direct narrow escape

```python
from stage_escape import BrownianMotion, Escape, NaiveNarrowEscape, Surface

surface = Surface("unit disk", [lambda p: p[0] ** 2 + p[1] ** 2 - 1.0])
escape = Escape([
    lambda p: p[0] > 0.99,
    lambda p: abs(p[1]) < 0.1,
])
motion = BrownianMotion(100, 1e-4, dimension=2, D=1.0, seed=123)
result = NaiveNarrowEscape(motion, surface, [escape]).run(max_steps=1_000_000)
```

The direct solver determines the earliest admissible boundary intersection along each proposal and computes the escape time with sub-step interpolation.

## Persistence

```python
from stage_escape.result_io import load_result, save_result

save_result(result, "artifacts/reference")
loaded = load_result("artifacts/reference")
```

Metadata is stored in JSON and numerical arrays in compressed NPZ. Writes are atomic per file, schema versions are validated, and schemas 1–3 are readable.

## Reproducible scripts

```bash
python scripts/run_abp_abf_smoke.py --steps 200 --seed 123
python scripts/run_reference_1d.py --steps 100000 --seed 2026
python scripts/make_report_figures.py artifacts/reference_1d
```

Long simulations should run in scripts and save results. Notebooks should load those results for interpretation and figure composition.

## Compatibility notes

- `ABPMetaDynamics` is the canonical class; `ABPMetadynamics` is retained as an alias.
- `simulate(max_iters=...)` remains available for ABP and ABF.
- `run_simulation_straight_exit()` and `run_simulation()` remain deprecated wrappers around the unified `NarrowEscapeResult` API.
- Plotting no longer calls `plt.show()` unless explicitly requested.
