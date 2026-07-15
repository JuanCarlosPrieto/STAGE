# STAGE

Research code for the numerical study of Brownian dynamics, metastable transitions, adaptive biasing methods, and narrow escape problems.

This repository was developed during a research internship and contains implementations of direct Monte Carlo simulations, Adaptive Biasing Potential (ABP), Adaptive Biasing Force (ABF), transition-time estimators, geometric escape detection, statistical analysis tools, and exploratory notebooks.

## Project objectives

The main objective is to study rare transitions and escape events in stochastic systems, with particular emphasis on:

* Brownian motion in one and two dimensions;
* metastable potentials and transition times;
* narrow escape problems with small absorbing windows;
* direct and equivalent narrow escape formulations;
* Adaptive Biasing Potential methods;
* Adaptive Biasing Force methods;
* reconstruction of equilibrium distributions;
* estimation of physical transition and escape times;
* comparison between direct and accelerated sampling methods.

The project also aims to provide reproducible numerical experiments and publication-quality figures for the final internship report.

## Repository structure

```text
STAGE/
├── notebooks/                  # Exploratory analyses and numerical experiments
├── src/
│   └── stage_escape/
│       ├── abp_abf/            # ABP, ABF, potentials and transition detection
│       ├── brownian.py         # Brownian dynamics
│       ├── escape.py           # Escape-window definitions
│       ├── geometry_utilities.py
│       ├── naive_narrow_escape.py
│       ├── equivalent_narrow_escape.py
│       ├── statistics.py
│       ├── surface.py
│       └── visualization.py
├── tests/                      # Automated tests
├── pyproject.toml              # Package configuration and dependencies
├── requirements.txt            # Environment dependencies
└── README.md
```

The codebase is currently being refactored to separate:

1. stochastic simulation;
2. geometry and escape detection;
3. statistical post-processing;
4. visualization;
5. reproducible report-generation workflows.

## Installation

Clone the repository:

```bash
git clone https://github.com/JuanCarlosPrieto/STAGE.git
cd STAGE
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Upgrade `pip` and install the package in editable mode:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

For development and testing:

```bash
python -m pip install -e ".[dev]"
```

When the optional development dependencies are not available through `pyproject.toml`, install the requirements directly:

```bash
python -m pip install -r requirements.txt
```

## Running the tests

Run the complete test suite from the repository root:

```bash
python -m pytest -q
```

For detailed output:

```bash
python -m pytest -vv
```

Run a specific test file:

```bash
python -m pytest tests/test_abf_real_time.py -vv
```

The tests currently focus on:

* random-number reproducibility;
* Brownian trajectory generation;
* dimensional consistency;
* ABP and ABF behavior;
* deterministic results for fixed seeds;
* numerical validity of reconstructed quantities.

## Reproducibility

All stochastic simulations should receive an explicit random seed.

Two simulations using the same:

* initial condition;
* physical parameters;
* numerical parameters;
* potential;
* transition criterion;
* random seed;

should produce identical trajectories and estimators.

Example convention:

```python
seed = 123
```

For Monte Carlo experiments, each independent realization should use a distinct but recorded seed.

A reproducible numerical result should store at least:

```text
seed
diffusion coefficient
time step
number of integration steps
initial position
potential parameters
biasing parameters
transition criterion
number of replicas
Git commit
```

## Main components

### Brownian dynamics

The Brownian motion implementation is based on the overdamped stochastic differential equation

[
dX_t = b(X_t),dt + \sqrt{2D},dW_t,
]

where:

* (X_t) is the particle position;
* (b(X_t)) is the deterministic drift;
* (D) is the diffusion coefficient;
* (W_t) is a standard Wiener process.

The numerical implementation uses an Euler–Maruyama discretization.

### Narrow escape problem

The narrow escape problem studies the time required for a Brownian particle to leave a bounded domain through one or more small absorbing windows.

The repository contains two approaches:

* `NaiveNarrowEscape`: direct geometric simulation inside the physical domain;
* `EquivalentNarrowEscape`: simulation of an equivalent formulation based on an effective potential or transformed problem.

The direct approach includes:

* domain-boundary detection;
* escape-window detection;
* trajectory processing;
* estimation of escape locations and escape times.

### Adaptive Biasing Potential

The ABP implementation progressively constructs a biasing potential in order to flatten energetic barriers and increase the frequency of rare transitions.

The method stores deposited bias contributions and evaluates:

* the physical potential;
* the biasing potential;
* the effective potential;
* statistical weights;
* transition events.

The weighted trajectory can then be used to reconstruct physical distributions and observables.

### Adaptive Biasing Force

The ABF implementation estimates the mean force along a discretized reaction coordinate.

For a one-dimensional reaction coordinate, the state space is divided into bins. The algorithm estimates the mean physical force in every visited bin and constructs an adaptive bias from this estimator.

The ABF implementation is currently restricted to one-dimensional potentials.

### Transition detection

Transition events are detected using a collective variable and a threshold criterion.

A typical detector evaluates:

[
\xi(X_t) \geq \xi_{\mathrm{threshold}},
]

where (\xi) is a collective variable.

The transition detector returns the index of the first trajectory point satisfying the transition condition.

### Statistical analysis

The statistical utilities are used to estimate:

* empirical probability densities;
* weighted histograms;
* mean transition times;
* mean escape times;
* exponential-law parameters;
* uncertainty indicators;
* comparisons between numerical and theoretical distributions.

## Notebooks

The `notebooks/` directory contains exploratory numerical experiments.

Typical notebook topics include:

* Brownian motion validation;
* direct narrow escape simulations;
* equivalent narrow escape formulations;
* ABP behavior in one and two dimensions;
* reconstruction of equilibrium densities;
* exponential transition-time distributions;
* parameter sensitivity;
* comparison of direct and biased methods.

The notebooks are intended for exploration and interpretation. Long simulations should progressively be moved to standalone scripts, while notebooks should load previously generated data.

Recommended workflow:

```text
simulation script
    ↓
raw numerical results
    ↓
processed data
    ↓
analysis notebook
    ↓
report figure
```

## Numerical validation

Before using a result in a report, the following checks should be performed.

### Brownian motion

Without drift:

[
\mathbb{E}[X_t] \approx X_0,
]

and

[
\operatorname{Var}(X_t) \approx 2Dt.
]

### Equilibrium distribution

For a system with potential (V), the theoretical equilibrium density is

[
\rho(x)
=======

\frac{1}{Z}
\exp\left(-\frac{V(x)}{D}\right),
]

where

[
Z
=

\int
\exp\left(-\frac{V(x)}{D}\right),dx.
]

Weighted ABP or ABF results should be compared against this theoretical density.

### Time-step convergence

Results should be compared for several values of the integration time step:

```text
Δt
Δt / 2
Δt / 4
```

Relevant observables should stabilize as the time step decreases.

### Monte Carlo uncertainty

Transition and escape times should be estimated using several independent realizations.

Reported results should include an uncertainty measure such as:

* standard error;
* confidence interval;
* bootstrap confidence interval.

The standard deviation of individual escape times should not be confused with the uncertainty of the estimated mean.

## Report figures

Final figures should be generated independently from the simulation code.

Recommended report figures include:

1. geometry of the narrow escape problem;
2. representative Brownian trajectory;
3. validation of Brownian variance;
4. physical and biasing potentials;
5. weighted empirical density versus theoretical density;
6. transition-time distribution;
7. survival function;
8. mean escape time as a function of the escape-window size;
9. comparison of direct and adaptive methods;
10. computational efficiency and estimator variance.

Figures should preferably be saved in both formats:

```text
PDF: vector format for the written report
PNG: high-resolution raster format for presentations
```

Recommended export parameters:

```python
fig.savefig(
    "figure.pdf",
    bbox_inches="tight",
)

fig.savefig(
    "figure.png",
    dpi=300,
    bbox_inches="tight",
)
```

Plotting functions should return `fig` and `ax` and should avoid calling `plt.show()` internally.

## Author

Juan Carlos Prieto Calderón
