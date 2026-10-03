# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# bayes-cep
Non-parametric Bayesian inference of myocardial fiber orientations from electrical activation-time
data (Python ≥3.14). Assembles a fiber-orientation posterior from two sibling packages consumed as
local editable path deps: `ls_bayesian` (`../ls-bayesian`, generic Bayesian inverse-problem toolbox
+ SPDE priors) and `eikonax` (`../eikonax`, differentiable eikonal solver) — check those source
trees directly when debugging across the package boundary.

**Procedure (non-trivial tasks):** inspect code/conventions → short plan → smallest appropriate
change → run checks → review diff.

## Environment (pixi)
- NEVER bare `python`/`pip`/`conda`/`uv`; always `pixi run ...`. Ask before adding deps. Never
  hand-edit `pixi.lock`; commit it with `pyproject.toml` (holds all pixi config; no `pixi.toml`).
- Envs: `default` (numpy/scipy/beartype/dolfinx/scifem/jax/pyvista/meshio), `dev` (+ruff,
  pre-commit, jupyter, plotting), `test` (+pytest, pytest-xdist, pytest-mock, nbclient). Tools need
  `-e`: `pixi run -e dev ruff check src` / `ruff format src`; `pixi run -e test pytest`.
- No `tests/` directory exists yet, though `pyproject.toml` already points `testpaths` at it —
  create it following `ls_bayesian`'s layout (`unit/`, `integration/`, `helpers.py`, `conftest.py`)
  for the first tests.
- `example_data/` (git-tracked) holds one small reference patient's data: `raw/` (mesh, fiber field,
  basis vectors) and `preprocessing/` (derived preprocessing outputs); not part of the installable
  package, just a fixed example. `working_data/` (gitignored) is where simulation-study sweep outputs
  go — one parameter-hash-named folder per run, each with its own metadata file.

## Architecture
Builds one `ls_bayesian.posterior.posterior.LogPosterior` by supplying `ls_bayesian`'s three
interfaces (`Likelihood`/`ParameterToSolutionMap`/`GaussianPrior`) with domain implementations. The
parameter $m$ is a fiber-orientation angle per mesh vertex.

- `mesh/io.py`: pyvista → dolfinx mesh conversion, matching vertex order (`create_dolfinx_mesh`),
  for `ls_bayesian`'s dolfinx-based SPDE prior; `load_pyvista_mesh` validates triangle-only.
- `mesh/interpolation.py`: `InterpolationStrategy` ABC assembling a sparse vertex→simplex matrix
  (linear-average vs. nearest-neighbor), moving the per-vertex angle onto per-simplex tensors.
- `posterior/prior.py`: `FiberAnglePrior` adapts an `ls_bayesian` bilaplacian `SPDEPrior` to
  `GaussianPrior` by delegation — the same decoupling pattern `ls_bayesian` uses internally.
- `posterior/fiber_tensor.py`: `FiberTensor` (`eikonax.tensorfield.AbstractSimplexTensor`) builds
  the per-simplex anisotropic conductivity tensor from a scalar fiber angle in each simplex's local
  tangent-plane basis and the longitudinal/transversal velocities; derivative via `jax.jacfwd`.
- `posterior/eikonal_map.py`: `EikonalParameterToSolutionMap` implements `ParameterToSolutionMap`
  directly by wrapping `eikonax`'s solver (forward) and discrete adjoint (gradient); Jacobian-/
  Hessian-vector products raise `NotImplementedError` (not exposed by `eikonax`, not needed for
  MAP). `_DerivativeCache` skips reassembling the adjoint when gradient follows forward at the same
  parameter — mirrors the caching `LogPosterior` does one level up.
- `posterior/likelihood.py`: builds an `ls_bayesian` `GaussianLogLikelihood` (i.i.d. homoscedastic
  noise) for sparse activation-time observations at a subset of vertices.
- `posterior/builder.py`: `PosteriorBuilder` wires the above from a `PosteriorSettings` dataclass.
  Prior, forward map, and likelihood are all built in `build()`, never `__init__` — a `LogPosterior`
  is tied to one dataset, so there's no cheaper partial rebuild for a different one.
- `optimization/`: MAP estimation, via `ls_bayesian.optimization`. `model.py` adapts a
  `LogPosterior` to `OptimizationModel` in two geometries — `CameronMartinPosteriorModel` (the
  metric-consistent choice: `LogPosterior.evaluate_gradient` is a "dual" vector, and applying the
  prior's covariance operator to it gives exactly the Cameron-Martin representer) and
  `EuclideanPosteriorModel` (the unmodified gradient, for Euclidean-only backends).
  `strategies.py`'s `OptimizerStrategy` (`CustomLBFGSStrategy`/`ScipyLBFGSBStrategy`) builds each
  optimizer backend together with its matching model as one pair — never mix the two models and
  backends across strategies. `builder.py`'s `MAPEstimationBuilder` mirrors `PosteriorBuilder`,
  delegating to the chosen strategy in `build()`.

## Design & style
- Priorities: numerical correctness > reproducibility > clear APIs > performance > convenience.
- Explicit over clever; composition over inheritance; frozen dataclasses for settings; adapter
  pattern to connect independently-developed interfaces (see `prior.py`). Vectorise, no element
  loops. Validate once at the public API (`ValueError` with the offending value).
- Google docstrings, full type hints, `@override`, ruff (`E,W,F,I,UP,B,SIM,TID252`, line-length
  100, no relative imports); no `# noqa` without a documented reason.
- Docstring math/shapes/dtypes; LaTeX via raw docstrings (`r"""..."""`); mkdocstrings full-path
  links, e.g. `` [`FiberAnglePrior`][bayes_cep.posterior.prior.FiberAnglePrior] ``.

## Numerical code
Before changing an algorithm: understand the math, preserve semantics, check shapes, vertex-vs-
simplex indexing, and conditioning; don't "fix" unusual math without knowing why it's there.
`mesh/io.py`, `posterior/eikonal_map.py`, and `posterior/prior.py` all assume one fixed vertex
ordering across the pyvista→dolfinx conversion and the vertex→simplex interpolation matrix.
