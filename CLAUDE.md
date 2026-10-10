# CLAUDE.md

Guidance for Claude Code (claude.ai/code) in this repository.

# bayes-cep
Non-parametric Bayesian inference of myocardial fiber orientations from electrical activation-time
data (Python ≥3.14). Assembles a fiber-orientation posterior from two sibling packages consumed as
local editable path deps: `ls_bayesian` (`../ls-bayesian`, Bayesian inverse-problem toolbox + SPDE
priors) and `eikonax` (`../eikonax`, differentiable eikonal solver). Check their source directly
when debugging across the package boundary.

**Procedure (non-trivial tasks):** inspect code/conventions → short plan → smallest appropriate
change → run checks → review diff.

## Environment (pixi)
- NEVER bare `python`/`pip`/`conda`/`uv`; always `pixi run ...`. Ask before adding deps. Never
  hand-edit `pixi.lock`; commit it with `pyproject.toml` (all pixi config; no `pixi.toml`).
- Envs: `default` (numpy/scipy/pandas/pyarrow/submitit/dolfinx/scifem/jax/pyvista/matplotlib/...),
  `dev` (+ruff, pre-commit, jupyter), `test` (+pytest). Tools need `-e`:
  `pixi run -e dev ruff check src` / `ruff format src`; `pixi run -e test pytest`.
- No `tests/` yet, though `pyproject.toml` points `testpaths` there; create it following
  `ls_bayesian`'s layout (`unit/`, `integration/`, `helpers.py`, `conftest.py`).
- Tasks (run from the repository root; relative config paths resolve against it): `single`
  (`scripts/run.py`, one run), `study` (`run/cli.py`), `example` and `example-preprocessing`/`-map`/
  `-mcmc` (regenerate the example data, all or one stage; later stages read earlier ones, e.g.
  `pixi run example config:example-synthetic`).
- `example_data/` (git-tracked, not part of the package): `raw/` (mesh, fiber field, basis vectors)
  and the reference MAP run in the plain layout without JSON records (`preprocessing/`, `map/`,
  `mcmc/` gitignored, `logs/`). `working_data/` (gitignored) holds study directories.

## Architecture
Builds one `ls_bayesian.posterior.posterior.LogPosterior` by supplying `ls_bayesian`'s three
interfaces (`Likelihood`/`ParameterToSolutionMap`/`GaussianPrior`) with domain implementations. The
parameter $m$ is a fiber-orientation angle per mesh vertex.

- `mesh/io.py`: pyvista → dolfinx mesh conversion matching vertex order; `load_pyvista_mesh`
  validates triangle-only. `mesh/interpolation.py`: `InterpolationStrategy` ABC assembling the
  sparse vertex→simplex matrix (linear-average or nearest-neighbor).
- `posterior/prior.py`: `FiberAnglePrior` adapts an `ls_bayesian` bilaplacian `SPDEPrior` to
  `GaussianPrior` by delegation (the decoupling pattern `ls_bayesian` uses internally).
- `posterior/fiber_tensor.py`: `FiberTensor` (an `eikonax` `AbstractSimplexTensor`) builds the
  per-simplex anisotropic conductivity from the fiber angle in each simplex's tangent-plane basis
  and the longitudinal/transversal velocities; derivative via `jax.jacfwd`.
- `posterior/eikonal_map.py`: `EikonalParameterToSolutionMap` wraps `eikonax`'s solver (forward) and
  discrete adjoint (gradient); Jacobian-/Hessian-vector products raise `NotImplementedError` (not
  needed for MAP). `_DerivativeCache` skips reassembling the adjoint when gradient follows forward
  at the same parameter.
- `posterior/likelihood.py`: `GaussianLogLikelihood` (i.i.d. homoscedastic noise) for sparse
  activation-time observations. `posterior/builder.py`: `PosteriorBuilder` wires the pieces from a
  `PosteriorSettings`; everything is built in `build()`, never `__init__` (a `LogPosterior` is tied
  to one dataset).
- `optimization/`: MAP estimation via `ls_bayesian.optimization`. `model.py` adapts a
  `LogPosterior` in two geometries: `CameronMartinPosteriorModel` (metric-consistent: the gradient
  is a "dual" vector, and the prior covariance maps it to the Cameron-Martin representer) and
  `EuclideanPosteriorModel` (raw gradient). `strategies.py`: `OptimizerStrategy`
  (`CustomLBFGSStrategy`/`ScipyLBFGSBStrategy`) builds a backend together with its matching model
  as one pair, never mix them; callers use `strategy.build(log_posterior, prior, logger)` directly.
- `preprocessing/` (ground-truth strategies, constant prior mean, synthetic observations),
  `mcmc/` (sampler builder), `statistics/` (axial statistics, correlation length) serve the runs.

## Runs and studies
A **run** is a pure function of one frozen config, identified by an 8-hex content hash of its
canonical JSON; a **study** is a fixed list of runs from a base config plus sweeps. `run/` is
generic (no domain imports); concrete runs live in `single_runs/` and studies in `studies/`, found
through the `PYTHONPATH` set by the pixi activation. Modules of `run/`:
- `config.py`: `RunConfig` (frozen dataclass; JSON round trip driven by type hints and `__type__`
  tags, `run_id`, `describe()`). Class names are part of a config's identity.
- `template.py`: `Run[ConfigT]` ABC. `execute(run_dir, environment=None)` is the only way to run: it
  records `config.json`, `metadata.json`, `status.json`, `run.log`, `metrics.json` and failures or
  interruptions (state `RunState`: submitted/running/done/failed; no directory = pending). A subclass
  names its config as the generic argument (gives `config_type`), sets `outputs`, implements
  `_execute`, `report` (plots; separate local step, so cluster runs stay headless) and
  `input_files` (hashed into the metadata). `open_run_logger` is public for partial runs.
- `directories.py`: on-disk layout (`RunDirectory`, `StudyDirectory`), atomic JSON records,
  repository root and `resolve_repository_path`, the time format.
- `provenance.py`: `Environment` (git states, editable packages, `pixi.lock` hash; collected once
  per submission), `RunMetadata` (time, host, SLURM ids, input hashes), `EnvironmentArchive`
  (lock/pyproject/conda spec/patch of uncommitted changes, copied into the study directory).
- `executor.py`: `Executor` runs given `Run` objects in their run directories: `local` serially in
  this process, `slurm` as a `submitit` job array (records `SUBMITTED` first; `wait=False` queues
  and returns). Knows nothing about studies.
- `study.py`: sweep nodes (`Axis`/`Zip`/`Product` over dotted config paths), `Study(run_type, base,
  sweep)` (definition, resolves to `ResolvedRun`s) and `CreatedStudy` (named by its module and
  `--root`: `create` writes `<root>/<name>/study/` atomically with `study.json`, run ids and
  environment; `load` re-resolves the module and requires the recorded run ids; `execute_runs`
  skips done runs and, unless `include_active`, submitted/running ones; `plot_finished_runs`;
  `build_run_table`/`write_run_table`).
- `cli.py`: the `study` command (`create`, `show`, `run`, `status`, `collect`, `report`), each
  taking the study module.

`single_runs/` holds the concrete runs: `config.py` (`PriorRunConfig`, `MapRunConfig`; MAP ground
truth and observation settings have no defaults on purpose, studies vary them), `prior.py`
(`PriorRun`), `map.py` (`MapRun`, `MapStage`, `MapPaths`; `run_stages` performs stages separately),
`example_data.py` (example layout without run records, built from `run_stages`), `progress.py`,
`plots.py`. A new run kind = config + `Run` subclass. `studies/*.py` define `STUDY`; they are not
archived with a study (the recorded commit and patch cover them).

## Design & style
- Priorities: numerical correctness > reproducibility > clear APIs > performance > convenience.
- Explicit over clever; composition over inheritance; frozen dataclasses for settings; adapters for
  independently developed interfaces. Vectorise, no element loops. Validate once at the public API
  (`ValueError` with the offending value). Long, self-explanatory names. Prefer methods over free
  functions for behavior private to a class.
- Google docstrings, full type hints, `@override`, ruff (`E,W,F,I,UP,B,SIM,TID252`, line-length
  100, no relative imports); no `# noqa` without a documented reason.
- Docstring math/shapes/dtypes; LaTeX via raw docstrings (`r"""..."""`); mkdocstrings full-path
  links, e.g. `` [`FiberAnglePrior`][bayes_cep.posterior.prior.FiberAnglePrior] ``.

## Numerical code
Before changing an algorithm: understand the math, preserve semantics, check shapes, vertex-vs-
simplex indexing, and conditioning; don't "fix" unusual math without knowing why it's there.
`mesh/io.py`, `posterior/eikonal_map.py` and `posterior/prior.py` assume one fixed vertex ordering
across the pyvista→dolfinx conversion and the vertex→simplex interpolation matrix.
