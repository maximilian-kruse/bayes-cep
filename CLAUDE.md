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
  (`scripts/run.py`, one run), `study` (`run/study_cli.py`), `example` and `example-preprocessing`/`-map`/
  `-mcmc` (regenerate the example data, all or one run; map depends on preprocessing, mcmc on
  map). The example directories are arguments of these tasks in `pyproject.toml`, not defaults in
  Python code.
- `example_data/` (git-tracked, not part of the package): `raw/` (mesh, fiber field, basis vectors)
  and the output of three reference runs, each a run directory with its records:
  `preprocessing/`, `map/` and `mcmc/` (chain gitignored). Result files are written flat
  (`--flat-results`); every other run keeps them in its `results/` subdirectory. `working_data/`
  (gitignored) holds study directories.

## Architecture
Builds one `ls_bayesian.posterior.posterior.LogPosterior` by supplying `ls_bayesian`'s three
interfaces (`Likelihood`/`ParameterToSolutionMap`/`GaussianPrior`) with domain implementations. The
parameter $m$ is a fiber-orientation angle per mesh vertex.

- `mesh/io.py`: pyvista → dolfinx mesh conversion matching vertex order; `load_pyvista_mesh`
  validates triangle-only. `mesh/interpolation.py`: `InterpolationStrategy` ABC assembling the
  sparse vertex→simplex matrix (linear-average or nearest-neighbor). `mesh/plotting.py`:
  off-screen pyvista rendering of per-simplex and per-vertex fields into PNGs.
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
  `mcmc/` (sampler builder) and `statistics/` (axial statistics, correlation length) serve the
  runs.

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
- `study.py`: sweep nodes (`Axis`/`Zip`/`Product` over dotted config paths), `StudySetup(run_type,
  base, sweep, root, executor)` (what a study module defines as `STUDY`: the runs, the `root`
  (the study directory, named after the study), and the `ExecutorSettings` they execute with, local or SLURM
  resources; resolves to `ResolvedRun`s; `run_directories()` gives the run directories of the
  created study, for other studies to read) and `Study` (`create` writes `<root>/study/`
  atomically with `study.json`, run ids and environment, and requires the input files of all runs
  to exist; `load` re-resolves the module and requires the recorded run ids; `execute_runs` skips
  done runs and, unless `include_active`, submitted/running ones; `plot_finished_runs`;
  `build_run_table`/`write_run_table`).
- `progress.py`: `StepReporter` (numbered, timed steps in the log) and `describe_array`.
- `study_cli.py`: the `study` command (`create`, `show`, `run`, `status`, `collect`, `report`), each
  taking the study module.

`single_runs/` holds the concrete runs, each module with its own config. The configs have no
defaults for scientific parameters: `reference.py` holds all of them (constants and
`reference_*_config` factories of the example data), and presets and studies start from there:
- `prior.py`: `PriorParameters`, `PriorRunConfig`, `PriorRun` (samples the prior; needs only the mesh).
- `preprocessing.py`: `PreprocessingRun` is the only place where data is generated (ground truth,
  prior mean, synthetic observations). `PreprocessedData` is its output (including the
  noise variance); ground truth and observation settings have no defaults on purpose, studies vary
  them.
- `inference.py`: `InferenceProblemConfig` (raw dir, `preprocessed_data_dir`, prior, eikonal,
  interpolation), embedded as `problem` in the MAP and MCMC configs, with `assemble_posterior`.
- `map.py`: `MapRun` only optimizes, from the prior mean; `mcmc.py`: `McmcRun` only samples, from the
  `initial_state_path`, a `.npy` file (the prior mean of the preprocessing run, or a MAP run's
  `map_estimate.npy`; the example data starts at the MAP). Both read the preprocessed data and
  nothing else of the data side, so many inference runs share one preprocessing run.

Each run plots in its own `report` method. A new run kind = config + `Run` subclass in one module.
`studies/*.py` define `STUDY`; they are not archived with a study (the recorded commit and patch
cover them), and set their own `root`. The reference factories of `single_runs/reference.py`
take the raw and preprocessed data directories as arguments (no defaults); each study module
states its `RAW_DIR`. A study of inference runs sweeps `problem.preprocessed_data_dir` over
`PREPROCESSING_STUDY.results_directories()` of a preprocessing study, which must be created and run
first.

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
