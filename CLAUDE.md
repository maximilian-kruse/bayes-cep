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
- Envs: `default`, `dev` (+ruff, pre-commit, jupyter), `test` (+pytest). Tools need `-e`:
  `pixi run -e dev ruff check src` / `ruff format src`.
- There is no test suite: the tests were removed on purpose (commit `92fb8f1`). The `test` env and
  pytest settings stay for a future one; do not recreate tests unasked.
- CLIs in `src/bayes_cep/cli/`, installed as the scripts `bcep-single` (`run_single`) and
  `bcep-study` (`run_study`); from another directory:
  `pixi run --manifest-path <repo> bcep-single ...` (relative paths then resolve against that
  directory). They still import `single_runs`/`studies` from the repository root via `PYTHONPATH`.
- Tasks (run from the repository root; relative config paths resolve against it): `single`, `study`,
  `example` and `example-preprocessing`/`-map`/`-mcmc` (regenerate the example data; map depends on preprocessing, mcmc on map). The
  example directories are arguments of these tasks in `pyproject.toml`, not defaults in Python code.
- `example_data/` (git-tracked): `raw/` (mesh, fiber field, basis vectors) and the output of three
  reference runs (`preprocessing/`, `map/`, `mcmc/`, chain gitignored), written flat
  (`--flat-results`); other runs keep results in `results/`. `working_data/` (gitignored) holds
  study directories.

## Architecture
Builds one `ls_bayesian` `LogPosterior` by supplying its three interfaces (`Likelihood`,
`ParameterToSolutionMap`, `GaussianPrior`) with domain implementations. The parameter $m$ is a
fiber-orientation angle per mesh vertex.

- `mesh/`: `io.py` (pyvista → dolfinx in matching vertex order, triangle-only validation),
  `topology.py` (the one place that reads triangles; edges, boundary), `geodesic.py` (edge-length
  graph, distances), `interpolation.py` (vertex→simplex matrix; the strategy is always passed
  explicitly).
- `posterior/`: `PriorParameters` (kappa, tau, seed; the one class for them; the mean is passed
  separately), `FiberAnglePrior` (delegating adapter of an `ls_bayesian` `SPDEPrior`),
  `FiberTensor` (per-simplex conductivity from the angle; `jax.jacfwd` derivative),
  `EikonalParameterToSolutionMap` (eikonax forward + discrete adjoint; Jacobian-/Hessian-vector
  products not implemented, not needed for MAP), `GaussianLogLikelihood`, and `PosteriorBuilder`
  (everything built in `build()`, stateless; a `LogPosterior` is tied to one dataset).
  The prior is `LogPosterior.prior`, so optimizer models and the MCMC builder take only the posterior.
  - `_DerivativeCache` reuses the adjoint only for exactly the same parameter vector
    (`np.array_equal` on a copy): a tolerance would let nearby MCMC proposals reuse another state's
    adjoint. `FiberTensor` and eikonax run in single precision, the posterior in double.
- `optimization/`: `OptimizerStrategy.build(log_posterior, logger)` returns an optimizer together
  with its matching model as one pair, never mix them: `CameronMartinPosteriorModel` (the gradient
  is a dual vector; the prior covariance maps it to the Cameron-Martin representer) for
  `CustomLBFGSStrategy`, `EuclideanPosteriorModel` (raw gradient) for `ScipyLBFGSBStrategy`.
- `preprocessing/` (ground-truth strategies, constant prior mean, synthetic observations), `mcmc/`
  (sampler builder) and `statistics/` serve the runs. `statistics/` reads (zarr) chains block by
  block through `SampleStore`; angles are axial (mod $\pi$): means via the doubled-angle resultant,
  diagnostics and correlations on `sin(2(theta - mean))`. The in-memory and the streamed axial mean
  are separate on purpose (see the module docstring of `axial_statistics.py`).
- `postprocessing/`: generic plotting on arrays and paths, never configs, imported only by the
  report step. Reports use these helpers; no `pyplot` in the runs.
- Settings in `src/` carry no defaults for scientific parameters; the reference values are in
  `single_runs/reference.py`.

## Runs and studies
A **run** is a pure function of one frozen config, identified by an 8-hex content hash of its
canonical JSON (class names are part of the identity; input file contents and code state are in
`metadata.json`, not in the id). A **study** is a fixed list of runs from a base config plus sweeps.
`run/` is generic (no domain imports); concrete runs are in `single_runs/`, studies in `studies/`,
found through the `PYTHONPATH` set by pixi. Modules of `run/`:
- `config.py`: `RunConfig` (JSON round trip driven by type hints and `__type__` tags, `run_id`).
- `template.py`: `Run[ConfigT]` ABC; `execute(run_dir, environment)` is the only way to run and
  records config, metadata, status, log, metrics and failures. A subclass sets `config_type` (to the
  same class as its generic argument) and `outputs`, and implements `_execute`, `report` (plots, a
  separate local step so cluster runs stay headless) and `input_files` (hashed into the metadata).
- `directories.py`: on-disk layout (`RunDirectory`, `StudyDirectory`), atomic JSON records,
  `resolve_repository_path`, and `ArrayRecord` (a frozen dataclass stored as one `.npy` per field
  from its `FILE_NAMES` mapping), the base of result records.
- `provenance.py`: `Environment` (git states, editable packages, `pixi.lock` hash), `RunMetadata`,
  `EnvironmentArchive` (specification copied into the study directory).
- `executor.py`: `Executor` runs `Run` objects locally in this process or as a `submitit` SLURM job
  array (`ExecutorSettings.cluster`). It is the only one that removes the files of an earlier attempt
  (`bayes_cep.cli.run_single` only decides whether it may).
- `study.py`: sweep nodes (`Axis`/`Zip`/`Product` over dotted config paths), `StudySetup` (what a
  study module defines as `STUDY`; `results_directories()` for other studies to read) and `Study`
  (`create` writes `<root>/study/` atomically and requires all input files to exist; `root` itself
  may already exist, e.g. as the folder of the study module, only `study/` must not; `load`
  re-resolves the module and requires the recorded run ids; `execute_runs` skips done runs and,
  unless `include_active`, submitted/running ones).
- `progress.py`: `StepReporter`, `describe_array`.

`single_runs/` holds the concrete runs, each module with its own config (no defaults for scientific
parameters; `reference.py` holds them as constants and `reference_*_config` factories):
- `data.py`: file layout of the data defined once: `RawDataDirectory`, `PreprocessedData`.
- `preprocessing.py`: the only place where data is generated. `inference.py`: `InferenceProblemConfig`
  (embedded as `problem` in the MAP and MCMC configs) and `assemble_posterior`.
- `map.py` only optimizes (from the prior mean); `mcmc.py` only samples (from `initial_state_path`, a
  `.npy`: the prior mean or a MAP estimate). Both read the preprocessed data and nothing else of the
  data side, so many inference runs share one preprocessing run. `prior.py` samples the prior.
  After sampling, `mcmc.py` streams the chain diagnostics (`ChainDiagnostics` record: ESS and
  autocorrelation time per vertex; trace and autocorrelation of `diagnostics.plotted_components`,
  none/one/several vertices) and its `report` only plots them.
- A new run type = config + `Run` subclass in one module, with a class attribute `name` (the
  `config:<name>` subcommand of `bayes_cep.cli.run_single`; no name = not on the command line). There is
  no global registry: `bayes_cep.cli.run_single` creates a `RunRegistry` (`run/registry.py`), registers
  every named run type defined in the `single_runs` package, plus those of each file given as
  `--run-file path.py` (repeatable, before the subcommand), and builds its subcommands from it.
  Result file names are defined once per run (an `ArrayRecord` or module constants) and used by
  `outputs`, `_execute` and `report`.

`studies/*.py` define `STUDY`, set their own `root` and `RAW_DIR`, and are not archived with a study
(the recorded commit and patch cover them). A bare relative path in a config is relative to the
repository root (`resolve_repository_path`), not to the module or the working directory. A study
module gives its paths (`root`, `RAW_DIR`, other input paths) relative to itself with
`module_relative_path(__file__, "../working_data/<name>")` (`run/directories.py`), so a module (and
its root, data) may live outside the repository. The result is repo-relative inside the repository
(`repository_relative_path`, also applied to `root` by `StudySetup`), else absolute: the run id
hashes the config paths, which derive from `root` for studies that read other studies' results, so
an absolute path inside the repository would tie run ids (and `Study.load` of existing studies) to
the checkout location.
A study of inference runs sweeps `problem.data_dir` over `PREPROCESSING_STUDY.results_directories()`;
that preprocessing study must be created and run first.

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
