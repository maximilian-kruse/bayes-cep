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
- There is no test suite (removed on purpose, commit `92fb8f1`); the `test` env stays for a future
  one. Do not recreate tests unasked.
- CLIs in `src/bayes_cep/cli/`: `bcep-single` (`run_single`) and `bcep-study` (`run_study`); from
  another directory `pixi run --manifest-path <repo> bcep-single ...`. They import
  `single_runs`/`studies` from the repository root via `PYTHONPATH`.
- Tasks (from the repository root; relative config paths resolve against it): `single`, `study`,
  `example` and `example-preprocessing`/`-map`/`-mcmc`/`-laplace` (regenerate the example data; map
  depends on preprocessing, mcmc and laplace on map). The example directories are arguments of the
  tasks in `pyproject.toml`, not defaults in Python code.
- `example_data/` (git-tracked): `raw/` (mesh, fiber field, basis vectors) and the output of the
  reference runs (`preprocessing/`, `map/`, `mcmc/` with the chain gitignored, `laplace/`), written
  with `--example-layout`: `metadata/` (config, metadata, status, metrics, logs) and `results/`.
  Other runs keep their records in the run directory and the results in `results/`. `working_data/`
  (gitignored) holds study directories.

## Architecture
Builds one `ls_bayesian` `LogPosterior` by supplying its three interfaces (`Likelihood`,
`ParameterToSolutionMap`, `GaussianPrior`) with domain implementations. The parameter $m$ is a
fiber-orientation angle per mesh vertex.

- `mesh/`: `io.py` (pyvista → dolfinx in matching vertex order, triangle-only validation),
  `topology.py` (the one place that reads triangles), `geodesic.py`, `interpolation.py` (vertex→
  simplex matrix; the strategy is always passed explicitly).
- `posterior/`: `PriorParameters`, `FiberAnglePrior` (adapter of an `ls_bayesian` `SPDEPrior`),
  `FiberTensor` (per-simplex conductivity from the angle), `EikonalParameterToSolutionMap`
  (eikonax forward, discrete adjoint and the forward triangular solve for Jacobian-vector products;
  Hessian-vector products not implemented), `GaussianLogLikelihood`, `PosteriorBuilder` (stateless;
  `build_components()` returns the parts uncomposed). The prior is `LogPosterior.prior`.
  `_DerivativeCache` is a 3-entry LRU keyed on exactly equal parameter vectors (no tolerance: nearby
  MCMC proposals must not reuse another state's adjoint). `FiberTensor` and eikonax run in single
  precision, the posterior in double.
- `laplace/`: `LaplaceBuilder(settings, components).build(map_estimate)` gives a low-rank Laplace
  approximation (`ls_bayesian.lowrank`) from `GaussNewtonMisfitHessian` and
  `PriorReferenceCovariance`. `calibration.py`: sample-informed preconditioner (DILI-style) from
  pilot chains; PMALA only uses its covariance, `recenter` is for `PCN_PRECONDITIONED`.
- `mcmc/`: `MCMCBuilder` for pCN, MALA and, with a Gaussian `approximation`,
  `PCN_PRECONDITIONED` and `PMALA` (a prior mean goes into the potential). The posterior is far
  from Gaussian: step widths of the preconditioned samplers must be small (0.05 in the reference),
  the rank must reach the number of observations.
- `optimization/`: `InitialGuessStrategy` (prior mean, prior draw, `.npy`; `ChainStateInitialGuess`
  for MCMC, since the MAP and the chain mean are not in the typical set). `OptimizerStrategy.build(
  log_posterior, logger)` returns an optimizer with its matching model as one pair, never mix them:
  `CameronMartinPosteriorModel` for `CustomLBFGSStrategy` and `NewtonCGStrategy`,
  `EuclideanPosteriorModel` for `ScipyLBFGSBStrategy`. The reference MAP has two `stages`
  (`MapRunConfig.stages`, each starting at the estimate of the one before): L-BFGS (1000
  iterations), then Newton-CG with Gauss-Newton products (at most 50). Newton-CG stalls at the
  float32 noise of the cost (5e-4 to 3e-3), so it stops when the predicted decrease is below 1e-3.
  The stage histories are joined; `map_stage_end_indices.npy` marks the boundaries and the metrics
  hold `stage<k>_*` entries.
- `preprocessing/` (ground-truth strategies, synthetic observations, prior-mean strategies),
  `statistics/` (chains read block by block through `SampleStore`; angles are axial, mod $\pi$:
  means via the doubled-angle resultant, diagnostics on `sin(2(theta - mean))`).
- `postprocessing/`: plotting on arrays and paths, never configs, used only by the report step; no
  `pyplot` in the runs.
- Settings in `src/` carry no defaults for scientific parameters; the reference values are in
  `single_runs/reference.py`.

## Runs and studies
A **run** is a pure function of one frozen config, identified by an 8-hex content hash of its
canonical JSON (class names are part of the identity; input file contents and code state go to
`metadata.json`, not into the id). A **study** is a fixed list of runs from a base config plus
sweeps. `run/` is generic (no domain imports); concrete runs are in `single_runs/`, studies in
`studies/`. Modules of `run/`:
- `config.py`: `RunConfig` (JSON round trip from type hints and `__type__` tags, `run_id`).
- `template.py`: `Run[ConfigT]` ABC; `execute(run_dir, environment)` is the only way to run and
  records config, metadata, status, log, metrics and failures. A subclass sets `config_type` and
  `outputs` and implements `_execute`, `report` (plots, a separate local step so cluster runs stay
  headless) and `input_files` (hashed into the metadata). `_check_outputs` compares the written
  files with `outputs`: a missing declared file fails the run, an undeclared one is a warning.
  Skippable outputs go in `optional_outputs`; file names never depend on the config.
- `directories.py`: layout (`RunDirectory`, `StudyDirectory`), atomic JSON records,
  `resolve_repository_path`, `ArrayRecord` (frozen dataclass, one `.npy` per field from
  `FILE_NAMES`; base of result records).
- `provenance.py` (environment, metadata), `executor.py` (`Executor`: local or `submitit` SLURM
  array; the only code that removes files of an earlier attempt), `progress.py` (`StepReporter`),
  `study.py` (sweep nodes `Axis`/`Zip`/`Product` over dotted config paths, `StudySetup`, `Study`
  with `create`/`load`/`execute_runs`).

`single_runs/` holds the concrete runs, each module with its own config and `reference.py` the
constants and `reference_*_config` factories:
- `data.py` (file layout), `preprocessing.py` (the only place where data is generated),
  `inference.py` (`InferenceProblemConfig`, embedded as `problem` in the MAP and MCMC configs).
- `map.py` only optimizes, `mcmc.py` only samples; both read the preprocessed data and nothing else
  of the data side. `prior.py` samples the prior; `laplace.py` builds the Laplace approximation at a
  stored MAP; `heldout.py` is the cross-validation MAP (`statistics/cross_validation.py`).
- MCMC variants are subclasses with their own config class, so other run ids do not change:
  `mcmc_laplace.py` (Laplace proposal/preconditioner), `mcmc_calibrated.py` (+ pilot-chain
  calibration; `notebooks/mcmc_calibrated.ipynb`), `calibrate.py` + `mcmc_stored_calibration.py`
  (calibration stored once, shared by a sweep), `rejection_diagnostics.py` and `hmc_trajectory.py`
  (where PMALA rejects; Hamiltonian trajectories of `PHMCAlgorithm`). `mcmc.py` streams chain
  diagnostics (`ChainDiagnostics`); its `report` only plots them.
- Diagnostics of the sampling geometry (read stored chains or their projections, no config of the
  posterior beyond the problem): `chain_projection.py` (chain onto the Laplace eigenvectors),
  `hessian_drift.py` and `hessian_average.py` (Gauss-Newton Hessian at chain states, its average as
  covariance), `calibration_validation.py` (leave-one-chain-out scores of calibrations, with a
  Hessian-average shrinkage target), `averaged_gaussian.py` (Gaussian from the Hessian average,
  read like a calibration), `mass_stability.py` (PHMC step-angle limit per mass matrix),
  `gradient_check.py` (adjoint gradient against finite differences by local acceptance).
  Eigenvectors are reproducible across runs on one machine (the runs check the eigenvalues).
- `prior_fit.py` scores a kappa/tau grid against a patient field (`studies/prior_moment_fit.py`).
- Post-processing scripts in `scripts/`, run after `bcep-study collect`:
  `postprocess_map_study.py`, `postprocess_mcmc_study.py`, `postprocess_prior_fit_study.py`,
  `postprocess_prior_selection_study.py` (prior selection by skill, Laplace evidence or held-out
  score).
- A new run type = config + `Run` subclass in one module with a class attribute `name` (the
  `config:<name>` subcommand; no name = not on the command line). `bayes_cep.cli.run_single` builds
  a `RunRegistry` (`run/registry.py`) from the named run types in `single_runs` plus each
  `--run-file path.py`. Result, plot and log file names are defined once per run (an `ArrayRecord`
  or module constants, `METRICS_RECORD` for the metrics) and used by `outputs`, `_execute` and
  `report`; never repeat a file name as a string literal there.

`studies/*.py` define `STUDY`, set their own `root` and `RAW_DIR`, and are not archived with a study.
A bare relative path in a config is relative to the repository root (`resolve_repository_path`). A
study module gives its paths relative to itself with `module_relative_path(__file__, "../working_data/
<name>")`; the result is repo-relative inside the repository, else absolute, so run ids do not
depend on the checkout location. A study of inference runs sweeps `problem.data_dir` over
`PREPROCESSING_STUDY.results_directories()`; create and run the preprocessing study first.

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
