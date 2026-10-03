"""Compute the MAP estimate of the fiber-angle field, given preprocessed data.

`--data-dir` is required and never assumed, so the same script drives simulation studies over many
parameterizations, each pointed at its own data directory. Run
`pixi run python scripts/run_map.py --help` for the full option list, or
`pixi run map -- --data-dir <dir>` via the pixi task. Requires `<data-dir>/raw/` and
`<data-dir>/preprocessing/` to already exist (see `scripts/preprocess_data.py`).

The optimizer comes from exactly one `OptimizerStrategy`, chosen via `--optimizer`:
`custom-lbfgs-strategy` (default; L-BFGS in the prior's Cameron-Martin space, with cautious
updating, matching `bayes_cep.optimization.model`'s Cameron-Martin gradient) or
`scipy-lbfgs-b-strategy` (plain Euclidean L-BFGS-B via `scipy.optimize.minimize`, ignoring the
prior's geometry entirely). `custom-lbfgs-strategy`'s own line search is itself selectable via
`--optimizer.line-search-settings:<strategy-name>`: `armijo-backtracking-line-
search-settings` (default, sufficient decrease only) or `strong-wolfe-line-search-settings`
(bracketing + zoom, also enforces a curvature condition -- see
`bayes_cep.optimization.strategies`'s module docstring for why that matters).

The posterior's eikonal forward map interpolates the vertex-based angle field onto simplices via
exactly one `InterpolationStrategy`, chosen via `--interpolation`:
`nearest-neighbor-interpolation-strategy` (default, takes the value at the vertex nearest each
triangle's centroid) or `linear-interpolation-strategy` (averages a triangle's three vertex
values); see `mesh.interpolation`'s docstring.

The optimizer's `logger_settings` prints to the console and, by default, also to
`<data_dir>/logs/optimizer.log` (relative `--*-logfile-path` values are resolved against
`data_dir`). The prior and the posterior get their own loggers (`--prior-logfile-path`,
`--posterior-logfile-path`), file-only by default (`<data_dir>/logs/{prior,posterior}.log`): pass
`None` to any of the three `--*-logfile-path` options to disable that file. In addition,
`--run-logfile-path` (default `<data_dir>/logs/run_map.log`, overwritten on every run) receives the
script's own output -- start time, command line, settings, progress steps, and summaries -- so the
settings are stored with the run. The optimizer's iteration table is deliberately not part of it:
it only goes to the optimizer's own log (see above).

To compute a MAP estimate for `example_data/` with the reference settings below:

    pixi run python scripts/run_map.py --data-dir example_data
"""

from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import tyro
from ls_bayesian.common.logging import BaseLogger, LoggerSettings
from ls_bayesian.optimization.algorithms.custom_lbfgs import CustomLBFGSSettings

from bayes_cep.mesh.interpolation import (
    LinearInterpolationStrategy,
    NearestNeighborInterpolationStrategy,
)
from bayes_cep.mesh.io import load_pyvista_mesh
from bayes_cep.optimization.strategies import CustomLBFGSStrategy, ScipyLBFGSBStrategy
from bayes_cep.posterior.builder import PosteriorBuilder, PosteriorSettings
from bayes_cep.posterior.eikonal_map import EikonalSolverSettings
from bayes_cep.posterior.likelihood import LikelihoodSettings
from bayes_cep.posterior.prior import PriorSettings
from bayes_cep.reporting.console import (
    describe_array,
    report_step,
    resolve_logfile_path,
    run_logger,
)

# Reference settings, used only as `RunMAPSettings` defaults below; every one
# of them is overridable on the command line. `eikonal` mirrors `scripts/preprocess_data.py`'s
# reference forward-solver settings, since the MAP estimate's forward model should match the one
# the data was generated with unless a mismatch is being deliberately studied.
REFERENCE_INITIAL_SITE_IND = 12650
REFERENCE_LONGITUDINAL_VELOCITY = 1.5
REFERENCE_TRANSVERSAL_VELOCITY = 1.0
REFERENCE_PRIOR_KAPPA = 0.05
REFERENCE_PRIOR_TAU = 10.0
REFERENCE_PRIOR_SEED = 0
REFERENCE_NOISE_VARIANCE = 1e-3
REFERENCE_GRADIENT_NORM_TOLERANCE = 1.0
REFERENCE_MAX_NUM_ITERATIONS = 2000


# ==================================================================================================
@dataclass(frozen=True)
class InferencePriorSettings:
    r"""Bilaplacian SPDE prior hyperparameters used for MAP estimation.

    Independent of whatever prior (if any) generated the ground truth in `preprocessing/`: the
    inference prior is a modeling choice, not something read back from preprocessing outputs.

    Note:
        This duplicates the `kappa`/`tau`/`seed` fields of
        [`PriorSettings`][bayes_cep.posterior.prior.PriorSettings] on purpose: that class also
        requires `mean_vector`, an array loaded from `preprocessing/` at runtime, so it has no
        default and `tyro` cannot expose it as a command-line field. `main` copies these values
        into a `PriorSettings` together with the loaded mean.

    Attributes:
        kappa (float): SPDE parameter $\kappa > 0$, controlling correlation length. Defaults to the
            legacy reference value `0.05`.
        tau (float): SPDE parameter $\tau > 0$, controlling marginal variance. Defaults to the
            legacy reference value `10.0`.
        seed (int): Random seed for prior sampling. Defaults to `0`.
    """

    kappa: float = REFERENCE_PRIOR_KAPPA
    tau: float = REFERENCE_PRIOR_TAU
    seed: int = REFERENCE_PRIOR_SEED


# ==================================================================================================
@dataclass(frozen=True)
class RunMAPSettings:
    r"""Settings for `run_map.py`.

    Only `data_dir` has no default: every other field is a reference scientific setting (reproduced
    from the legacy single-patient MAP study), while the data location itself must always be given
    explicitly rather than assumed.

    Attributes:
        data_dir (Path): Directory containing a `raw/` subfolder (`mesh.vtu`, `basis_vecs.npy`) and
            a `preprocessing/` subfolder (`prior_mean_angle_field.npy`,
            `observed_vertex_indices.npy`, `observed_activation_times.npy`), e.g. produced by
            `scripts/preprocess_data.py`. MAP outputs are written to `<data_dir>/map/`.
        eikonal (EikonalSolverSettings): Eikonal forward-solver settings for the posterior's forward
            map.
        prior (InferencePriorSettings): Bilaplacian SPDE prior hyperparameters for MAP estimation.
        noise_variance (float): Assumed variance $\sigma^2$ of the i.i.d. Gaussian observation
            noise, used by the likelihood.
        interpolation (LinearInterpolationStrategy | NearestNeighborInterpolationStrategy): Strategy
            interpolating the vertex-based angle field onto simplices for the posterior's eikonal
            forward map; defaults to nearest-neighbor interpolation, mirroring
            `scripts/preprocess_data.py`'s default, since the MAP estimate's forward model should
            match the one the data was generated with unless a mismatch is being deliberately
            studied. Selected on the command line via `--interpolation:<strategy-name>`.
        optimizer (CustomLBFGSStrategy | ScipyLBFGSBStrategy): Strategy selecting and configuring
            the optimizer backend and its matching `OptimizationModel` geometry; defaults to the
            metric-consistent `CustomLBFGSStrategy`, which converges once the Cameron-Martin
            gradient norm is at most `1.0` (`gradient_norm_tolerance`), far looser than the
            library default of `1e-6`, so runs stop earlier; tighten it via
            `--optimizer.lbfgs-settings.gradient-norm-tolerance`. The iteration cap is `2000`
            (library default `1000`), adjustable via
            `--optimizer.lbfgs-settings.maximum-num-iterations`. The strategy's default seed
            scaling is tuned to this problem (`gamma_min=1e-10`, first-iteration
            `fallback_value=1e-6`; see `bayes_cep.optimization.strategies`). Selected on the
            command line via
            `--optimizer:<strategy-name>`.
        logger_settings (LoggerSettings): Output channels for iteration-by-iteration optimizer
            progress reports; prints to the console and, by default, to
            `<data_dir>/logs/optimizer.log` (a relative `logfile_path` is resolved against
            `data_dir`; pass `--logger-settings.logfile-path None` to disable the file).
        prior_logfile_path (Path | None): File the prior's evaluation diagnostics are logged to,
            resolved against `data_dir` if relative; never printed to the console. Defaults to
            `logs/prior.log`. `None` disables prior logging entirely.
        posterior_logfile_path (Path | None): File the composed `LogPosterior`'s evaluation
            diagnostics are logged to, resolved against `data_dir` if relative; never printed to
            the console. Defaults to `logs/posterior.log`. `None` disables posterior logging
            entirely.
        run_logfile_path (Path | None): File the script's own output (settings, progress,
            summaries; not the optimizer's iteration table, which goes to `logger_settings`' log
            file) is logged to in addition to the console, so the settings are stored with the
            run. Resolved against `data_dir` if relative; defaults to `logs/run_map.log`.
            Overwritten on every run; `None` disables it.
    """

    data_dir: Path
    eikonal: EikonalSolverSettings = field(
        default_factory=lambda: EikonalSolverSettings(
            initial_site_ind=REFERENCE_INITIAL_SITE_IND,
            longitudinal_velocity=REFERENCE_LONGITUDINAL_VELOCITY,
            transversal_velocity=REFERENCE_TRANSVERSAL_VELOCITY,
        )
    )
    prior: InferencePriorSettings = field(default_factory=InferencePriorSettings)
    noise_variance: float = REFERENCE_NOISE_VARIANCE
    interpolation: LinearInterpolationStrategy | NearestNeighborInterpolationStrategy = field(
        default_factory=NearestNeighborInterpolationStrategy
    )
    optimizer: CustomLBFGSStrategy | ScipyLBFGSBStrategy = field(
        default_factory=lambda: CustomLBFGSStrategy(
            lbfgs_settings=CustomLBFGSSettings(
                maximum_num_iterations=REFERENCE_MAX_NUM_ITERATIONS,
                gradient_norm_tolerance=REFERENCE_GRADIENT_NORM_TOLERANCE,
            )
        )
    )
    logger_settings: LoggerSettings = field(
        default_factory=lambda: LoggerSettings(logfile_path=Path("logs/optimizer.log"))
    )
    prior_logfile_path: Path | None = Path("logs/prior.log")
    posterior_logfile_path: Path | None = Path("logs/posterior.log")
    run_logfile_path: Path | None = Path("logs/run_map.log")


def _open_file_logger(
    stack: ExitStack, data_dir: Path, logfile_path: Path | None, prefix: str
) -> BaseLogger | None:
    """Open a file-only, console-silent logger under `data_dir`, or `None` if `logfile_path` is
    `None`."""
    resolved_path = resolve_logfile_path(data_dir, logfile_path)
    if resolved_path is None:
        return None
    logger_settings = LoggerSettings(print_to_console=False, logfile_path=resolved_path)
    return stack.enter_context(BaseLogger(logger_settings, prefix=prefix))


# ==================================================================================================
def main(settings: RunMAPSettings) -> None:
    """Run MAP estimation, logging its output to `settings.run_logfile_path`."""
    with run_logger(resolve_logfile_path(settings.data_dir, settings.run_logfile_path)) as logger:
        _run(settings, logger)


# ==================================================================================================
def _run(settings: RunMAPSettings, logger: BaseLogger) -> None:
    """Build the posterior and optimizer, run MAP estimation, cache to `<data_dir>/map`."""
    raw_dir = settings.data_dir / "raw"
    preprocessing_dir = settings.data_dir / "preprocessing"
    map_dir = settings.data_dir / "map"
    num_steps = 5

    logger.info("MAP estimation settings")
    logger.info(f"  data directory : {settings.data_dir}")
    logger.info(f"  prior          : {settings.prior}")
    logger.info(f"  noise variance : {settings.noise_variance}")
    logger.info(f"  interpolation  : {settings.interpolation}")
    logger.info(f"  eikonal solver : {settings.eikonal}")
    logger.info(f"  optimizer      : {settings.optimizer}")
    logger.info("")

    with report_step(
        logger, 1, num_steps, f"Loading raw and preprocessed data from {settings.data_dir}"
    ):
        mesh = load_pyvista_mesh(raw_dir / "mesh.vtu")
        basis_vectors = np.load(raw_dir / "basis_vecs.npy")
        prior_mean_angle_field = np.load(preprocessing_dir / "prior_mean_angle_field.npy")
        observed_vertex_indices = np.load(preprocessing_dir / "observed_vertex_indices.npy")
        observed_activation_times = np.load(preprocessing_dir / "observed_activation_times.npy")
    logger.info(f"      mesh: {mesh.n_points} vertices, {mesh.n_cells} triangles")
    logger.info(f"      observed {observed_vertex_indices.shape[0]} of {mesh.n_points} vertices")
    logger.info(f"      {describe_array('observed activation times', observed_activation_times)}")

    posterior_settings = PosteriorSettings(
        mesh=mesh,
        basis_vectors=basis_vectors,
        prior_settings=PriorSettings(
            mean_vector=prior_mean_angle_field,
            kappa=settings.prior.kappa,
            tau=settings.prior.tau,
            seed=settings.prior.seed,
        ),
        eikonal_settings=settings.eikonal,
        likelihood_settings=LikelihoodSettings(
            num_vertices=mesh.points.shape[0],
            observed_vertex_indices=observed_vertex_indices,
            data_vector=observed_activation_times,
            noise_variance=settings.noise_variance,
        ),
    )

    with ExitStack() as stack:
        prior_logger = _open_file_logger(
            stack, settings.data_dir, settings.prior_logfile_path, prefix="prior"
        )
        posterior_logger = _open_file_logger(
            stack, settings.data_dir, settings.posterior_logfile_path, prefix="posterior"
        )
        interpolation_name = type(settings.interpolation).__name__
        with report_step(
            logger,
            2,
            num_steps,
            f"Building the posterior (SPDE prior, {interpolation_name} forward map)",
        ):
            posterior_builder = PosteriorBuilder(
                posterior_settings,
                interpolation_strategy=settings.interpolation,
                prior_logger=prior_logger,
                posterior_logger=posterior_logger,
            )
            log_posterior = posterior_builder.build()

        optimizer_logger_settings = replace(
            settings.logger_settings,
            logfile_path=resolve_logfile_path(
                settings.data_dir, settings.logger_settings.logfile_path
            ),
        )
        optimizer_logger = stack.enter_context(BaseLogger(optimizer_logger_settings, prefix="map"))
        optimizer_name = type(settings.optimizer).__name__
        with report_step(logger, 3, num_steps, f"Building the optimizer ({optimizer_name})"):
            model, optimizer = settings.optimizer.build(
                log_posterior, posterior_builder.prior, optimizer_logger
            )

        # The optimizer's own iteration table (iteration, time, loss, gradient norm) is printed to
        # the console by `optimizer_logger`, unless `--logger-settings.print-to-console` is off.
        with report_step(
            logger,
            4,
            num_steps,
            "Running MAP estimation (iteration table: console and optimizer log)",
        ):
            result = optimizer.run(initial_guess=prior_mean_angle_field, model=model)

    outcome = "converged" if result.success else "did not converge"
    logger.info(f"      MAP estimation {outcome} after {result.num_iterations} iterations")
    logger.info(f"      status: {result.status_message}")
    logger.info(f"      loss: {result.loss_history[0]:.6e} -> {result.loss_history[-1]:.6e}")
    logger.info(
        f"      gradient norm: {result.gradient_norm_history[0]:.6e} -> "
        f"{result.gradient_norm_history[-1]:.6e}"
    )
    logger.info(f"      {describe_array('MAP estimate [rad]', result.result)}")

    outputs = {
        "map_estimate.npy": result.result,
        "map_loss_history.npy": result.loss_history,
        "map_gradient_norm_history.npy": result.gradient_norm_history,
    }
    with report_step(logger, 5, num_steps, f"Writing outputs to {map_dir}"):
        map_dir.mkdir(exist_ok=True)
        for filename, array in outputs.items():
            np.save(map_dir / filename, array)
    for filename, array in outputs.items():
        logger.info(f"      {filename} {array.shape} {array.dtype}")
    logger.info("MAP estimation finished.")


if __name__ == "__main__":
    main(tyro.cli(RunMAPSettings))
