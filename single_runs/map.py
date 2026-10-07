"""MAP estimation run: generate synthetic data from a ground truth and compute the MAP estimate.

The run has stages, to regenerate the example data piece by piece: `preprocessing` (ground truth,
prior mean, synthetic observations), `map` (MAP estimate; reads the preprocessing data) and `mcmc`
(chain started at the MAP estimate; reads both). `all` runs them in sequence, the MCMC stage only if
the configuration has MCMC settings.

Classes:
    MapRun: The MAP estimation run.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import override

import numpy as np
from ls_bayesian.common.logging import BaseLogger, LoggerSettings
from ls_bayesian.mcmc.sampler import SamplerSettings
from ls_bayesian.posterior.posterior import LogPosterior
from pyvista import UnstructuredGrid

from bayes_cep.mcmc.builder import MCMCBuilder, MCMCSettings, ZarrStorageSettings
from bayes_cep.mesh.io import load_pyvista_mesh
from bayes_cep.posterior.builder import PosteriorBuilder, PosteriorSettings
from bayes_cep.posterior.eikonal_map import EikonalParameterToSolutionMap
from bayes_cep.posterior.likelihood import LikelihoodSettings
from bayes_cep.posterior.prior import PriorSettings
from bayes_cep.preprocessing.prior_mean import build_constant_prior_mean
from bayes_cep.preprocessing.synthetic_observations import generate_synthetic_observations
from bayes_cep.run.config import resolve_path
from bayes_cep.run.logging import Steps, describe_array
from bayes_cep.run.template import Metrics, Run
from bayes_cep.statistics.axial_statistics import compute_axial_data_diff
from single_runs.config import MapRunConfig


# ==================================================================================================
@dataclass(frozen=True)
class _Paths:
    """Where the stages write: the folders of the example layout, or `results/` for a run."""

    data_dir: Path
    map_dir: Path
    mcmc_dir: Path
    optimizer_log: Path
    sampler_log: Path

    @classmethod
    def of(cls, run_dir: Path, example_layout: bool) -> _Paths:
        if example_layout:
            logs = run_dir / "logs"
            return cls(
                run_dir / "preprocessing",
                run_dir / "map",
                run_dir / "mcmc",
                logs / "optimizer.log",
                logs / "sampler.log",
            )
        results = run_dir / "results"
        return cls(results, results, results, run_dir / "optimizer.log", run_dir / "sampler.log")


# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class _Data:
    """Output of the preprocessing stage."""

    ground_truth: np.ndarray
    prior_mean: np.ndarray
    observed_vertex_indices: np.ndarray
    observed_activation_times: np.ndarray


# ==================================================================================================
class MapRun(Run[MapRunConfig]):
    """Generate synthetic observations from a ground truth, then compute the MAP estimate."""

    config_type = MapRunConfig
    stages = ("all", "preprocessing", "map", "mcmc")
    outputs = {
        "results/ground_truth_angle_field.npy": "Ground-truth fiber angle [rad] per vertex.",
        "results/prior_mean_angle_field.npy": "Constant prior mean [rad] per vertex: the axial "
        "mean of the ground truth.",
        "results/observed_vertex_indices.npy": "Indices of the observed vertices.",
        "results/observed_activation_times.npy": "Noisy synthetic activation times at those "
        "vertices.",
        "results/map_estimate.npy": "MAP estimate of the fiber angle [rad] per vertex.",
        "results/map_loss_history.npy": "Loss at every accepted optimizer iterate.",
        "results/map_gradient_norm_history.npy": "Gradient norm at every accepted iterate.",
        "metrics.json": "Optimization statistics (success, num_iterations, loss and gradient norm "
        "first/last, seconds_optimization) and axial error [rad] of the MAP estimate against the "
        "ground truth (error_mean_abs/median_abs/max_abs/rmse), plus the mean absolute error of "
        "the prior mean as a baseline (prior_mean_error_mean_abs).",
        "results/samples.zarr": "MCMC chain (zarr store, one sample per row); only with mcmc "
        "settings.",
        "optimizer.log": "Iteration table of the optimizer.",
        "plots/": "ground_truth.png, map_estimate.png, difference_to_truth.png, "
        "optimization_history.png (study report).",
    }

    @override
    def input_files(self) -> list[Path]:
        raw_dir = resolve_path(self.config.raw_dir)
        return [raw_dir / name for name in ("mesh.vtu", "basis_vecs.npy", "fiber_field.npy")]

    @override
    def report(self, run_dir: Path) -> None:
        # Imported here: plotting needs pyvista and matplotlib, which are not needed to run.
        from single_runs.plots import report_map_run

        report_map_run(self.config, run_dir)

    # ----------------------------------------------------------------------------------------------
    @override
    def _execute(self, run_dir: Path, logger: BaseLogger) -> Metrics:
        config = self.config
        stage = self.stage
        if stage == "mcmc" and config.mcmc is None:
            raise ValueError("The mcmc stage needs MCMC settings in the configuration.")
        sample = stage == "mcmc" or (stage == "all" and config.mcmc is not None)
        paths = _Paths.of(run_dir, self.example_layout)
        total = {"preprocessing": 3, "map": 5, "mcmc": 6, "all": 6}[stage] + 2 * (
            stage == "all" and sample
        )
        steps = Steps(logger, total)
        metrics: Metrics = {}

        raw_dir = resolve_path(config.raw_dir)
        with steps(f"Loading the mesh and basis vectors from {raw_dir}"):
            mesh = load_pyvista_mesh(raw_dir / "mesh.vtu")
            basis_vectors = np.load(raw_dir / "basis_vecs.npy")
        logger.info(f"      mesh: {mesh.n_points} vertices, {mesh.n_cells} triangles")

        if stage in ("all", "preprocessing"):
            data = self._preprocess(mesh, basis_vectors, paths, steps, logger)
        else:
            with steps(f"Loading the preprocessing data from {paths.data_dir}"):
                data = _load_data(paths.data_dir)
        if stage == "preprocessing":
            return metrics

        posterior_builder, log_posterior = self._build_posterior(mesh, basis_vectors, data, steps)
        if stage in ("all", "map"):
            map_estimate = self._estimate_map(
                posterior_builder, log_posterior, data, paths, steps, logger, metrics
            )
        else:
            with steps(f"Loading the MAP estimate from {paths.map_dir}"):
                map_estimate = np.load(paths.map_dir / "map_estimate.npy")
        if sample:
            self._sample(
                posterior_builder, log_posterior, map_estimate, paths, steps, logger, metrics
            )
        return metrics

    # ----------------------------------------------------------------------------------------------
    def _preprocess(
        self,
        mesh: UnstructuredGrid,
        basis_vectors: np.ndarray,
        paths: _Paths,
        steps: Steps,
        logger: BaseLogger,
    ) -> _Data:
        """Build ground truth and prior mean, generate the observations, and save all four."""
        config = self.config
        strategy_name = type(config.ground_truth).__name__
        with steps(f"Building ground truth and prior mean ({strategy_name})"):
            fiber_field = np.load(resolve_path(config.raw_dir) / "fiber_field.npy")
            ground_truth = config.ground_truth.build(mesh, fiber_field, basis_vectors)
            prior_mean = build_constant_prior_mean(ground_truth)
        logger.info(f"      {describe_array('ground truth [rad]', ground_truth)}")
        logger.info(f"      prior mean angle: {prior_mean[0]:.4f} rad")

        with steps("Generating synthetic observations"):
            forward_map = EikonalParameterToSolutionMap(
                mesh, basis_vectors, config.eikonal, config.interpolation
            )
            observed_vertex_indices, observed_activation_times = generate_synthetic_observations(
                forward_map, ground_truth, config.observations
            )
        logger.info(
            f"      observed {observed_vertex_indices.shape[0]} of {mesh.n_points} vertices"
        )

        data = _Data(ground_truth, prior_mean, observed_vertex_indices, observed_activation_times)
        paths.data_dir.mkdir(parents=True, exist_ok=True)
        np.save(paths.data_dir / "ground_truth_angle_field.npy", data.ground_truth)
        np.save(paths.data_dir / "prior_mean_angle_field.npy", data.prior_mean)
        np.save(paths.data_dir / "observed_vertex_indices.npy", data.observed_vertex_indices)
        np.save(paths.data_dir / "observed_activation_times.npy", data.observed_activation_times)
        return data

    # ----------------------------------------------------------------------------------------------
    def _build_posterior(
        self, mesh: UnstructuredGrid, basis_vectors: np.ndarray, data: _Data, steps: Steps
    ) -> tuple[PosteriorBuilder, LogPosterior]:
        config = self.config
        with steps("Building the posterior"):
            posterior_builder = PosteriorBuilder(
                PosteriorSettings(
                    mesh=mesh,
                    basis_vectors=basis_vectors,
                    prior_settings=PriorSettings(
                        mean_vector=data.prior_mean,
                        kappa=config.prior.kappa,
                        tau=config.prior.tau,
                        seed=config.prior.seed,
                    ),
                    eikonal_settings=config.eikonal,
                    likelihood_settings=LikelihoodSettings(
                        num_vertices=mesh.n_points,
                        observed_vertex_indices=data.observed_vertex_indices,
                        data_vector=data.observed_activation_times,
                        noise_variance=config.observations.noise_variance,
                    ),
                ),
                interpolation_strategy=config.interpolation,
            )
            return posterior_builder, posterior_builder.build()

    # ----------------------------------------------------------------------------------------------
    def _estimate_map(
        self,
        posterior_builder: PosteriorBuilder,
        log_posterior: LogPosterior,
        data: _Data,
        paths: _Paths,
        steps: Steps,
        logger: BaseLogger,
        metrics: Metrics,
    ) -> np.ndarray:
        """Compute and save the MAP estimate; add its metrics."""
        optimizer_logger_settings = LoggerSettings(
            print_to_console=self.console, logfile_path=paths.optimizer_log
        )
        with BaseLogger(optimizer_logger_settings, prefix="map") as optimizer_logger:
            optimizer_name = type(self.config.optimizer).__name__
            with steps(f"Building the optimizer ({optimizer_name})"):
                model, optimizer = self.config.optimizer.build(
                    log_posterior, posterior_builder.prior, optimizer_logger
                )
            optimization_start = time.perf_counter()
            with steps("Running MAP estimation"):
                result = optimizer.run(initial_guess=data.prior_mean, model=model)
            optimization_seconds = time.perf_counter() - optimization_start

        outcome = "converged" if result.success else "did not converge"
        logger.info(f"      MAP estimation {outcome} after {result.num_iterations} iterations")
        logger.info(f"      status: {result.status_message}")
        logger.info(f"      loss: {result.loss_history[0]:.6e} -> {result.loss_history[-1]:.6e}")

        map_error = np.abs(compute_axial_data_diff(result.result, data.ground_truth))
        prior_mean_error = np.abs(compute_axial_data_diff(data.prior_mean, data.ground_truth))
        logger.info(f"      axial error of the MAP: mean {map_error.mean():.4f} rad")

        paths.map_dir.mkdir(parents=True, exist_ok=True)
        np.save(paths.map_dir / "map_estimate.npy", result.result)
        np.save(paths.map_dir / "map_loss_history.npy", result.loss_history)
        np.save(paths.map_dir / "map_gradient_norm_history.npy", result.gradient_norm_history)

        metrics.update(
            success=bool(result.success),
            num_iterations=int(result.num_iterations),
            status_message=str(result.status_message),
            loss_initial=float(result.loss_history[0]),
            loss_final=float(result.loss_history[-1]),
            gradient_norm_initial=float(result.gradient_norm_history[0]),
            gradient_norm_final=float(result.gradient_norm_history[-1]),
            seconds_optimization=optimization_seconds,
            error_mean_abs=float(map_error.mean()),
            error_median_abs=float(np.median(map_error)),
            error_max_abs=float(map_error.max()),
            error_rmse=float(np.sqrt(np.mean(map_error**2))),
            prior_mean_error_mean_abs=float(prior_mean_error.mean()),
        )
        return result.result

    # ----------------------------------------------------------------------------------------------
    def _sample(
        self,
        posterior_builder: PosteriorBuilder,
        log_posterior: LogPosterior,
        initial_state: np.ndarray,
        paths: _Paths,
        steps: Steps,
        logger: BaseLogger,
        metrics: Metrics,
    ) -> None:
        """Run the MCMC chain from `initial_state`, storing it as a zarr store; add its metrics."""
        settings = self.config.mcmc
        assert settings is not None
        paths.mcmc_dir.mkdir(parents=True, exist_ok=True)
        mcmc_builder = MCMCBuilder(
            MCMCSettings(
                algorithm=settings.algorithm,
                step_width=settings.step_width,
                storage=ZarrStorageSettings(chunk_size=settings.chunk_size),
            ),
            log_posterior,
            posterior_builder.prior,
            output_dir=paths.mcmc_dir,
        )
        sampler_logger_settings = LoggerSettings(
            print_to_console=self.console, logfile_path=paths.sampler_log
        )
        with BaseLogger(sampler_logger_settings, prefix="mcmc") as sampler_logger:
            with steps(f"Building the sampler ({settings.algorithm.name})"):
                sampler = mcmc_builder.build(logger=sampler_logger)
            sampling_start = time.perf_counter()
            with steps(f"Sampling {settings.num_samples} states"):
                sampler.run(
                    initial_state=initial_state,
                    settings=SamplerSettings(
                        num_samples=settings.num_samples, log_interval=settings.log_interval
                    ),
                    seed=settings.seed,
                )
            sampling_seconds = time.perf_counter() - sampling_start
        acceptance_rate = float(mcmc_builder.outputs[0].all_values[-1])
        logger.info(f"      final acceptance rate: {acceptance_rate:.3f}")
        metrics.update(mcmc_acceptance_rate=acceptance_rate, seconds_sampling=sampling_seconds)


# ==================================================================================================
def _load_data(data_dir: Path) -> _Data:
    """Read the output of the preprocessing stage."""
    return _Data(
        ground_truth=np.load(data_dir / "ground_truth_angle_field.npy"),
        prior_mean=np.load(data_dir / "prior_mean_angle_field.npy"),
        observed_vertex_indices=np.load(data_dir / "observed_vertex_indices.npy"),
        observed_activation_times=np.load(data_dir / "observed_activation_times.npy"),
    )
