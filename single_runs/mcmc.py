r"""MCMC run: sample the posterior of the preprocessed data.

The run reads the data of a preprocessing run (see
[`InferenceProblemConfig`][single_runs.inference.InferenceProblemConfig]) and does nothing but the
sampling. The chain starts at the vertex array in a `.npy` file, e.g. the prior mean of the
preprocessing run or the estimate of a MAP run; the MAP run is not part of this run, only its output
file is read.

After the sampling, the chain diagnostics are computed by streaming over the Zarr store: effective
sample size and autocorrelation time of every vertex, and trace and autocorrelation of the vertices
chosen in the configuration. The report only plots these results.

Classes:
    ChainDiagnosticsSettings: Burn-in, lags and the vertices to plot traces of.
    McmcRunConfig: Which posterior to sample, and with which sampler.
    ChainDiagnostics: The arrays of the chain diagnostics.
    McmcRun: The MCMC run.
"""

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, override

import numpy as np
from ls_bayesian.common.logging import BaseLogger, LoggerSettings
from ls_bayesian.mcmc.diagnostics import MINIMUM_NUM_SAMPLES_PER_CHAIN
from ls_bayesian.mcmc.sampler import SamplerSettings
from ls_bayesian.mcmc.storage import open_zarr_samples

from bayes_cep.mcmc.builder import MCMCBuilder, MCMCSettings, ZarrStorageSettings
from bayes_cep.postprocessing.curve_plots import (
    plot_autocorrelation,
    plot_histogram,
    plot_traces,
)
from bayes_cep.postprocessing.mesh_plots import render_point_field
from bayes_cep.run.config import RunConfig
from bayes_cep.run.directories import (
    METRICS_RECORD,
    ArrayRecord,
    RunDirectory,
    resolve_repository_path,
)
from bayes_cep.run.progress import StepReporter, describe_array
from bayes_cep.run.template import Metrics, Run
from bayes_cep.statistics.axial_diagnostics import (
    compute_axial_autocorrelation,
    compute_axial_effective_sample_size,
)
from single_runs.data import RawDataDirectory
from single_runs.inference import InferenceProblemConfig, assemble_posterior

EFFECTIVE_SAMPLE_SIZE_MAP_PLOT_FILE_NAME = "effective_sample_size_map.pdf"
EFFECTIVE_SAMPLE_SIZE_HISTOGRAM_PLOT_FILE_NAME = "effective_sample_size_histogram.pdf"
TRACES_PLOT_FILE_NAME = "traces.pdf"
AUTOCORRELATION_PLOT_FILE_NAME = "autocorrelation.pdf"
SAMPLER_LOG_FILE_NAME = "sampler.log"


# ==================================================================================================
@dataclass(frozen=True)
class ChainDiagnosticsSettings:
    """Settings of the chain diagnostics.

    Attributes:
        burn_in (int): Number of leading samples discarded from all diagnostics. The traces still
            show the whole chain, with the burn-in marked.
        max_lag (int): Largest lag of the autocovariance. It must be smaller than half the number
            of samples after the burn-in (the chain is split into two halves). Vertices whose
            autocorrelation has not died out by then are counted as truncated.
        plotted_components (tuple[int, ...]): Vertices whose trace and autocorrelation function
            are plotted: none (`()`), one or several. The effective sample size is always computed
            for all vertices.
    """

    burn_in: int
    max_lag: int
    plotted_components: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.burn_in < 0:
            raise ValueError(f"burn_in must be non-negative, got {self.burn_in}.")
        if self.max_lag < 1:
            raise ValueError(f"max_lag must be at least 1, got {self.max_lag}.")
        if any(index < 0 for index in self.plotted_components):
            raise ValueError(
                f"plotted_components must be non-negative, got {self.plotted_components}."
            )
        if len(set(self.plotted_components)) != len(self.plotted_components):
            raise ValueError(f"plotted_components must be unique, got {self.plotted_components}.")


# ==================================================================================================
@dataclass(frozen=True)
class McmcRunConfig(RunConfig):
    r"""Which posterior to sample, and with which sampler.

    Attributes:
        problem (InferenceProblemConfig): The data and the model of the posterior.
        initial_state_path (Path): `.npy` file with the state the chain starts at, one angle [rad]
            per vertex (e.g. `prior_mean_angle_field.npy` of the preprocessing run, or
            `map_estimate.npy` of a MAP run of the same problem).
        sampler (MCMCSettings): Sampling algorithm, step width and sample storage.
        diagnostics (ChainDiagnosticsSettings): Burn-in, lags and plotted vertices of the chain
            diagnostics.
        num_samples (int): Number of samples of the chain.
        seed (int): Random seed of the chain.
        log_interval (int): Number of samples between two rows of the progress table.
    """

    problem: InferenceProblemConfig
    initial_state_path: Path
    sampler: MCMCSettings
    diagnostics: ChainDiagnosticsSettings
    num_samples: int
    seed: int
    log_interval: int

    def __post_init__(self) -> None:
        half_length = (self.num_samples - self.diagnostics.burn_in) // 2
        if half_length <= self.diagnostics.max_lag or half_length < MINIMUM_NUM_SAMPLES_PER_CHAIN:
            raise ValueError(
                f"After the burn-in of {self.diagnostics.burn_in}, each half of the "
                f"{self.num_samples} samples has {half_length}; it needs more than max_lag = "
                f"{self.diagnostics.max_lag} and at least {MINIMUM_NUM_SAMPLES_PER_CHAIN}."
            )


# ==================================================================================================
@dataclass(frozen=True)
class ChainDiagnostics(ArrayRecord):
    """The arrays of the chain diagnostics.

    Attributes:
        effective_sample_size (np.ndarray): Split-chain effective sample size per vertex, shape
            `(num_vertices,)`; `nan` for a constant vertex.
        integrated_autocorrelation_time (np.ndarray): Same shape, in samples.
        truncated (np.ndarray): Boolean, same shape; the autocorrelation had not died out within
            `max_lag`, so the effective sample size is an upper bound.
        component_indices (np.ndarray): The plotted vertices, shape `(num_components,)`.
        component_traces (np.ndarray): Angle samples [rad] of these vertices over the whole chain,
            shape `(num_samples, num_components)`.
        component_autocorrelation (np.ndarray): Axial autocorrelation after the burn-in, shape
            `(max_lag + 1, num_components)`.
    """

    FILE_NAMES: ClassVar[Mapping[str, str]] = {
        "effective_sample_size": "effective_sample_size.npy",
        "integrated_autocorrelation_time": "integrated_autocorrelation_time.npy",
        "truncated": "truncated.npy",
        "component_indices": "component_indices.npy",
        "component_traces": "component_traces.npy",
        "component_autocorrelation": "component_autocorrelation.npy",
    }

    effective_sample_size: np.ndarray
    integrated_autocorrelation_time: np.ndarray
    truncated: np.ndarray
    component_indices: np.ndarray
    component_traces: np.ndarray
    component_autocorrelation: np.ndarray


_FILES = ChainDiagnostics.FILE_NAMES  # the names of the result files, defined once there


# ==================================================================================================
class McmcRun(Run[McmcRunConfig]):
    """Sample the posterior of the preprocessed data, from the configured initial state."""

    config_type = McmcRunConfig
    name = "mcmc"
    outputs = {
        f"results/{ZarrStorageSettings.store_name}": "MCMC chain (zarr store, one sample per row).",
        f"results/{_FILES['effective_sample_size']}": "Split-chain effective sample size per "
        "vertex.",
        f"results/{_FILES['integrated_autocorrelation_time']}": "Autocorrelation time per vertex.",
        f"results/{_FILES['truncated']}": "Vertices whose autocorrelation was cut off at max_lag.",
        f"results/{_FILES['component_indices']}": "Vertices of the trace and autocorrelation "
        "plots.",
        f"results/{_FILES['component_traces']}": "Angle chain of these vertices.",
        f"results/{_FILES['component_autocorrelation']}": "Axial autocorrelation of these "
        "vertices.",
        METRICS_RECORD: (
            "acceptance_rate (final), seconds_sampling and summaries of the effective sample size."
        ),
        SAMPLER_LOG_FILE_NAME: "Progress table of the sampler.",
        "plots/": f"{EFFECTIVE_SAMPLE_SIZE_MAP_PLOT_FILE_NAME}, "
        f"{EFFECTIVE_SAMPLE_SIZE_HISTOGRAM_PLOT_FILE_NAME}, {TRACES_PLOT_FILE_NAME}, "
        f"{AUTOCORRELATION_PLOT_FILE_NAME} (study report; the last two only if vertices are "
        "plotted).",
    }

    # ----------------------------------------------------------------------------------------------
    @override
    def input_files(self) -> list[Path]:
        return [
            *self.config.problem.input_files(),
            resolve_repository_path(self.config.initial_state_path),
        ]

    # ----------------------------------------------------------------------------------------------
    @override
    def report(self, run_dir: Path) -> None:
        config = self.config
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        metrics = RunDirectory(run_dir).read_metrics()
        diagnostics = ChainDiagnostics.load(self.results_dir(run_dir))
        mesh = RawDataDirectory.from_configured_path(config.problem.raw_dir).load_mesh()
        render_point_field(
            mesh,
            diagnostics.effective_sample_size,
            plots_dir / EFFECTIVE_SAMPLE_SIZE_MAP_PLOT_FILE_NAME,
            f"effective sample size, minimum {metrics['effective_sample_size_min']:.3g}",
            "effective sample size",
            "viridis",
        )
        plot_histogram(
            diagnostics.effective_sample_size,
            plots_dir / EFFECTIVE_SAMPLE_SIZE_HISTOGRAM_PLOT_FILE_NAME,
            "effective sample size",
            f"{100 * metrics['fraction_truncated']:.3g}% of the vertices truncated at "
            f"max_lag = {config.diagnostics.max_lag}",
        )
        if diagnostics.component_indices.size == 0:
            return
        labels = [f"vertex {index}" for index in diagnostics.component_indices]
        plot_traces(
            diagnostics.component_traces,
            plots_dir / TRACES_PLOT_FILE_NAME,
            labels,
            config.diagnostics.burn_in,
        )
        plot_autocorrelation(
            diagnostics.component_autocorrelation,
            plots_dir / AUTOCORRELATION_PLOT_FILE_NAME,
            labels,
            "axial autocorrelation after the burn-in",
        )

    # ----------------------------------------------------------------------------------------------
    @override
    def _execute(self, run_dir: Path, logger: BaseLogger) -> Metrics:
        config = self.config
        steps = StepReporter(logger)
        assembled = assemble_posterior(config.problem, steps, logger)
        initial_state_path = resolve_repository_path(config.initial_state_path)
        initial_state = np.load(initial_state_path)
        if initial_state.shape != assembled.data.prior_mean.shape:
            raise ValueError(
                f"The initial state in {initial_state_path} has shape {initial_state.shape}, but "
                f"the problem has {assembled.data.prior_mean.shape}."
            )
        results_dir = self.results_dir(run_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        mcmc_builder = MCMCBuilder(
            config.sampler,
            assembled.log_posterior,
            output_dir=results_dir,
        )
        sampler_logger_settings = LoggerSettings(
            print_to_console=self.write_to_console, logfile_path=run_dir / SAMPLER_LOG_FILE_NAME
        )
        with BaseLogger(sampler_logger_settings, prefix="mcmc") as sampler_logger:
            with steps.step(f"Building the sampler ({config.sampler.algorithm.name})"):
                sampler = mcmc_builder.build(logger=sampler_logger)
            sampling_start = time.perf_counter()
            with steps.step(f"Sampling {config.num_samples} states"):
                sampler.run(
                    initial_state=initial_state,
                    settings=SamplerSettings(
                        num_samples=config.num_samples, log_interval=config.log_interval
                    ),
                    seed=config.seed,
                )
            sampling_seconds = time.perf_counter() - sampling_start
        acceptance_rate = float(mcmc_builder.outputs[0].all_values[-1])
        logger.info(f"      final acceptance rate: {acceptance_rate:.3f}")
        with steps.step("Computing the chain diagnostics"):
            diagnostics = self._compute_chain_diagnostics(results_dir)
            diagnostics.save(results_dir)
        effective_sample_size = diagnostics.effective_sample_size
        logger.info(f"      {describe_array('effective sample size', effective_sample_size)}")
        return {
            "acceptance_rate": acceptance_rate,
            "seconds_sampling": sampling_seconds,
            "effective_sample_size_min": float(np.nanmin(effective_sample_size)),
            "effective_sample_size_median": float(np.nanmedian(effective_sample_size)),
            "effective_sample_size_per_second_min": float(
                np.nanmin(effective_sample_size) / sampling_seconds
            ),
            "autocorrelation_time_max": float(
                np.nanmax(diagnostics.integrated_autocorrelation_time)
            ),
            "fraction_truncated": float(diagnostics.truncated.mean()),
        }

    # ----------------------------------------------------------------------------------------------
    def _compute_chain_diagnostics(self, results_dir: Path) -> ChainDiagnostics:
        """Stream over the stored chain for the effective sample size of all vertices; the
        configured vertices are read into memory for their trace and autocorrelation."""
        config = self.config
        settings = config.diagnostics
        chain = open_zarr_samples(results_dir / config.sampler.storage.store_name)
        num_samples, num_vertices = chain.shape
        if any(index >= num_vertices for index in settings.plotted_components):
            raise ValueError(
                f"plotted_components {settings.plotted_components} must be smaller than the "
                f"number of vertices {num_vertices}."
            )
        indices = np.array(settings.plotted_components, dtype=np.int64)
        result = compute_axial_effective_sample_size(chain, settings.max_lag, settings.burn_in)
        traces = np.empty((num_samples, 0))
        autocorrelation = np.empty((settings.max_lag + 1, 0))
        if indices.size > 0:
            traces = np.asarray(chain.get_orthogonal_selection((slice(None), indices)))
            autocorrelation = compute_axial_autocorrelation(
                traces, settings.max_lag, settings.burn_in
            )
        return ChainDiagnostics(
            effective_sample_size=result.effective_sample_size,
            integrated_autocorrelation_time=result.integrated_autocorrelation_time,
            truncated=result.truncated,
            component_indices=indices,
            component_traces=traces,
            component_autocorrelation=autocorrelation,
        )
