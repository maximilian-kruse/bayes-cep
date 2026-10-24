r"""Laplace run: Gaussian approximation of the posterior at the MAP estimate.

The run reads the data of a preprocessing run (see
[`InferenceProblemConfig`][single_runs.inference.InferenceProblemConfig]) and the vertex array of
a MAP run (only its output file is read) and approximates the posterior by a low-rank Gaussian
at that estimate (see [`LaplaceBuilder`][bayes_cep.laplace.builder.LaplaceBuilder]).

The summary is comparable to the one of the MCMC run: the axial mean and variance of samples of the
approximation, and the variance of the unwrapped angle over the same samples. The diagonal of
$\Gamma$ is not computed: the prior part of it is only available as a probing estimate, which is
too noisy for the small prior variance of this problem.

Classes:
    LaplaceRunConfig: Which posterior to approximate, where, and with which eigensolver.
    LaplaceResults: The arrays a Laplace run writes.
    LaplaceRun: The Laplace run.
"""

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, override

import numpy as np
from ls_bayesian.common.logging import BaseLogger

from bayes_cep.laplace.builder import LaplaceBuilder, LaplaceSettings
from bayes_cep.postprocessing.curve_plots import plot_eigenvalue_spectrum, plot_histogram
from bayes_cep.postprocessing.mesh_plots import render_axial_angle_field_pair, render_point_field
from bayes_cep.run.config import RunConfig
from bayes_cep.run.directories import (
    METRICS_RECORD,
    PLOTS_DIR_NAME,
    ArrayRecord,
    resolve_repository_path,
)
from bayes_cep.run.progress import StepReporter, describe_array
from bayes_cep.run.template import Metrics, Run
from bayes_cep.statistics.axial_statistics import (
    compute_axial_mean_and_variance,
    compute_standard_deviation_degrees,
)
from single_runs.data import PreprocessedData, RawDataDirectory
from single_runs.inference import InferenceProblemConfig, assemble_posterior_components

MEAN_PLOT_FILE_NAME = "truth_vs_laplace_mean.pdf"
POINTWISE_STANDARD_DEVIATION_PLOT_FILE_NAME = "pointwise_standard_deviation.pdf"
AXIAL_STANDARD_DEVIATION_PLOT_FILE_NAME = "axial_standard_deviation.pdf"
SPECTRUM_PLOT_FILE_NAME = "eigenvalue_spectrum.pdf"
STD_HISTOGRAM_PLOT_FILE_NAME = "pointwise_std_histogram.pdf"


# ==================================================================================================
@dataclass(frozen=True)
class LaplaceRunConfig(RunConfig):
    """Which posterior to approximate, where, and with which eigensolver.

    Attributes:
        problem (InferenceProblemConfig): The data and the model of the posterior.
        map_estimate_path (Path): `.npy` file with the MAP estimate, one angle [rad] per vertex,
            e.g. the `map_estimate.npy` of a MAP run; a relative path is relative to the
            repository. The approximation is centered there and the Hessian evaluated there.
        laplace (LaplaceSettings): Eigensolver, rank and sample seed of the approximation.
        num_samples (int): Number of samples of the approximation for the pointwise variance and
            the axial summary.
    """

    problem: InferenceProblemConfig
    map_estimate_path: Path
    laplace: LaplaceSettings
    num_samples: int

    def __post_init__(self) -> None:
        if self.num_samples < 2:
            raise ValueError(f"num_samples must be at least 2, got {self.num_samples}.")


# ==================================================================================================
@dataclass(frozen=True)
class LaplaceResults(ArrayRecord):
    r"""The arrays a Laplace run writes.

    Attributes:
        eigenvalues (np.ndarray): Retained eigenvalues $\lambda_i$ of $Hv = \lambda\Gamma_0^{-1}v$
            in descending order, shape `(rank,)`.
        eigenvectors (np.ndarray): The $\Gamma_0^{-1}$-orthonormal eigenvectors, shape
            `(num_vertices, rank)`.
        pointwise_variance (np.ndarray): Sample variance of the angle [rad$^2$] of the samples of
            the approximation, shape `(num_vertices,)`, with a relative error of about
            $\sqrt{2/N}$ for $N$ samples. The angles are not wrapped: they are the Gaussian
            marginals, unlike `axial_variance`.
        axial_mean (np.ndarray): Axial mean angle [rad] of the samples in $(-\pi/2, \pi/2]$,
            shape `(num_vertices,)`.
        axial_variance (np.ndarray): Axial variance of the samples, shape `(num_vertices,)`; as
            `PosteriorSummary.axial_variance` of the MCMC run.
    """

    FILE_NAMES: ClassVar[Mapping[str, str]] = {
        "eigenvalues": "hessian_eigenvalues.npy",
        "eigenvectors": "hessian_eigenvectors.npy",
        "pointwise_variance": "laplace_pointwise_variance.npy",
        "axial_mean": "laplace_axial_mean.npy",
        "axial_variance": "laplace_axial_variance.npy",
    }

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    pointwise_variance: np.ndarray
    axial_mean: np.ndarray
    axial_variance: np.ndarray


_FILES = LaplaceResults.FILE_NAMES  # the names of the result files, defined once there


# ==================================================================================================
class LaplaceRun(Run[LaplaceRunConfig]):
    """Approximate the posterior of the preprocessed data by a low-rank Gaussian at the MAP."""

    config_type = LaplaceRunConfig
    name = "laplace"
    outputs = {
        f"results/{_FILES['eigenvalues']}": "Retained eigenvalues of the misfit Hessian relative "
        "to the prior precision, descending.",
        f"results/{_FILES['eigenvectors']}": "Their prior-precision-orthonormal eigenvectors, "
        "one column each.",
        f"results/{_FILES['pointwise_variance']}": "Sample variance of the angle of the "
        "approximation, per vertex.",
        f"results/{_FILES['axial_mean']}": "Axial mean angle of the samples of the "
        "approximation, per vertex.",
        f"results/{_FILES['axial_variance']}": "Axial variance of these samples, per vertex.",
        METRICS_RECORD: "Rank and extreme eigenvalues, the log-determinant ratio of covariance and "
        "prior covariance, the seconds of the approximation (forward solve and eigensolve) and "
        "the mean/max/min of the pointwise standard deviation and of the axial standard "
        "deviation, in degrees.",
        "plots/": f"{MEAN_PLOT_FILE_NAME}, {POINTWISE_STANDARD_DEVIATION_PLOT_FILE_NAME}, "
        f"{AXIAL_STANDARD_DEVIATION_PLOT_FILE_NAME}, {SPECTRUM_PLOT_FILE_NAME}, "
        f"{STD_HISTOGRAM_PLOT_FILE_NAME} (study report).",
    }

    # ----------------------------------------------------------------------------------------------
    @override
    def input_files(self) -> list[Path]:
        return [
            *self.config.problem.input_files(),
            resolve_repository_path(self.config.map_estimate_path),
        ]

    # ----------------------------------------------------------------------------------------------
    @override
    def report(self, run_dir: Path) -> None:
        problem = self.config.problem
        plots_dir = run_dir / PLOTS_DIR_NAME
        plots_dir.mkdir(exist_ok=True)
        data = PreprocessedData.load(resolve_repository_path(problem.data_dir))
        results = LaplaceResults.load(self.results_dir(run_dir))
        mesh = RawDataDirectory.from_configured_path(problem.raw_dir).load_mesh()

        render_axial_angle_field_pair(
            mesh,
            data.ground_truth,
            results.axial_mean,
            plots_dir / MEAN_PLOT_FILE_NAME,
            "ground truth",
            "laplace mean (samples)",
        )
        render_point_field(
            mesh,
            compute_standard_deviation_degrees(results.pointwise_variance),
            plots_dir / POINTWISE_STANDARD_DEVIATION_PLOT_FILE_NAME,
            "standard deviation of the angle (samples)",
            "standard deviation [deg]",
            "viridis",
        )
        render_point_field(
            mesh,
            compute_standard_deviation_degrees(results.axial_variance),
            plots_dir / AXIAL_STANDARD_DEVIATION_PLOT_FILE_NAME,
            "axial standard deviation of the samples",
            "axial standard deviation [deg]",
            "viridis",
        )
        plot_eigenvalue_spectrum(
            results.eigenvalues,
            plots_dir / SPECTRUM_PLOT_FILE_NAME,
            "eigenvalues of the misfit Hessian relative to the prior precision",
        )
        plot_histogram(
            compute_standard_deviation_degrees(results.pointwise_variance),
            plots_dir / STD_HISTOGRAM_PLOT_FILE_NAME,
            "pointwise standard deviation [deg]",
            "pointwise standard deviation of the angle (samples)",
        )

    # ----------------------------------------------------------------------------------------------
    @override
    def _execute(self, run_dir: Path, logger: BaseLogger) -> Metrics:
        config = self.config
        steps = StepReporter(logger)
        assembled = assemble_posterior_components(config.problem, steps, logger)
        map_estimate = np.load(resolve_repository_path(config.map_estimate_path))
        logger.info(f"      {describe_array('MAP estimate [rad]', map_estimate)}")

        approximation_start = time.perf_counter()
        with steps.step(
            f"Computing {config.laplace.num_eigenvalues} eigenpairs of the misfit Hessian "
            f"({type(config.laplace.eigensolver).__name__})"
        ):
            approximation = LaplaceBuilder(config.laplace, assembled.components).build(map_estimate)
        approximation_seconds = time.perf_counter() - approximation_start
        eigenpairs = approximation.eigenpairs
        logger.info(f"      {describe_array('eigenvalues', eigenpairs.eigenvalues)}")
        logger.info(
            f"      kept {eigenpairs.eigenvalues.size} of {config.laplace.num_eigenvalues} "
            f"eigenpairs; smallest kept eigenvalue {eigenpairs.eigenvalues[-1]:.3e} "
            "(directions beyond it are not corrected)"
        )

        with steps.step(f"Drawing {config.num_samples} samples"):
            samples = np.stack(
                [approximation.gaussian.generate_sample() for _ in range(config.num_samples)]
            )
            pointwise_variance = samples.var(axis=0, ddof=1)
            axial_mean, axial_variance = compute_axial_mean_and_variance(samples)
        pointwise_std_degrees = compute_standard_deviation_degrees(pointwise_variance)
        axial_std_degrees = compute_standard_deviation_degrees(axial_variance)
        logger.info(f"      {describe_array('pointwise std [deg]', pointwise_std_degrees)}")
        logger.info(f"      {describe_array('axial std [deg]', axial_std_degrees)}")

        LaplaceResults(
            eigenvalues=eigenpairs.eigenvalues,
            eigenvectors=eigenpairs.eigenvectors,
            pointwise_variance=pointwise_variance,
            axial_mean=axial_mean,
            axial_variance=axial_variance,
        ).save(self.results_dir(run_dir))
        return {
            "num_eigenpairs": int(eigenpairs.eigenvalues.size),
            "eigenvalue_max": float(eigenpairs.eigenvalues[0]),
            "eigenvalue_min": float(eigenpairs.eigenvalues[-1]),
            "log_determinant_ratio": approximation.gaussian.compute_log_determinant_ratio(),
            "seconds_approximation": approximation_seconds,
            "pointwise_std_mean_deg": float(np.mean(pointwise_std_degrees)),
            "pointwise_std_min_deg": float(np.min(pointwise_std_degrees)),
            "pointwise_std_max_deg": float(np.max(pointwise_std_degrees)),
            "axial_std_mean_deg": float(np.mean(axial_std_degrees)),
            "axial_std_max_deg": float(np.max(axial_std_degrees)),
        }
