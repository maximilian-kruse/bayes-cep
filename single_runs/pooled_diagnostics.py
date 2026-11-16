r"""Convergence diagnostics and credible intervals of several finished chains of one posterior.

A single chain's effective sample size says little about whether the chains agree, and the
effective sample size of a vertex mean says little about the tails or about the coefficients that
mix slowest. The run reads the Zarr stores of finished MCMC runs of the same problem and computes,
on the draws of all chains together (after the burn-in and with every `thinning`-th sample):

- per used vertex: the rank-normalized split $\hat R$, the bulk and the tail effective sample size
  (`bayes_cep.statistics.multichain_diagnostics`, applied to the signed axial deviation from the
  pooled axial mean, a linear quantity), the pooled axial mean and variance, the equal-tailed
  credible interval and whether it contains the ground truth;
- per coefficient: the same $\hat R$ and effective sample sizes of the coefficient chains of the
  projection runs (`single_runs.chain_projection`) of these chains, if given. The coefficients
  are the projections onto the eigenvectors of the Laplace approximation, ordered by decreasing
  eigenvalue; all chains must be projected onto the same eigenvectors.

Only the used vertices (those that enter the forward model, see
`bayes_cep.mesh.interpolation.compute_used_vertex_mask`) are evaluated; the arrays have the length
of all vertices and are `nan` at the others.

Classes:
    PooledDiagnosticsRunConfig: The chains, the burn-in and the thinning.
    PooledDiagnosticsResults: The arrays the run writes.
    PooledDiagnosticsRun: The run.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, override

import numpy as np
from ls_bayesian.common.logging import BaseLogger
from ls_bayesian.mcmc.storage import open_zarr_samples

from bayes_cep.run.config import RunConfig
from bayes_cep.run.directories import ArrayRecord, resolve_repository_path
from bayes_cep.run.progress import StepReporter, describe_array
from bayes_cep.run.template import Metrics, Run
from bayes_cep.statistics.axial_error_metrics import compute_axial_error_metrics
from bayes_cep.statistics.axial_statistics import compute_axial_data_diff
from bayes_cep.statistics.multichain_diagnostics import compute_multichain_diagnostics
from bayes_cep.statistics.vertex_selection import scatter_to_vertices
from single_runs.data import PreprocessedData
from single_runs.inference import InferenceProblemConfig, load_used_vertex_mask

MINIMUM_NUM_CHAINS = 2
ESS_THRESHOLD = 400
RHAT_THRESHOLDS = (1.01, 1.05)
SLOWEST_FRACTION = 0.01
READ_BLOCK_SIZE = 200


# ==================================================================================================
@dataclass(frozen=True)
class PooledDiagnosticsRunConfig(RunConfig):
    """The chains to pool and how to read them.

    Attributes:
        problem (InferenceProblemConfig): The posterior the chains sample; gives the mesh, the
            interpolation (which vertices are used) and the ground truth.
        chain_paths (tuple[Path, ...]): The Zarr stores (`samples.zarr`) of the finished chains,
            at least two, all with the number of vertices of the mesh.
        coefficient_chain_paths (tuple[Path, ...]): The `coefficient_chain.npy` of a projection
            run of each chain, in the same order, or empty. Their row `i` need not belong to the
            sample `i * thinning` of the chain; they are diagnosed on their own.
        burn_in (int): Number of samples discarded at the start of every chain.
        thinning (int): Every `thinning`-th sample after the burn-in is used. It should stay below
            the shortest autocorrelation time of the vertices of interest.
        level (float): Level of the equal-tailed credible interval, in $(0, 1)$.
    """

    problem: InferenceProblemConfig
    chain_paths: tuple[Path, ...]
    coefficient_chain_paths: tuple[Path, ...]
    burn_in: int
    thinning: int
    level: float

    def __post_init__(self) -> None:
        if len(self.chain_paths) < MINIMUM_NUM_CHAINS:
            raise ValueError(
                f"At least {MINIMUM_NUM_CHAINS} chains are needed, got {len(self.chain_paths)}."
            )
        if self.coefficient_chain_paths and len(self.coefficient_chain_paths) != len(
            self.chain_paths
        ):
            raise ValueError(
                f"Got {len(self.chain_paths)} chains but {len(self.coefficient_chain_paths)} "
                "coefficient chains."
            )
        if self.burn_in < 0 or self.thinning < 1:
            raise ValueError(
                f"burn_in must be non-negative and thinning positive, got {self.burn_in}, "
                f"{self.thinning}."
            )
        if not 0.0 < self.level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {self.level}.")


# ==================================================================================================
@dataclass(frozen=True)
class PooledDiagnosticsResults(ArrayRecord):
    r"""The arrays the run writes; per-vertex arrays are `nan` at the unused vertices.

    Attributes:
        axial_mean (np.ndarray): Pooled axial mean angle [rad] per vertex, shape `(num_vertices,)`.
        axial_variance (np.ndarray): Pooled axial variance $-\tfrac12\ln R$ per vertex.
        rhat (np.ndarray): Rank-normalized split $\hat R$ per vertex.
        effective_sample_size_bulk (np.ndarray): Bulk effective sample size over all chains, in
            the thinned draws.
        effective_sample_size_tail (np.ndarray): Tail effective sample size, in the thinned draws.
        interval_lower (np.ndarray): Lower end [rad] of the credible interval per vertex, on the
            axial branch around the pooled mean.
        interval_upper (np.ndarray): Upper end [rad] of the interval.
        covers_truth (np.ndarray): 1.0 where the interval contains the ground truth, else 0.0.
        coefficient_rhat (np.ndarray): $\hat R$ per coefficient, shape `(num_coefficients,)`;
            empty without coefficient chains.
        coefficient_effective_sample_size_bulk (np.ndarray): Bulk effective sample size.
        coefficient_effective_sample_size_tail (np.ndarray): Tail effective sample size.
    """

    FILE_NAMES: ClassVar[Mapping[str, str]] = {
        "axial_mean": "pooled_axial_mean.npy",
        "axial_variance": "pooled_axial_variance.npy",
        "rhat": "rhat_rank.npy",
        "effective_sample_size_bulk": "effective_sample_size_bulk.npy",
        "effective_sample_size_tail": "effective_sample_size_tail.npy",
        "interval_lower": "interval_lower.npy",
        "interval_upper": "interval_upper.npy",
        "covers_truth": "covers_truth.npy",
        "coefficient_rhat": "coefficient_rhat_rank.npy",
        "coefficient_effective_sample_size_bulk": "coefficient_effective_sample_size_bulk.npy",
        "coefficient_effective_sample_size_tail": "coefficient_effective_sample_size_tail.npy",
    }

    axial_mean: np.ndarray
    axial_variance: np.ndarray
    rhat: np.ndarray
    effective_sample_size_bulk: np.ndarray
    effective_sample_size_tail: np.ndarray
    interval_lower: np.ndarray
    interval_upper: np.ndarray
    covers_truth: np.ndarray
    coefficient_rhat: np.ndarray
    coefficient_effective_sample_size_bulk: np.ndarray
    coefficient_effective_sample_size_tail: np.ndarray


_FILES = PooledDiagnosticsResults.FILE_NAMES  # the names of the result files, defined once there


# ==================================================================================================
class PooledDiagnosticsRun(Run[PooledDiagnosticsRunConfig]):
    """Pool the chains of one posterior and diagnose their agreement."""

    config_type = PooledDiagnosticsRunConfig
    name = "pooled-diagnostics"
    outputs = {
        f"results/{_FILES['axial_mean']}": "Pooled axial mean angle per vertex.",
        f"results/{_FILES['axial_variance']}": "Pooled axial variance per vertex.",
        f"results/{_FILES['rhat']}": "Rank-normalized split R-hat per vertex.",
        f"results/{_FILES['effective_sample_size_bulk']}": "Bulk effective sample size per vertex.",
        f"results/{_FILES['effective_sample_size_tail']}": "Tail effective sample size per vertex.",
        f"results/{_FILES['interval_lower']}": "Lower end of the credible interval per vertex.",
        f"results/{_FILES['interval_upper']}": "Upper end of the credible interval per vertex.",
        f"results/{_FILES['covers_truth']}": "Whether the interval contains the ground truth.",
        f"results/{_FILES['coefficient_rhat']}": "Rank-normalized split R-hat per coefficient.",
        f"results/{_FILES['coefficient_effective_sample_size_bulk']}": "Bulk effective sample "
        "size per coefficient.",
        f"results/{_FILES['coefficient_effective_sample_size_tail']}": "Tail effective sample "
        "size per coefficient.",
    }

    # ----------------------------------------------------------------------------------------------
    @override
    def input_files(self) -> list[Path]:
        # The chains are directories (Zarr stores) and not hashed.
        return [
            *self.config.problem.input_files(),
            *(resolve_repository_path(path) for path in self.config.coefficient_chain_paths),
        ]

    # ----------------------------------------------------------------------------------------------
    @override
    def report(self, run_dir: Path) -> None:
        """No plots: the arrays are analysed across the runs of a study."""

    # ----------------------------------------------------------------------------------------------
    @override
    def _execute(self, run_dir: Path, logger: BaseLogger) -> Metrics:
        config = self.config
        steps = StepReporter(logger)
        used_indices = np.flatnonzero(load_used_vertex_mask(config.problem))
        data = PreprocessedData.load(resolve_repository_path(config.problem.data_dir))
        logger.info(f"      {used_indices.size} used vertices of {data.ground_truth.size}")

        with steps.step(f"Reading {len(config.chain_paths)} chains"):
            angles = self._read_used_angles(used_indices, data.ground_truth.size, logger)
        num_chains, num_draws, _ = angles.shape
        logger.info(f"      {num_chains} chains of {num_draws} thinned draws each")

        with steps.step("Pooling the chains and computing the diagnostics"):
            results, vertex_metrics = self._diagnose_vertices(
                angles, used_indices, data.ground_truth, logger
            )
        with steps.step("Diagnosing the coefficient chains"):
            coefficient_arrays, coefficient_metrics = self._diagnose_coefficients(logger)
        PooledDiagnosticsResults(**results, **coefficient_arrays).save(self.results_dir(run_dir))
        return {
            "num_chains": num_chains,
            "num_draws_per_chain": num_draws,
            "thinning": config.thinning,
            "num_used_vertices": int(used_indices.size),
            **vertex_metrics,
            **coefficient_metrics,
        }

    # ----------------------------------------------------------------------------------------------
    def _read_used_angles(
        self, used_indices: np.ndarray, num_vertices: int, logger: BaseLogger
    ) -> np.ndarray:
        """Thinned angles of the used vertices of all chains, float32, shape `(C, N, U)`; the
        chains are cut to the length of the shortest."""
        config = self.config
        kept: list[np.ndarray] = []
        for path in config.chain_paths:
            chain = open_zarr_samples(resolve_repository_path(path))
            if chain.shape[1] != num_vertices:
                raise ValueError(f"{path} has {chain.shape[1]} vertices, expected {num_vertices}.")
            blocks = []
            for first in range(config.burn_in, chain.shape[0], READ_BLOCK_SIZE):
                last = min(first + READ_BLOCK_SIZE, chain.shape[0])
                rows = np.arange(first, last)
                rows = rows[(rows - config.burn_in) % config.thinning == 0]
                if rows.size:
                    block = np.asarray(chain[first:last])[rows - first]
                    blocks.append(block[:, used_indices].astype(np.float32))
            kept.append(np.concatenate(blocks, axis=0))
            logger.info(f"      {path}: {chain.shape[0]} samples, {kept[-1].shape[0]} kept")
        num_draws = min(draws.shape[0] for draws in kept)
        if num_draws < 8:
            raise ValueError(f"The shortest chain keeps {num_draws} draws, at least 8 are needed.")
        return np.stack([draws[:num_draws] for draws in kept])

    # ----------------------------------------------------------------------------------------------
    def _diagnose_vertices(
        self,
        angles: np.ndarray,
        used_indices: np.ndarray,
        ground_truth: np.ndarray,
        logger: BaseLogger,
    ) -> tuple[dict[str, np.ndarray], Metrics]:
        num_vertices = ground_truth.size
        resultant = np.exp(2j * angles.astype(np.float64)).mean(axis=(0, 1))
        mean = np.angle(resultant) / 2
        variance = -0.5 * np.log(np.abs(resultant))
        deviation = compute_axial_data_diff(angles.astype(np.float64), mean).astype(np.float32)
        diagnostics = compute_multichain_diagnostics(deviation)

        pooled = deviation.reshape(-1, deviation.shape[2])
        tail = (1.0 - self.config.level) / 2.0
        lower, upper = np.quantile(pooled, [tail, 1.0 - tail], axis=0)
        truth_deviation = compute_axial_data_diff(ground_truth[used_indices], mean)
        covers = ((truth_deviation >= lower) & (truth_deviation <= upper)).astype(float)

        def scatter(values: np.ndarray) -> np.ndarray:
            return scatter_to_vertices(values, used_indices, num_vertices)

        results = {
            "axial_mean": scatter(mean),
            "axial_variance": scatter(variance),
            "rhat": scatter(diagnostics.rhat),
            "effective_sample_size_bulk": scatter(diagnostics.ess_bulk),
            "effective_sample_size_tail": scatter(diagnostics.ess_tail),
            "interval_lower": scatter(mean + lower),
            "interval_upper": scatter(mean + upper),
            "covers_truth": scatter(covers),
        }
        logger.info(f"      {describe_array('rank R-hat', diagnostics.rhat)}")
        logger.info(f"      {describe_array('bulk ESS', diagnostics.ess_bulk)}")
        logger.info(f"      {describe_array('tail ESS', diagnostics.ess_tail)}")

        metrics: dict[str, float | int] = {
            "rhat_max": float(np.nanmax(diagnostics.rhat)),
            "rhat_q99": float(np.nanquantile(diagnostics.rhat, 0.99)),
            "ess_bulk_min": float(np.nanmin(diagnostics.ess_bulk)),
            "ess_bulk_median": float(np.nanmedian(diagnostics.ess_bulk)),
            "ess_tail_min": float(np.nanmin(diagnostics.ess_tail)),
            "ess_tail_median": float(np.nanmedian(diagnostics.ess_tail)),
            f"num_vertices_ess_bulk_below_{ESS_THRESHOLD}": int(
                np.sum(diagnostics.ess_bulk < ESS_THRESHOLD)
            ),
            f"num_vertices_ess_tail_below_{ESS_THRESHOLD}": int(
                np.sum(diagnostics.ess_tail < ESS_THRESHOLD)
            ),
            "interval_level": self.config.level,
            "coverage_of_truth": float(covers.mean()),
        }
        for threshold in RHAT_THRESHOLDS:
            metrics[f"fraction_rhat_above_{threshold}"] = float(
                np.mean(diagnostics.rhat > threshold)
            )
        slowest = np.argsort(np.nan_to_num(diagnostics.ess_bulk, nan=np.inf))[
            : max(1, int(SLOWEST_FRACTION * used_indices.size))
        ]
        metrics["coverage_of_truth_slowest"] = float(covers[slowest].mean())
        metrics["ess_bulk_max_of_slowest"] = float(np.nanmax(diagnostics.ess_bulk[slowest]))
        metrics["slowest_fraction"] = SLOWEST_FRACTION
        error = compute_axial_error_metrics(
            compute_axial_data_diff(mean, ground_truth[used_indices])
        )
        metrics.update({f"pooled_{name}": value for name, value in error.items()})
        return results, metrics

    # ----------------------------------------------------------------------------------------------
    def _diagnose_coefficients(self, logger: BaseLogger) -> tuple[dict[str, np.ndarray], Metrics]:
        paths = self.config.coefficient_chain_paths
        empty = np.empty(0)
        arrays = {
            "coefficient_rhat": empty,
            "coefficient_effective_sample_size_bulk": empty,
            "coefficient_effective_sample_size_tail": empty,
        }
        if not paths:
            return arrays, {}
        chains = [np.load(resolve_repository_path(path)) for path in paths]
        num_draws = min(chain.shape[0] for chain in chains)
        diagnostics = compute_multichain_diagnostics(
            np.stack([chain[:num_draws] for chain in chains]).astype(np.float64)
        )
        arrays = {
            "coefficient_rhat": diagnostics.rhat,
            "coefficient_effective_sample_size_bulk": diagnostics.ess_bulk,
            "coefficient_effective_sample_size_tail": diagnostics.ess_tail,
        }
        logger.info(f"      {describe_array('coefficient rank R-hat', diagnostics.rhat)}")
        logger.info(f"      {describe_array('coefficient bulk ESS', diagnostics.ess_bulk)}")
        metrics: Metrics = {
            "num_coefficients": int(diagnostics.rhat.size),
            "num_coefficient_draws_per_chain": int(num_draws),
            "coefficient_rhat_max": float(np.nanmax(diagnostics.rhat)),
            "coefficient_ess_bulk_min": float(np.nanmin(diagnostics.ess_bulk)),
            "coefficient_ess_tail_min": float(np.nanmin(diagnostics.ess_tail)),
            f"num_coefficients_ess_bulk_below_{ESS_THRESHOLD}": int(
                np.sum(diagnostics.ess_bulk < ESS_THRESHOLD)
            ),
            f"num_coefficients_ess_tail_below_{ESS_THRESHOLD}": int(
                np.sum(diagnostics.ess_tail < ESS_THRESHOLD)
            ),
        }
        for threshold in RHAT_THRESHOLDS:
            metrics[f"fraction_coefficients_rhat_above_{threshold}"] = float(
                np.mean(diagnostics.rhat > threshold)
            )
        return arrays, metrics
