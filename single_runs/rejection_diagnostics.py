r"""Where do the PMALA proposals get rejected? The acceptance by block of directions.

PMALA with a calibrated preconditioner $\overline K$ accepts only small step widths. The acceptance
probability of the full proposal is one number, and it does not say which directions are
responsible. The run answers that at states of finished chains, for several step widths $\delta$.

The eigenvectors $v_i$ of the calibrated Gaussian ($\Gamma_0^{-1}$-orthonormal, eigenvectors of
$\overline K\Gamma_0^{-1}$) split the space into blocks that are orthogonal for $\Gamma_0^{-1}$ and
$\overline K^{-1}$ alike: bands of consecutive eigenvectors (stiff first, edges in the config) and
the rest (the weaker eigenvectors and the complement of the span, where $\overline K$ is the
prior). Because $\overline K$ and the reference precision are block diagonal, PMALA restricted to
one block, the same proposal with noise and drift projected onto it and the other blocks left as
they are, is a valid Metropolis-Hastings kernel of its own with the acceptance probability of the
module [`pmala`][ls_bayesian.mcmc.algorithms.pmala] evaluated on the block. For every state, step
width and noise draw (the same noise for all blocks and step widths) the run computes

- the log acceptance ratio of the full proposal, and its split: the change $-\Delta\Phi$ of the
  potential, and the contribution of each block to the remaining, quadratic terms. A block whose
  contribution is large and negative is where the proposal is too bold for the Gaussian
  $\overline K$; $\Delta\Phi$ is what the Gaussian does not know;
- the log acceptance ratio of the proposal restricted to each block alone, and the squared jump
  $\|P_b(v-u)\|^2_{\overline K^{-1}}$ it makes in the block. The step width at which a block alone
  reaches an acceptance of about 0.57 is the one it would take if it were the only limit.

States are rows of the `pilot_states.npy` of a calibration run. Each proposal costs one potential
and gradient evaluation, a few seconds at most.

Classes:
    RejectionDiagnosticsRunConfig: The posterior, the calibration, the states and the blocks.
    RejectionDiagnosticsResults: The arrays the run writes.
    RejectionDiagnosticsRun: The diagnostic run.
"""

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, cast, override

import numpy as np
from ls_bayesian.common.logging import BaseLogger
from ls_bayesian.mcmc.algorithms.pmala import PMALAAlgorithm
from ls_bayesian.mcmc.measures import GaussianMeasure
from ls_bayesian.mcmc.model import MCMCModel

from bayes_cep.laplace.calibration import CalibratedGaussianRecord
from bayes_cep.laplace.reference_covariance import PriorReferenceCovariance
from bayes_cep.mcmc.measures import (
    CenteredGaussianMeasure,
    LikelihoodTargetMeasure,
    LowRankGaussianMeasure,
    MeanTiltedTargetMeasure,
    PriorGaussianMeasure,
)
from bayes_cep.run.config import RunConfig
from bayes_cep.run.directories import METRICS_RECORD, ArrayRecord, resolve_repository_path
from bayes_cep.run.progress import StepReporter
from bayes_cep.run.template import Metrics, Run
from single_runs.calibrate import PilotStates
from single_runs.inference import InferenceProblemConfig, assemble_posterior_components

_VectorType = np.ndarray[tuple[int], np.dtype[np.float64]]


# ==================================================================================================
@dataclass(frozen=True)
class RejectionDiagnosticsRunConfig(RunConfig):
    """The posterior, the calibration, the states and the blocks.

    Attributes:
        problem (InferenceProblemConfig): The posterior.
        calibration_path (Path): The `results` directory of a finished calibration run of this
            posterior; a relative path is relative to the repository. Its calibrated Gaussian is
            the preconditioner and its `pilot_states.npy` holds the states.
        state_rows (tuple[int, ...]): Rows of the saved pilot states to use.
        step_widths (tuple[float, ...]): The step widths $\\delta$ to test.
        band_edges (tuple[int, ...]): Indices of the eigenvectors that bound the bands, starting
            with 0, increasing, the last at most the rank of the calibrated Gaussian. The band `j`
            holds the eigenvectors from `band_edges[j]` up to `band_edges[j + 1]`; the last block
            is the rest.
        num_proposals (int): Number of noise draws per state.
        seed (int): Seed of the noise; the draws of a state also depend on its row.
    """

    problem: InferenceProblemConfig
    calibration_path: Path
    state_rows: tuple[int, ...]
    step_widths: tuple[float, ...]
    band_edges: tuple[int, ...]
    num_proposals: int
    seed: int

    def __post_init__(self) -> None:
        if len(self.state_rows) == 0 or any(row < 0 for row in self.state_rows):
            raise ValueError("state_rows must be a non-empty tuple of non-negative rows.")
        if len(self.step_widths) == 0 or any(width <= 0 for width in self.step_widths):
            raise ValueError("step_widths must be a non-empty tuple of positive widths.")
        edges = self.band_edges
        if (
            len(edges) < 2
            or edges[0] != 0
            or any(a >= b for a, b in zip(edges, edges[1:], strict=False))
        ):
            raise ValueError("band_edges must start with 0 and increase strictly.")
        if self.num_proposals < 1:
            raise ValueError(f"num_proposals must be positive, got {self.num_proposals}.")


# ==================================================================================================
@dataclass(frozen=True)
class RejectionDiagnosticsResults(ArrayRecord):
    r"""The arrays of a rejection diagnostic run; $S$ states, $D$ step widths, $R$ noise draws and
    $B$ blocks (the bands and the rest).

    Attributes:
        step_widths (np.ndarray): The step widths, shape `(D,)`.
        band_edges (np.ndarray): The band edges, shape `(B,)`.
        state_rows (np.ndarray): The rows of the pilot states, shape `(S,)`.
        full_log_ratio (np.ndarray): Log acceptance ratio of the full proposal (not capped at 0),
            shape `(S, D, R)`.
        full_delta_potential (np.ndarray): $\\Phi(v) - \\Phi(u)$ of the full proposal, same shape.
        full_block_terms (np.ndarray): Contribution of each block to the log ratio of the full
            proposal from the quadratic terms; with `-full_delta_potential` they add up to
            `full_log_ratio`; shape `(S, D, R, B)`.
        full_displacement (np.ndarray): $\\|P_b(v-u)\\|^2_{\\overline K^{-1}}$ of the full
            proposal, shape `(S, D, R, B)`.
        block_log_ratio (np.ndarray): Log acceptance ratio of the proposal restricted to each
            block, shape `(S, D, R, B)`.
        block_delta_potential (np.ndarray): $\\Phi(v_b) - \\Phi(u)$ of these, same shape.
        block_displacement (np.ndarray): The squared jump $\\|v_b-u\\|^2_{\\overline K^{-1}}$ of
            these, same shape.
    """

    FILE_NAMES: ClassVar[Mapping[str, str]] = {
        "step_widths": "step_widths.npy",
        "band_edges": "band_edges.npy",
        "state_rows": "state_rows.npy",
        "full_log_ratio": "full_log_ratio.npy",
        "full_delta_potential": "full_delta_potential.npy",
        "full_block_terms": "full_block_terms.npy",
        "full_displacement": "full_displacement.npy",
        "block_log_ratio": "block_log_ratio.npy",
        "block_delta_potential": "block_delta_potential.npy",
        "block_displacement": "block_displacement.npy",
    }

    step_widths: np.ndarray
    band_edges: np.ndarray
    state_rows: np.ndarray
    full_log_ratio: np.ndarray
    full_delta_potential: np.ndarray
    full_block_terms: np.ndarray
    full_displacement: np.ndarray
    block_log_ratio: np.ndarray
    block_delta_potential: np.ndarray
    block_displacement: np.ndarray


# ==================================================================================================
class _BlockSplit:
    r"""The split of the space into the bands of eigenvectors and the rest.

    A vector $a$ has the coefficients $c = V^T\Gamma_0^{-1}a$ in the eigenvectors, and the band $b$
    holds $V_bc_b$. The quadratic forms of $\Gamma_0^{-1}$ and $\overline K^{-1}$ on a band follow
    from the coefficients, those of the rest from the total less the bands.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        eigenvectors: np.ndarray,
        edges: tuple[int, ...],
        reference: GaussianMeasure,
        approximation: GaussianMeasure,
    ) -> None:
        rank = eigenvectors.shape[1]
        if edges[-1] > rank:
            raise ValueError(f"The last band edge {edges[-1]} exceeds the rank {rank}.")
        self._vectors = eigenvectors
        self._edges = edges
        self._reference = reference
        self._approximation = approximation
        precision_of_vectors = np.stack(
            [approximation.apply_precision_operator(eigenvectors[:, i]) for i in range(rank)],
            axis=1,
        )
        projected = eigenvectors.T @ precision_of_vectors
        self._precision_ratios = np.diag(projected).copy()
        off_diagonal = projected - np.diag(self._precision_ratios)
        self.diagonality_error = float(
            np.abs(off_diagonal).max() / np.abs(self._precision_ratios).max()
        )

    # ----------------------------------------------------------------------------------------------
    @property
    def num_blocks(self) -> int:
        """The bands and the rest."""
        return len(self._edges)

    # ----------------------------------------------------------------------------------------------
    def forms(self, vector: _VectorType) -> tuple[np.ndarray, np.ndarray]:
        r"""The forms $\langle P_ba, \overline K^{-1}P_ba\rangle$ and $\langle P_ba,
        \Gamma_0^{-1}P_ba\rangle$ of every block."""
        reference_applied = self._reference.apply_precision_operator(vector)
        coefficients = self._vectors.T @ reference_applied
        squared = coefficients**2
        approximation_forms = np.empty(self.num_blocks)
        reference_forms = np.empty(self.num_blocks)
        for block, (start, stop) in enumerate(zip(self._edges, self._edges[1:], strict=False)):
            reference_forms[block] = squared[start:stop].sum()
            approximation_forms[block] = (
                squared[start:stop] * self._precision_ratios[start:stop]
            ).sum()
        reference_forms[-1] = vector @ reference_applied - reference_forms[:-1].sum()
        approximation_forms[-1] = (
            vector @ self._approximation.apply_precision_operator(vector)
            - approximation_forms[:-1].sum()
        )
        return approximation_forms, reference_forms

    # ----------------------------------------------------------------------------------------------
    def project(self, vector: _VectorType, block: int) -> _VectorType:
        r"""The part $P_ba$ of the vector in the block."""
        coefficients = self._vectors.T @ self._reference.apply_precision_operator(vector)
        if block < len(self._edges) - 1:
            start, stop = self._edges[block], self._edges[block + 1]
            return self._vectors[:, start:stop] @ coefficients[start:stop]
        last = self._edges[-1]
        return vector - self._vectors[:, :last] @ coefficients[:last]


# ==================================================================================================
class _InstrumentedPMALA(PMALAAlgorithm):
    """PMALA with its state evaluation and transition potential made available."""

    def evaluate(self, state: _VectorType):  # noqa: ANN201 - the cache type is private
        """Potential, score and preconditioned score at `state`."""
        return self._evaluate_state_and_cache(state)

    def transition_potential(self, state: _VectorType, other: _VectorType, cache) -> float:  # noqa: ANN001
        """The $\\varrho(u, v)$ of the module docstring of the algorithm."""
        return self._evaluate_log_transition_potential(state, other, cache)


# ==================================================================================================
def _quadratic_terms(
    split: _BlockSplit,
    step_width: float,
    state: _VectorType,
    cache,  # noqa: ANN001
    other: _VectorType,
) -> np.ndarray:
    r"""The quadratic part of $\varrho(u, v)$ per block: $\frac12\langle P_bc, \overline
    K^{-1}P_bc\rangle - \frac12\langle P_bw, \Gamma_0^{-1}P_bw\rangle$ with $c = \sqrt{\delta/2}g -
    w$ and the innovation $w$ of the algorithm."""
    rho = (2 - step_width) / (2 + step_width)
    scale = np.sqrt(8 * step_width) / (2 + step_width)
    g = state - cache.preconditioned_score
    innovation = (other - rho * state) / scale
    correction = np.sqrt(step_width / 2) * g - innovation
    correction_forms, _ = split.forms(correction)
    _, innovation_forms = split.forms(innovation)
    return 0.5 * correction_forms - 0.5 * innovation_forms


# ==================================================================================================
class RejectionDiagnosticsRun(Run[RejectionDiagnosticsRunConfig]):
    """Acceptance of PMALA proposals by block of directions at states of finished chains."""

    config_type = RejectionDiagnosticsRunConfig
    name = "rejection-diagnostics"
    outputs = {
        f"results/{file_name}": f"The array {field_name}."
        for field_name, file_name in RejectionDiagnosticsResults.FILE_NAMES.items()
    } | {
        METRICS_RECORD: "Acceptance of the full proposal and of each block per step width, and "
        "the seconds per evaluation.",
    }

    # ----------------------------------------------------------------------------------------------
    @override
    def input_files(self) -> list[Path]:
        calibration_dir = resolve_repository_path(self.config.calibration_path)
        return [
            *self.config.problem.input_files(),
            *CalibratedGaussianRecord.file_paths(calibration_dir),
            *PilotStates.file_paths(calibration_dir),
        ]

    # ----------------------------------------------------------------------------------------------
    @override
    def report(self, run_dir: Path) -> None:
        """No plots: the arrays are analysed in a notebook."""

    # ----------------------------------------------------------------------------------------------
    @override
    def _execute(self, run_dir: Path, logger: BaseLogger) -> Metrics:
        config = self.config
        steps = StepReporter(logger)
        assembled = assemble_posterior_components(config.problem, steps, logger)
        prior = assembled.components.prior
        calibration_dir = resolve_repository_path(config.calibration_path)
        with steps.step("Loading the calibrated Gaussian and the states"):
            record = CalibratedGaussianRecord.load(calibration_dir)
            pilot = PilotStates.load(calibration_dir)
            gaussian = record.build_gaussian(PriorReferenceCovariance(prior), config.seed)
        if max(config.state_rows) >= pilot.states.shape[0]:
            raise ValueError(
                f"state_rows reach {max(config.state_rows)}, there are {pilot.states.shape[0]}."
            )

        target = MeanTiltedTargetMeasure(
            LikelihoodTargetMeasure(assembled.components.to_log_posterior()), prior
        )
        reference = CenteredGaussianMeasure(PriorGaussianMeasure(prior))
        approximation = LowRankGaussianMeasure(gaussian)
        algorithm_for = {
            width: _InstrumentedPMALA(
                MCMCModel(target=target, reference=reference, approximation=approximation), width
            )
            for width in config.step_widths
        }
        with steps.step("Setting up the blocks of directions"):
            split = _BlockSplit(record.eigenvectors, config.band_edges, reference, approximation)
        logger.info(
            f"      {split.num_blocks} blocks, eigenvector precision diagonal to within "
            f"{split.diagonality_error:.2e} of its largest entry"
        )

        shape = (len(config.state_rows), len(config.step_widths), config.num_proposals)
        blocks = split.num_blocks
        arrays = {
            name: np.full(shape + extra, np.nan)
            for name, extra in (
                ("full_log_ratio", ()),
                ("full_delta_potential", ()),
                ("full_block_terms", (blocks,)),
                ("full_displacement", (blocks,)),
                ("block_log_ratio", (blocks,)),
                ("block_delta_potential", (blocks,)),
                ("block_displacement", (blocks,)),
            )
        }
        decomposition_error = 0.0
        evaluations = 0
        start = time.perf_counter()
        with steps.step("Evaluating the proposals"):
            for position, row in enumerate(config.state_rows):
                state = pilot.states[row]
                cache = next(iter(algorithm_for.values())).evaluate(state)
                rng = np.random.default_rng([config.seed, int(row)])
                for proposal_index in range(config.num_proposals):
                    noise = approximation.apply_covariance_factorization(
                        rng.normal(size=approximation.random_vector_size)
                    )
                    for d, width in enumerate(config.step_widths):
                        algorithm = algorithm_for[width]
                        rho = (2 - width) / (2 + width)
                        scale = np.sqrt(8 * width) / (2 + width)
                        g = state - cache.preconditioned_score
                        increment = (rho - 1) * state + scale * (np.sqrt(width / 2) * g + noise)
                        index = (position, d, proposal_index)

                        proposal = state + increment
                        proposal_cache = algorithm.evaluate(proposal)
                        evaluations += 1
                        exponent = algorithm.transition_potential(
                            state, proposal, cache
                        ) - algorithm.transition_potential(proposal, state, proposal_cache)
                        terms = _quadratic_terms(
                            split, width, state, cache, proposal
                        ) - _quadratic_terms(split, width, proposal, proposal_cache, state)
                        delta_potential = proposal_cache.potential - cache.potential
                        decomposition_error = max(
                            decomposition_error,
                            abs(exponent - (-delta_potential + terms.sum())),
                        )
                        arrays["full_log_ratio"][index] = exponent
                        arrays["full_delta_potential"][index] = delta_potential
                        arrays["full_block_terms"][index] = terms
                        arrays["full_displacement"][index] = split.forms(increment)[0]

                        for block in range(blocks):
                            block_proposal = state + split.project(increment, block)
                            block_cache = algorithm.evaluate(block_proposal)
                            evaluations += 1
                            block_terms = _quadratic_terms(
                                split, width, state, cache, block_proposal
                            ) - _quadratic_terms(split, width, block_proposal, block_cache, state)
                            block_delta = block_cache.potential - cache.potential
                            arrays["block_log_ratio"][index + (block,)] = (
                                -block_delta + block_terms[block]
                            )
                            arrays["block_delta_potential"][index + (block,)] = block_delta
                            arrays["block_displacement"][index + (block,)] = split.forms(
                                block_proposal - state
                            )[0][block]
                logger.info(
                    f"      state {position + 1} of {len(config.state_rows)} (row {row}) done, "
                    f"{(time.perf_counter() - start) / 60:.1f} min elapsed"
                )
        seconds_per_evaluation = (time.perf_counter() - start) / evaluations

        results_dir = self.results_dir(run_dir)
        RejectionDiagnosticsResults(
            step_widths=np.array(config.step_widths),
            band_edges=np.array(config.band_edges),
            state_rows=np.array(config.state_rows),
            **arrays,
        ).save(results_dir)

        metrics: Metrics = {
            "num_blocks": blocks,
            "num_evaluations": evaluations,
            "seconds_per_evaluation": seconds_per_evaluation,
            "decomposition_error_max": decomposition_error,
            "diagonality_error": split.diagonality_error,
        }
        for d, width in enumerate(config.step_widths):
            full = np.minimum(1.0, np.exp(arrays["full_log_ratio"][:, d]))
            metrics[f"acceptance_full_{width:g}"] = float(np.mean(full))
            per_block = np.minimum(1.0, np.exp(arrays["block_log_ratio"][:, d]))
            for block in range(blocks):
                metrics[f"acceptance_block{block}_{width:g}"] = float(
                    np.mean(per_block[..., block])
                )
        return cast(Metrics, metrics)
