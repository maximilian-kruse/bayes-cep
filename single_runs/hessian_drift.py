r"""Drift of the Gauss-Newton Hessian over a stored MCMC chain, and the cost of its products.

The Laplace approximation of the preconditioned samplers is built at one point, the MAP estimate.
How much does the Gauss-Newton Hessian $H(u)$ differ at the states of the chain, and is the
difference low-rank? The run computes the eigenpairs of $Hv = \lambda\Gamma_0^{-1}v$ at the MAP
estimate and at evenly spaced states of the chain, and compares them in the prior-whitened
coordinates, where the operator is $\hat H(u) = \Gamma_0^{1/2}H(u)\Gamma_0^{1/2} =
U\Lambda U^T$ with $U = \Gamma_0^{-1/2}V$ orthonormal. Per state it reports

- the eigenvalues of the difference $\hat H(u) - \hat H(m_\mathrm{MAP})$, which lives in the span
  of both eigenvector sets: how many directions does the Hessian change in?
- the eigenvalues of $(I + \hat H(m_\mathrm{MAP}))^{-1}(I + \hat H(u))$: the factor by which the
  precision at the state differs from the one the preconditioner assumes, per direction (1 outside
  the span);
- the cosines of the principal angles between the leading eigenspaces of both;
- the eigenvalues themselves, and the distance of the state from the MAP estimate.

It also times the Gauss-Newton products and eigenproblems of small rank at the MAP estimate, the
cost of a state-dependent preconditioner.

Classes:
    HessianDriftRunConfig: The chain, the posterior it samples and the eigenproblems.
    HessianDriftResults: The arrays the run writes.
    HessianDriftRun: The drift run.

Functions:
    compare_gauss_newton_eigenpairs: The comparison of two sets of eigenpairs.
"""

import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar, override

import numpy as np
import scipy.linalg
from ls_bayesian.common.logging import BaseLogger
from ls_bayesian.mcmc.storage import open_zarr_samples

from bayes_cep.laplace.builder import LaplaceBuilder, LaplaceSettings
from bayes_cep.laplace.hessian import GaussNewtonMisfitHessian
from bayes_cep.run.config import RunConfig
from bayes_cep.run.directories import ArrayRecord, resolve_repository_path
from bayes_cep.run.progress import StepReporter, describe_array
from bayes_cep.run.template import Metrics, Run
from single_runs.inference import InferenceProblemConfig, assemble_posterior_components

# Eigenvalues of the generalized problems below this fraction of the largest are dropped when
# the span of both eigenvector sets is orthonormalized (the sets nearly coincide if the drift is
# small).
_SPAN_TOLERANCE = 1e-10


# ==================================================================================================
@dataclass(frozen=True)
class HessianDriftRunConfig(RunConfig):
    """The chain, the posterior it samples and the eigenproblems.

    Attributes:
        problem (InferenceProblemConfig): The posterior the chain samples.
        map_estimate_path (Path): `.npy` file with the MAP estimate the approximation of the chain
            was centered at.
        laplace (LaplaceSettings): Eigensolver and rank of the eigenproblems at the MAP estimate
            and at the states.
        chain_path (Path): The Zarr store (`samples.zarr`) of a finished MCMC run of `problem`.
        burn_in (int): Number of samples discarded from the start of the chain.
        num_states (int): Number of states, evenly spaced over the chain after the burn-in.
        subspace_sizes (tuple[int, ...]): Dimensions of the leading eigenspaces compared by
            principal angles.
        num_products (int): Number of Gauss-Newton products timed at the MAP estimate.
        timing_ranks (tuple[int, ...]): Ranks of the eigenproblems timed at the MAP estimate.
    """

    problem: InferenceProblemConfig
    map_estimate_path: Path
    laplace: LaplaceSettings
    chain_path: Path
    burn_in: int
    num_states: int
    subspace_sizes: tuple[int, ...]
    num_products: int
    timing_ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.burn_in < 0 or self.num_states < 1 or self.num_products < 1:
            raise ValueError(
                "burn_in must be non-negative, num_states and num_products positive, got "
                f"{self.burn_in}, {self.num_states} and {self.num_products}."
            )


# ==================================================================================================
@dataclass(frozen=True)
class HessianDriftResults(ArrayRecord):
    r"""The arrays a drift run writes; $S$ is the number of states, $R$ the largest rank.

    Attributes:
        map_eigenvalues (np.ndarray): Eigenvalues at the MAP estimate, descending, shape `(R0,)`.
        state_indices (np.ndarray): Sample index of each state in the chain, shape `(S,)`.
        state_distances (np.ndarray): $\|u - m_\mathrm{MAP}\|_{\Gamma_0^{-1}}$ of each state,
            shape `(S,)`.
        state_eigenvalues (np.ndarray): Eigenvalues at each state, descending, NaN-padded, shape
            `(S, R)`.
        relative_drift (np.ndarray): Frobenius norm of the difference of the whitened operators
            over the one of the MAP operator, shape `(S,)`.
        drift_eigenvalues (np.ndarray): Eigenvalues of the difference, descending in absolute
            value, zero-padded, shape `(S, 2R)`.
        metric_eigenvalues (np.ndarray): Eigenvalues of $(I + \hat H_\mathrm{MAP})^{-1}(I + \hat
            H(u))$, descending, one-padded, shape `(S, 2R)`.
        subspace_cosines (np.ndarray): Smallest cosine of the principal angles between the leading
            eigenspaces of each of `subspace_sizes` dimensions, shape `(S, len(subspace_sizes))`.
        subspace_sizes (np.ndarray): These dimensions, shape `(len(subspace_sizes),)`.
        seconds_eigenproblem (np.ndarray): Wall time of the eigenproblem at each state, including
            the forward solve, shape `(S,)`.
        product_seconds (np.ndarray): Wall time of each timed Gauss-Newton product at the MAP
            estimate, shape `(num_products,)`.
        timing_ranks (np.ndarray): Ranks of the timed eigenproblems, shape `(len(timing_ranks),)`.
        timing_seconds (np.ndarray): Wall time of these at the MAP estimate, same shape.
    """

    FILE_NAMES: ClassVar[Mapping[str, str]] = {
        "map_eigenvalues": "map_eigenvalues.npy",
        "state_indices": "state_indices.npy",
        "state_distances": "state_distances.npy",
        "state_eigenvalues": "state_eigenvalues.npy",
        "relative_drift": "relative_drift.npy",
        "drift_eigenvalues": "drift_eigenvalues.npy",
        "metric_eigenvalues": "metric_eigenvalues.npy",
        "subspace_cosines": "subspace_cosines.npy",
        "subspace_sizes": "subspace_sizes.npy",
        "seconds_eigenproblem": "seconds_eigenproblem.npy",
        "product_seconds": "product_seconds.npy",
        "timing_ranks": "timing_ranks.npy",
        "timing_seconds": "timing_seconds.npy",
    }

    map_eigenvalues: np.ndarray
    state_indices: np.ndarray
    state_distances: np.ndarray
    state_eigenvalues: np.ndarray
    relative_drift: np.ndarray
    drift_eigenvalues: np.ndarray
    metric_eigenvalues: np.ndarray
    subspace_cosines: np.ndarray
    subspace_sizes: np.ndarray
    seconds_eigenproblem: np.ndarray
    product_seconds: np.ndarray
    timing_ranks: np.ndarray
    timing_seconds: np.ndarray


_FILES = HessianDriftResults.FILE_NAMES  # the names of the result files, defined once there


# ==================================================================================================
@dataclass(frozen=True)
class EigenpairComparison:
    """The comparison of two sets of eigenpairs.

    Attributes:
        relative_drift (float): $\\|\\hat H_1 - \\hat H_0\\|_F / \\|\\hat H_0\\|_F$.
        drift_eigenvalues (np.ndarray): Eigenvalues of $\\hat H_1 - \\hat H_0$, descending in
            absolute value, shape `(d,)`, $d \\le R_0 + R_1$ the dimension of the common span.
        metric_eigenvalues (np.ndarray): Eigenvalues of $(I + \\hat H_0)^{-1}(I + \\hat H_1)$ in
            the common span, descending, shape `(d,)`.
        subspace_cosines (np.ndarray): Smallest cosine of the principal angles of the leading
            eigenspaces, one per requested size.
    """

    relative_drift: float
    drift_eigenvalues: np.ndarray
    metric_eigenvalues: np.ndarray
    subspace_cosines: np.ndarray


# ==================================================================================================
def compare_gauss_newton_eigenpairs(
    eigenvalues_0: np.ndarray,
    eigenvectors_0: np.ndarray,
    precision_eigenvectors_0: np.ndarray,
    eigenvalues_1: np.ndarray,
    eigenvectors_1: np.ndarray,
    subspace_sizes: tuple[int, ...],
) -> EigenpairComparison:
    r"""Compare two low-rank operators $V\Lambda V^T$ in the coordinates where $V^T Q V = I$.

    The eigenvectors are $Q$-orthonormal, $Q = \Gamma_0^{-1}$, so in the whitened coordinates
    $Q^{1/2}x$ the operators $\hat H_i = Q^{1/2}V_i\Lambda_iV_i^TQ^{1/2}$ have orthonormal
    eigenvectors. Both live in the span of $[V_0, V_1]$, whose Gram matrix $G$ in the $Q$ inner
    product is $[[I, X], [X^T, I]]$ with $X = V_0^TQV_1$; the matrices of $\hat H_0$ and
    $\hat H_1$ in an orthonormal basis of the span follow from $G$ alone, and the product with
    $Q$ is applied only to the eigenvectors of the first set.

    Args:
        eigenvalues_0 (np.ndarray): Eigenvalues of the first operator, shape `(R0,)`.
        eigenvectors_0 (np.ndarray): Its eigenvectors, shape `(n, R0)`.
        precision_eigenvectors_0 (np.ndarray): $QV_0$, shape `(n, R0)`.
        eigenvalues_1 (np.ndarray): Eigenvalues of the second operator, shape `(R1,)`.
        eigenvectors_1 (np.ndarray): Its eigenvectors, shape `(n, R1)`.
        subspace_sizes (tuple[int, ...]): Dimensions of the leading eigenspaces whose principal
            angles are computed; clipped to the smaller rank.

    Returns:
        EigenpairComparison: Drift, metric ratio and subspace angles of the second relative to
            the first.
    """
    num_0, num_1 = eigenvalues_0.size, eigenvalues_1.size
    cross = precision_eigenvectors_0.T @ eigenvectors_1  # V0^T Q V1, shape (R0, R1)
    gram = np.block([[np.eye(num_0), cross], [cross.T, np.eye(num_1)]])
    gram_values, gram_vectors = np.linalg.eigh(gram)
    retained = gram_values > _SPAN_TOLERANCE * gram_values[-1]
    # Coordinates of Q^{1/2} V_i in the orthonormal basis of the span: sqrt(g) Z^T E_i.
    scaled = np.sqrt(gram_values[retained])[:, None] * gram_vectors[:, retained].T
    matrix_0 = (scaled[:, :num_0] * eigenvalues_0) @ scaled[:, :num_0].T
    matrix_1 = (scaled[:, num_0:] * eigenvalues_1) @ scaled[:, num_0:].T
    difference = matrix_1 - matrix_0
    drift = np.linalg.eigvalsh(difference)
    identity = np.eye(difference.shape[0])
    metric = scipy.linalg.eigh(identity + matrix_1, identity + matrix_0, eigvals_only=True)
    cosines = np.array(
        [
            np.linalg.svd(
                cross[: min(size, num_0, num_1), : min(size, num_0, num_1)], compute_uv=False
            ).min()
            for size in subspace_sizes
        ]
    )
    return EigenpairComparison(
        relative_drift=float(np.linalg.norm(difference) / np.linalg.norm(matrix_0)),
        drift_eigenvalues=drift[np.argsort(-np.abs(drift))],
        metric_eigenvalues=np.sort(metric)[::-1],
        subspace_cosines=cosines,
    )


# ==================================================================================================
def _pad(values: np.ndarray, length: int, fill: float) -> np.ndarray:
    """`values` extended by `fill` to `length`."""
    return np.concatenate([values, np.full(length - values.size, fill)])


# ==================================================================================================
class HessianDriftRun(Run[HessianDriftRunConfig]):
    """Compare the Gauss-Newton eigenpairs at states of a chain with those at the MAP estimate."""

    config_type = HessianDriftRunConfig
    name = "hessian-drift"
    outputs = {
        f"results/{_FILES['map_eigenvalues']}": "Eigenvalues at the MAP estimate.",
        f"results/{_FILES['state_indices']}": "Sample index of each state.",
        f"results/{_FILES['state_distances']}": "Prior-precision distance of each state from the "
        "MAP estimate.",
        f"results/{_FILES['state_eigenvalues']}": "Eigenvalues at each state.",
        f"results/{_FILES['relative_drift']}": "Relative Frobenius norm of the change of the "
        "whitened Gauss-Newton Hessian.",
        f"results/{_FILES['drift_eigenvalues']}": "Eigenvalues of that change.",
        f"results/{_FILES['metric_eigenvalues']}": "Eigenvalues of the precision at the state "
        "relative to the one at the MAP estimate.",
        f"results/{_FILES['subspace_cosines']}": "Smallest cosines of the principal angles of "
        "the leading eigenspaces.",
        f"results/{_FILES['subspace_sizes']}": "Dimensions of these eigenspaces.",
        f"results/{_FILES['seconds_eigenproblem']}": "Wall time of the eigenproblem per state.",
        f"results/{_FILES['product_seconds']}": "Wall time of single Gauss-Newton products.",
        f"results/{_FILES['timing_ranks']}": "Ranks of the timed eigenproblems.",
        f"results/{_FILES['timing_seconds']}": "Wall time of the eigenproblems of these ranks.",
    }

    # ----------------------------------------------------------------------------------------------
    @override
    def input_files(self) -> list[Path]:
        # The chain is a directory (a Zarr store) and not hashed.
        return [
            *self.config.problem.input_files(),
            resolve_repository_path(self.config.map_estimate_path),
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
        assembled = assemble_posterior_components(config.problem, steps, logger)
        components = assembled.components
        prior = components.prior
        map_estimate = np.load(resolve_repository_path(config.map_estimate_path))

        def solve_eigenproblem(
            state: np.ndarray, settings: LaplaceSettings
        ) -> tuple[np.ndarray, np.ndarray, float]:
            start = time.perf_counter()
            pairs = LaplaceBuilder(settings, components).build(state).eigenpairs
            return pairs.eigenvalues, pairs.eigenvectors, time.perf_counter() - start

        with steps.step(f"Timing {config.num_products} Gauss-Newton products at the MAP estimate"):
            hessian = GaussNewtonMisfitHessian(
                components.forward_map, components.likelihood, map_estimate
            )
            generator = np.random.default_rng(0)
            product_seconds = np.empty(config.num_products)
            for index in range(config.num_products):
                vector = generator.standard_normal(hessian.size)
                start = time.perf_counter()
                hessian.apply(vector)
                product_seconds[index] = time.perf_counter() - start
        logger.info(f"      {describe_array('product seconds', product_seconds)}")

        timing_seconds = np.empty(len(config.timing_ranks))
        for index, rank in enumerate(config.timing_ranks):
            with steps.step(f"Timing the eigenproblem of rank {rank} at the MAP estimate"):
                _, _, timing_seconds[index] = solve_eigenproblem(
                    map_estimate, replace(config.laplace, num_eigenvalues=rank)
                )
            logger.info(f"      {timing_seconds[index]:.2f} s")

        with steps.step("Eigenproblem at the MAP estimate"):
            values_0, vectors_0, _ = solve_eigenproblem(map_estimate, config.laplace)
        with steps.step("Applying the prior precision to its eigenvectors"):
            precision_vectors_0 = np.stack(
                [prior.apply_precision_operator(vector) for vector in vectors_0.T], axis=1
            )

        chain = open_zarr_samples(resolve_repository_path(config.chain_path))
        indices = np.linspace(config.burn_in, chain.shape[0] - 1, config.num_states).astype(int)
        rank_bound = config.laplace.num_eigenvalues
        num_states = indices.size
        distances = np.empty(num_states)
        relative_drift = np.empty(num_states)
        seconds = np.empty(num_states)
        state_eigenvalues = np.full((num_states, rank_bound), np.nan)
        drift_eigenvalues = np.zeros((num_states, 2 * rank_bound))
        metric_eigenvalues = np.ones((num_states, 2 * rank_bound))
        cosines = np.empty((num_states, len(config.subspace_sizes)))
        for position, index in enumerate(indices):
            state = np.asarray(chain[int(index)])
            offset = state - map_estimate
            distances[position] = np.sqrt(offset @ prior.apply_precision_operator(offset))
            with steps.step(f"State {position + 1} of {num_states} (sample {index})"):
                values_1, vectors_1, seconds[position] = solve_eigenproblem(state, config.laplace)
                comparison = compare_gauss_newton_eigenpairs(
                    values_0,
                    vectors_0,
                    precision_vectors_0,
                    values_1,
                    vectors_1,
                    config.subspace_sizes,
                )
            relative_drift[position] = comparison.relative_drift
            state_eigenvalues[position] = _pad(values_1, rank_bound, np.nan)
            drift_eigenvalues[position] = _pad(comparison.drift_eigenvalues, 2 * rank_bound, 0.0)
            metric_eigenvalues[position] = _pad(comparison.metric_eigenvalues, 2 * rank_bound, 1.0)
            cosines[position] = comparison.subspace_cosines
            logger.info(
                f"      relative drift {relative_drift[position]:.3f}, "
                f"distance {distances[position]:.2f}, "
                f"largest metric ratio {comparison.metric_eigenvalues[0]:.3g}, "
                f"smallest {comparison.metric_eigenvalues[-1]:.3g}"
            )

        HessianDriftResults(
            map_eigenvalues=values_0,
            state_indices=indices,
            state_distances=distances,
            state_eigenvalues=state_eigenvalues,
            relative_drift=relative_drift,
            drift_eigenvalues=drift_eigenvalues,
            metric_eigenvalues=metric_eigenvalues,
            subspace_cosines=cosines,
            subspace_sizes=np.array(config.subspace_sizes),
            seconds_eigenproblem=seconds,
            product_seconds=product_seconds,
            timing_ranks=np.array(config.timing_ranks),
            timing_seconds=timing_seconds,
        ).save(self.results_dir(run_dir))
        return {
            "num_states": int(num_states),
            "product_seconds_median": float(np.median(product_seconds)),
            "relative_drift_median": float(np.median(relative_drift)),
            "relative_drift_max": float(np.max(relative_drift)),
            "metric_ratio_max": float(metric_eigenvalues.max()),
            "metric_ratio_min": float(metric_eigenvalues.min()),
            "eigenproblem_seconds_median": float(np.median(seconds)),
        }
