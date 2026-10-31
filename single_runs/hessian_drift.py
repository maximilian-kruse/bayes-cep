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

Is the posterior far from the Gaussian at the MAP estimate, and why? The chain mean is computed
over the whole chain after the burn-in and compared with the Laplace approximation along the
leading eigenvectors $v_i$ ($\delta = c\,v_i$, Laplace standard deviation $(1+\lambda_i)^{-1/2}$):

- the coefficients $c_i = v_i^T\Gamma_0^{-1}(\bar u - m_\mathrm{MAP})$ of the chain mean;
- the negative log posterior $J$ along the line from the MAP estimate through the chain mean and
  along the $v_i$, with the second-order Taylor model of $J$ at the MAP estimate that uses the
  Gauss-Newton Hessian, $J_0 + g^T\delta + \frac12\delta^T(\Gamma_0^{-1} + H)\delta$;
- the curvature $v_i^T\nabla^2 J\,v_i$ from finite differences of the gradient at the MAP estimate
  and at the chain mean, for several step sizes, against the Gauss-Newton value $1 + \lambda_i$ at
  the MAP estimate and at the chain mean. A finite-difference curvature below the Gauss-Newton one
  means the omitted term of the Hessian makes the true Gaussian approximation wider; a profile that
  departs from the model far from the MAP estimate means the potential is not quadratic there.

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
from ls_bayesian.mcmc.storage import SampleStore, iterate_sample_blocks, open_zarr_samples

from bayes_cep.laplace.builder import LaplaceBuilder, LaplaceSettings
from bayes_cep.laplace.hessian import GaussNewtonMisfitHessian
from bayes_cep.posterior.builder import PosteriorComponents
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
    r"""The chain, the posterior it samples and the eigenproblems.

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
        directions (tuple[int, ...]): Indices of the eigenvectors at the MAP estimate along which
            the potential is profiled and its curvature measured; indices beyond the rank are
            skipped.
        profile_extent (float): The profiles along the eigenvectors cover the coefficients $\pm$
            this many Laplace standard deviations.
        profile_points (int): Number of points of each profile.
        line_range (tuple[float, float]): Parameter range $t$ of the line $m_\mathrm{MAP} +
            t(\bar u - m_\mathrm{MAP})$; 0 is the MAP estimate, 1 the chain mean.
        curvature_steps (tuple[float, ...]): Finite-difference steps, in Laplace standard
            deviations of the direction; several to tell the quadratic from the noise of the
            single-precision forward solve.
        mean_block_size (int): Number of samples read at a time when averaging the chain.
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
    directions: tuple[int, ...]
    profile_extent: float
    profile_points: int
    line_range: tuple[float, float]
    curvature_steps: tuple[float, ...]
    mean_block_size: int

    def __post_init__(self) -> None:
        if self.burn_in < 0 or self.num_states < 1 or self.num_products < 1:
            raise ValueError(
                "burn_in must be non-negative, num_states and num_products positive, got "
                f"{self.burn_in}, {self.num_states} and {self.num_products}."
            )
        if self.profile_points < 2 or self.profile_extent <= 0 or self.mean_block_size < 1:
            raise ValueError(
                "profile_points must be at least 2, profile_extent and mean_block_size "
                f"positive, got {self.profile_points}, {self.profile_extent} and "
                f"{self.mean_block_size}."
            )
        if any(step <= 0 for step in self.curvature_steps) or any(
            index < 0 for index in self.directions
        ):
            raise ValueError("curvature_steps must be positive and directions non-negative.")


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
        mean_coefficients (np.ndarray): Coefficients $c_i = v_i^T\Gamma_0^{-1}(\bar u -
            m_\mathrm{MAP})$ of the chain mean on all MAP eigenvectors, shape `(R0,)`; in units of
            the Laplace standard deviation $(1+\lambda_i)^{-1/2}$ they are $c_i\sqrt{1+\lambda_i}$.
        mean_offset_norms (np.ndarray): $\|\bar u - m_\mathrm{MAP}\|_{\Gamma_0^{-1}}$ and
            $\sqrt{\sum_i c_i^2}$ (the part inside the retained subspace), shape `(2,)`.
        directions (np.ndarray): The profiled eigenvector indices (those below the rank), shape
            `(D,)`.
        line_parameters (np.ndarray): Parameters $t$ of the line through the MAP estimate and the
            chain mean, shape `(L,)`.
        line_potential (np.ndarray): $J$ along the line, shape `(L,)`.
        line_model (np.ndarray): The Gauss-Newton second-order model of $J$ at the MAP estimate
            along the line, shape `(L,)`.
        profile_scales (np.ndarray): Coefficients of the profiles in Laplace standard deviations,
            shape `(P,)`.
        profile_potential (np.ndarray): $J$ along each direction, shape `(D, P)`.
        profile_model (np.ndarray): The Gauss-Newton model along each direction, shape `(D, P)`.
        curvature_steps (np.ndarray): Finite-difference steps in Laplace standard deviations,
            shape `(H,)`.
        curvature_map (np.ndarray): Finite-difference curvature $v_i^T\nabla^2J\,v_i$ at the MAP
            estimate, shape `(D, H)`.
        curvature_mean (np.ndarray): The same at the chain mean, shape `(D, H)`.
        gauss_newton_curvature (np.ndarray): $1 + \lambda_i$ at the MAP estimate and
            $v_i^T(\Gamma_0^{-1} + H(\bar u))v_i$ at the chain mean, shape `(2, D)`.
        potential_values (np.ndarray): $J$ at the MAP estimate and at the chain mean, shape `(2,)`.
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
        "mean_coefficients": "mean_coefficients.npy",
        "mean_offset_norms": "mean_offset_norms.npy",
        "directions": "directions.npy",
        "line_parameters": "line_parameters.npy",
        "line_potential": "line_potential.npy",
        "line_model": "line_model.npy",
        "profile_scales": "profile_scales.npy",
        "profile_potential": "profile_potential.npy",
        "profile_model": "profile_model.npy",
        "curvature_steps": "curvature_steps.npy",
        "curvature_map": "curvature_map.npy",
        "curvature_mean": "curvature_mean.npy",
        "gauss_newton_curvature": "gauss_newton_curvature.npy",
        "potential_values": "potential_values.npy",
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
    mean_coefficients: np.ndarray
    mean_offset_norms: np.ndarray
    directions: np.ndarray
    line_parameters: np.ndarray
    line_potential: np.ndarray
    line_model: np.ndarray
    profile_scales: np.ndarray
    profile_potential: np.ndarray
    profile_model: np.ndarray
    curvature_steps: np.ndarray
    curvature_map: np.ndarray
    curvature_mean: np.ndarray
    gauss_newton_curvature: np.ndarray
    potential_values: np.ndarray


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
def _measure_non_quadratic_structure(
    config: HessianDriftRunConfig,
    components: PosteriorComponents,
    map_estimate: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    chain: SampleStore,
    steps: StepReporter,
    logger: BaseLogger,
) -> dict[str, np.ndarray]:
    """The chain mean, and the potential along the line to it and along the leading eigenvectors,
    against the Gauss-Newton model at the MAP estimate, and the finite-difference curvature along
    the eigenvectors at the MAP estimate and the chain mean (see the module docstring).

    Returns:
        dict[str, np.ndarray]: The arrays of `HessianDriftResults` from `mean_coefficients` on.
    """
    prior = components.prior
    log_posterior = components.to_log_posterior()

    with steps.step("Averaging the chain after the burn-in"):
        total = np.zeros(map_estimate.size)
        count = 0
        for block in iterate_sample_blocks(chain, config.burn_in, config.mean_block_size):
            total += block.sum(axis=0)
            count += block.shape[0]
        chain_mean = total / count
    offset = chain_mean - map_estimate
    precision_offset = prior.apply_precision_operator(offset)
    coefficients = eigenvectors.T @ precision_offset
    norms = np.array([np.sqrt(offset @ precision_offset), np.linalg.norm(coefficients)])
    logger.info(
        f"      offset of the chain mean from the MAP estimate: {norms[0]:.1f} in the prior "
        f"precision norm, {norms[1]:.1f} of it in the retained subspace"
    )

    directions = np.array([index for index in config.directions if index < eigenvalues.size])
    deviations = 1.0 / np.sqrt(1.0 + eigenvalues[directions])  # Laplace standard deviations

    with steps.step("Potential and gradient at the MAP estimate, potential at the chain mean"):
        potential_map = log_posterior.evaluate_cost(map_estimate)
        gradient_map = log_posterior.evaluate_gradient(map_estimate)
        potential_mean = log_posterior.evaluate_cost(chain_mean)
        hessian_map = GaussNewtonMisfitHessian(
            components.forward_map, components.likelihood, map_estimate
        )
        hessian_mean = GaussNewtonMisfitHessian(
            components.forward_map, components.likelihood, chain_mean
        )
    logger.info(
        f"      potential at the MAP estimate {potential_map:.6g}, at the chain mean "
        f"{potential_mean:.6g}, gradient norm at the MAP estimate "
        f"{np.linalg.norm(gradient_map):.3g}"
    )

    with steps.step("Potential along the line from the MAP estimate through the chain mean"):
        line_parameters = np.linspace(*config.line_range, config.profile_points)
        quadratic_form = offset @ precision_offset + offset @ hessian_map.apply(offset)
        linear_term = gradient_map @ offset
        line_potential = np.array(
            [log_posterior.evaluate_cost(map_estimate + t * offset) for t in line_parameters]
        )
        line_model = (
            potential_map
            + line_parameters * linear_term
            + 0.5 * line_parameters**2 * quadratic_form
        )

    scales = np.linspace(-config.profile_extent, config.profile_extent, config.profile_points)
    profile_potential = np.empty((directions.size, scales.size))
    profile_model = np.empty_like(profile_potential)
    with steps.step(f"Potential along {directions.size} eigenvectors"):
        for row, (index, deviation) in enumerate(zip(directions, deviations, strict=True)):
            vector = eigenvectors[:, index]
            coefficient = scales * deviation
            profile_potential[row] = [
                log_posterior.evaluate_cost(map_estimate + c * vector) for c in coefficient
            ]
            profile_model[row] = (
                potential_map
                + coefficient * (gradient_map @ vector)
                + 0.5 * coefficient**2 * (1.0 + eigenvalues[index])
            )

    def finite_difference_curvature(base: np.ndarray) -> np.ndarray:
        """Curvature along each eigenvector from central differences of the gradient, `(D, H)`."""
        curvature = np.empty((directions.size, len(config.curvature_steps)))
        for row, (index, deviation) in enumerate(zip(directions, deviations, strict=True)):
            vector = eigenvectors[:, index]
            for column, step in enumerate(config.curvature_steps):
                width = step * deviation
                difference = log_posterior.evaluate_gradient(
                    base + width * vector
                ) - log_posterior.evaluate_gradient(base - width * vector)
                curvature[row, column] = vector @ difference / (2.0 * width)
        return curvature

    with steps.step("Finite-difference curvature at the MAP estimate"):
        curvature_map = finite_difference_curvature(map_estimate)
    with steps.step("Finite-difference curvature at the chain mean"):
        curvature_mean = finite_difference_curvature(chain_mean)
    # The eigenvectors are Gamma_0^{-1}-orthonormal: v^T Gamma_0^{-1} v = 1.
    gauss_newton_curvature = np.stack(
        [
            1.0 + eigenvalues[directions],
            [
                1.0 + eigenvectors[:, index] @ hessian_mean.apply(eigenvectors[:, index])
                for index in directions
            ],
        ]
    )
    return {
        "mean_coefficients": coefficients,
        "mean_offset_norms": norms,
        "directions": directions,
        "line_parameters": line_parameters,
        "line_potential": line_potential,
        "line_model": line_model,
        "profile_scales": scales,
        "profile_potential": profile_potential,
        "profile_model": profile_model,
        "curvature_steps": np.array(config.curvature_steps),
        "curvature_map": curvature_map,
        "curvature_mean": curvature_mean,
        "gauss_newton_curvature": gauss_newton_curvature,
        "potential_values": np.array([potential_map, potential_mean]),
    }


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
        f"results/{_FILES['mean_coefficients']}": "Coefficients of the chain mean on the MAP "
        "eigenvectors.",
        f"results/{_FILES['mean_offset_norms']}": "Prior-precision norm of the offset of the "
        "chain mean and of its part in the retained subspace.",
        f"results/{_FILES['directions']}": "Eigenvector indices of the profiles and curvatures.",
        f"results/{_FILES['line_parameters']}": "Parameters of the line through the MAP estimate "
        "and the chain mean.",
        f"results/{_FILES['line_potential']}": "Negative log posterior along that line.",
        f"results/{_FILES['line_model']}": "Gauss-Newton model of it from the MAP estimate.",
        f"results/{_FILES['profile_scales']}": "Coefficients of the profiles in Laplace standard "
        "deviations.",
        f"results/{_FILES['profile_potential']}": "Negative log posterior along the eigenvectors.",
        f"results/{_FILES['profile_model']}": "Gauss-Newton model of it from the MAP estimate.",
        f"results/{_FILES['curvature_steps']}": "Finite-difference steps of the curvatures.",
        f"results/{_FILES['curvature_map']}": "Finite-difference curvature along the "
        "eigenvectors at the MAP estimate.",
        f"results/{_FILES['curvature_mean']}": "The same at the chain mean.",
        f"results/{_FILES['gauss_newton_curvature']}": "Gauss-Newton curvature along the "
        "eigenvectors at the MAP estimate and at the chain mean.",
        f"results/{_FILES['potential_values']}": "Negative log posterior at the MAP estimate and "
        "at the chain mean.",
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
        non_quadratic = _measure_non_quadratic_structure(
            config, components, map_estimate, values_0, vectors_0, chain, steps, logger
        )
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
            **non_quadratic,
        ).save(self.results_dir(run_dir))
        return {
            "num_states": int(num_states),
            "product_seconds_median": float(np.median(product_seconds)),
            "relative_drift_median": float(np.median(relative_drift)),
            "relative_drift_max": float(np.max(relative_drift)),
            "metric_ratio_max": float(metric_eigenvalues.max()),
            "metric_ratio_min": float(metric_eigenvalues.min()),
            "eigenproblem_seconds_median": float(np.median(seconds)),
            "mean_offset_norm": float(non_quadratic["mean_offset_norms"][0]),
            "mean_offset_norm_in_subspace": float(non_quadratic["mean_offset_norms"][1]),
            "potential_increase_to_mean": float(
                non_quadratic["potential_values"][1] - non_quadratic["potential_values"][0]
            ),
            "potential_increase_to_mean_model": float(
                non_quadratic["line_model"][
                    np.argmin(np.abs(non_quadratic["line_parameters"] - 1.0))
                ]
                - non_quadratic["potential_values"][0]
            ),
        }
