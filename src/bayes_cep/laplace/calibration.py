r"""Calibration of a Laplace approximation by samples of the posterior.

The Laplace approximation $\mathcal N(m_\mathrm{MAP}, \Gamma)$, $\Gamma = \Gamma_0 - V D V^T$,
describes the posterior by the Gauss-Newton Hessian at one point. For a nonlinear forward map the
posterior can be much wider than that in the stiff directions and coupled across them, and a
sampler preconditioned by $\Gamma$ then crawls along them. The remedy here is a frozen,
sample-informed preconditioner in the spirit of DILI (Cui, Law, Marzouk, 2016): the covariance in
the span of the leading eigenvectors is replaced by the empirical covariance of a pilot chain,
the prior covariance is kept in the complement, and the result is held fixed for the final run.
Any fixed Gaussian with a covariance that differs from the prior one by a finite-rank operator is
a valid preconditioner, so the final chain still targets the posterior.

With the $\Gamma_0^{-1}$-orthonormal eigenvectors $V$ of $Hv = \lambda\Gamma_0^{-1}v$, a state
splits as $u - m = Vc + w$ with the coefficients $c = V^T\Gamma_0^{-1}(u - m)$ and a remainder $w$
orthogonal to $V$ in the $\Gamma_0^{-1}$ inner product. Under the prior $c\sim\mathcal N(0, I)$ is
independent of $w$, and under the Laplace approximation $c\sim\mathcal N(0, \operatorname{diag}(1 /
(1+\lambda_i)))$. The calibrated Gaussian has $c\sim\mathcal N(0, \Sigma)$ for a covariance
$\Sigma$ estimated from the chain, and the prior covariance for $w$: $\Gamma = \Gamma_0 + V(\Sigma -
I)V^T$. With $\Sigma = R\operatorname{diag}(s)R^T$ this is again a `LowRankGaussian` with the
eigenvectors $VR$ and the eigenvalues $\tilde\lambda_j = 1/s_j - 1$, which must not be negative,
so a variance larger than the prior one cannot be represented and is cut off.

The mean of the Gaussian is irrelevant for `PMALA` (it cancels from the drift) and sets the
proposal of `PCN_PRECONDITIONED`; `CalibrationSettings.recenter` moves it to the chain mean.

Classes:
    CalibrationSettings: How the moments are estimated and regularized.
    SubspaceMoments: Chain mean and the mean and covariance of the leading coefficients.
    CalibratedApproximation: The calibrated approximation and what was done to the covariance.
    CalibratedGaussianRecord: The calibrated Gaussian as arrays, to be stored and rebuilt elsewhere.

Functions:
    estimate_subspace_moments: Stream over pilot chains for the moments.
    calibrate_approximation: The Laplace approximation with the empirical covariance in the
        leading subspace.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from ls_bayesian.lowrank.eigensolvers import GeneralizedEigenpairs
from ls_bayesian.lowrank.gaussian import LowRankGaussian
from ls_bayesian.lowrank.interfaces import ReferenceCovariance
from ls_bayesian.mcmc.storage import SampleStore

from bayes_cep.laplace.builder import LaplaceApproximation
from bayes_cep.run.directories import ArrayRecord

# Smallest variance of the calibrated covariance in units of the smallest Laplace variance of the
# calibrated coefficients, so that a degenerate empirical covariance cannot give an unbounded
# eigenvalue.
_MINIMUM_VARIANCE_FRACTION = 1e-6
# Largest variance of the calibrated covariance: the prior variance, 1, less a margin, as the
# eigenvalues of the low-rank Gaussian must be non-negative.
_MAXIMUM_VARIANCE = 1.0 - 1e-9


# ==================================================================================================
@dataclass(frozen=True)
class CalibrationSettings:
    r"""How the moments of the pilot chains are estimated and regularized.

    There are no defaults: the effective sample size of the stiffest coefficients of a pilot chain
    is small, which makes the empirical covariance noisy and, for the slowest directions, biased
    towards too small variances, so the values have to be chosen for the problem.

    Attributes:
        num_coefficients (int): Number $k$ of leading eigenvectors whose covariance is replaced; at
            most the rank of the approximation. The stiffest directions are the slowest ones.
        burn_in (int): Number of samples discarded from the start of each pilot chain.
        thinning (int): Every `thinning`-th sample after the burn-in is used.
        block_size (int): Number of samples read from a chain at a time (before thinning).
        coupling_shrinkage (float): Weight $\alpha\in[0, 1]$ of the diagonal in
            $\Sigma\leftarrow (1-\alpha)\Sigma + \alpha\operatorname{diag}\Sigma$; 0 keeps the
            empirical coupling between the coefficients, 1 keeps only the variances.
        variance_inflation (float): Factor $\ge 0$ multiplying the covariance after the shrinkage,
            e.g. above 1 against the bias of a short pilot chain.
        minimum_variance_ratio (float): Lower bound of the covariance in units of the Laplace
            covariance, $\Sigma\succeq r\operatorname{diag}(1/(1+\lambda_i))$ (in the sense of
            quadratic forms); 0 turns it off and 1 never lets the calibration narrow the
            approximation.
        recenter (bool): Whether the mean of the Gaussian is the mean of the pilot chains instead
            of the MAP estimate. It matters for the preconditioned pCN, not for PMALA.
    """

    num_coefficients: int
    burn_in: int
    thinning: int
    block_size: int
    coupling_shrinkage: float
    variance_inflation: float
    minimum_variance_ratio: float
    recenter: bool

    def __post_init__(self) -> None:
        if self.num_coefficients < 1 or self.thinning < 1 or self.block_size < 1:
            raise ValueError(
                "num_coefficients, thinning and block_size must be positive, got "
                f"{self.num_coefficients}, {self.thinning} and {self.block_size}."
            )
        if self.burn_in < 0:
            raise ValueError(f"burn_in must be non-negative, got {self.burn_in}.")
        if not 0.0 <= self.coupling_shrinkage <= 1.0:
            raise ValueError(
                f"coupling_shrinkage must be in [0, 1], got {self.coupling_shrinkage}."
            )
        if self.variance_inflation <= 0.0 or self.minimum_variance_ratio < 0.0:
            raise ValueError(
                "variance_inflation must be positive and minimum_variance_ratio non-negative, got "
                f"{self.variance_inflation} and {self.minimum_variance_ratio}."
            )


# ==================================================================================================
@dataclass(frozen=True)
class SubspaceMoments(ArrayRecord):
    r"""The moments of the pilot chains.

    Attributes:
        mean (np.ndarray): Mean of the states, shape `(num_vertices,)`.
        coefficient_mean (np.ndarray): Mean of the coefficients $c_i = v_i^T\Gamma_0^{-1}(u -
            m_\mathrm{ref})$ of the leading eigenvectors, shape `(k,)`.
        coefficient_covariance (np.ndarray): Their sample covariance, shape `(k, k)`.
        num_samples (np.ndarray): Number of samples used, shape `(1,)`.
    """

    FILE_NAMES: ClassVar[Mapping[str, str]] = {
        "mean": "pilot_mean.npy",
        "coefficient_mean": "pilot_coefficient_mean.npy",
        "coefficient_covariance": "pilot_coefficient_covariance.npy",
        "num_samples": "pilot_num_samples.npy",
    }

    mean: np.ndarray
    coefficient_mean: np.ndarray
    coefficient_covariance: np.ndarray
    num_samples: np.ndarray


# ==================================================================================================
@dataclass(frozen=True)
class CalibratedApproximation:
    r"""The calibrated approximation and what was done to its covariance.

    Attributes:
        approximation (LaplaceApproximation): The Gaussian and its eigenpairs, descending in the
            eigenvalues; the leading $k$ eigenvectors are rotated into the principal axes of the
            calibrated covariance.
        laplace_variances (np.ndarray): $1/(1+\lambda_i)$ of the $k$ calibrated coefficients,
            shape `(k,)`.
        empirical_variance_ratios (np.ndarray): Variance of each coefficient in the pilot chains
            over its Laplace variance, shape `(k,)`.
        calibrated_covariance (np.ndarray): The covariance $\Sigma$ of the coefficients after the
            regularization, shape `(k, k)`.
        calibrated_variance_ratios (np.ndarray): Its diagonal over the Laplace variances, shape
            `(k,)`.
        num_clipped (int): Number of principal variances cut off at the prior variance.
    """

    approximation: LaplaceApproximation
    laplace_variances: np.ndarray
    empirical_variance_ratios: np.ndarray
    calibrated_covariance: np.ndarray
    calibrated_variance_ratios: np.ndarray
    num_clipped: int


# ==================================================================================================
@dataclass(frozen=True)
class CalibratedGaussianRecord(ArrayRecord):
    """The calibrated Gaussian as arrays: its eigenpairs and mean.

    The moments of the pilot chains refer to the eigenvectors they were computed with, which a
    randomized eigensolver does not reproduce across machines (signs, and rotations within nearly
    degenerate eigenvalues), so the Gaussian is stored with its own eigenvectors and rebuilt from
    them.

    Attributes:
        eigenvalues (np.ndarray): Eigenvalues of the calibrated approximation, descending, shape
            `(R,)`.
        eigenvectors (np.ndarray): Their $\\Gamma_0^{-1}$-orthonormal eigenvectors, shape
            `(num_vertices, R)`.
        mean (np.ndarray): Mean of the Gaussian, shape `(num_vertices,)`.
    """

    FILE_NAMES: ClassVar[Mapping[str, str]] = {
        "eigenvalues": "calibrated_gaussian_eigenvalues.npy",
        "eigenvectors": "calibrated_gaussian_eigenvectors.npy",
        "mean": "calibrated_gaussian_mean.npy",
    }

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    mean: np.ndarray

    # ----------------------------------------------------------------------------------------------
    @classmethod
    def from_approximation(cls, approximation: LaplaceApproximation) -> CalibratedGaussianRecord:
        """The record of an approximation, e.g. `CalibratedApproximation.approximation`."""
        return cls(
            eigenvalues=approximation.eigenpairs.eigenvalues,
            eigenvectors=approximation.eigenpairs.eigenvectors,
            mean=approximation.gaussian.mean,
        )

    # ----------------------------------------------------------------------------------------------
    def build_gaussian(self, reference: ReferenceCovariance, seed: int) -> LowRankGaussian:
        """The Gaussian of the record; `reference` is the prior covariance it was built with."""
        return LowRankGaussian(
            reference,
            GeneralizedEigenpairs(self.eigenvalues, self.eigenvectors),
            mean=self.mean,
            seed=seed,
        )


# ==================================================================================================
def estimate_subspace_moments(
    chains: Sequence[SampleStore],
    weighted_eigenvectors: np.ndarray,
    reference_state: np.ndarray,
    settings: CalibrationSettings,
) -> SubspaceMoments:
    r"""Stream over the pilot chains for the mean of the states and the mean and covariance of the
    coefficients of the leading eigenvectors.

    The chains are pooled: the covariance contains the variation between them. Blocks are merged
    with the parallel update of Chan, Golub and LeVeque, so nothing but one block of states is held
    in memory.

    Args:
        chains (Sequence[SampleStore]): Chains of the same posterior, each of shape
            `(num_samples, num_vertices)`.
        weighted_eigenvectors (np.ndarray): $\Gamma_0^{-1}V_k$, shape `(num_vertices, k)`.
        reference_state (np.ndarray): State the coefficients are measured from, shape
            `(num_vertices,)`; the covariance does not depend on it, the mean of the coefficients
            does.
        settings (CalibrationSettings): Burn-in, thinning and block size.

    Returns:
        SubspaceMoments: The moments of all samples used.

    Raises:
        ValueError: If a chain does not match the dimension, is not longer than the burn-in, or
            fewer than two samples are used.
    """
    num_vertices, num_coefficients = weighted_eigenvectors.shape
    count = 0
    state_mean = np.zeros(num_vertices)
    coefficient_mean = np.zeros(num_coefficients)
    scatter = np.zeros((num_coefficients, num_coefficients))
    for chain in chains:
        num_samples, chain_vertices = chain.shape
        if chain_vertices != num_vertices:
            raise ValueError(
                f"A chain has {chain_vertices} vertices, the eigenvectors {num_vertices}."
            )
        if num_samples <= settings.burn_in:
            raise ValueError(
                f"A chain with {num_samples} samples is not longer than the burn-in "
                f"{settings.burn_in}."
            )
        stride = settings.block_size * settings.thinning
        for start in range(settings.burn_in, num_samples, stride):
            block = np.asarray(chain[start : min(start + stride, num_samples)])[
                :: settings.thinning
            ]
            coefficients = (block - reference_state) @ weighted_eigenvectors
            block_count = block.shape[0]
            block_coefficient_mean = coefficients.mean(axis=0)
            centered = coefficients - block_coefficient_mean
            total = count + block_count
            delta = block_coefficient_mean - coefficient_mean
            scatter += centered.T @ centered + np.outer(delta, delta) * (
                count * block_count / total
            )
            coefficient_mean += delta * (block_count / total)
            state_mean += (block.mean(axis=0) - state_mean) * (block_count / total)
            count = total
    if count < 2:
        raise ValueError(f"At least two samples are needed for a covariance, got {count}.")
    return SubspaceMoments(
        mean=state_mean,
        coefficient_mean=coefficient_mean,
        coefficient_covariance=scatter / (count - 1),
        num_samples=np.array([count]),
    )


# ==================================================================================================
def calibrate_approximation(
    approximation: LaplaceApproximation,
    reference: ReferenceCovariance,
    moments: SubspaceMoments,
    settings: CalibrationSettings,
    seed: int,
) -> CalibratedApproximation:
    r"""The approximation with the covariance of the leading coefficients replaced by the
    regularized empirical one.

    The regularization of $\Sigma$, the empirical covariance of the first $k$ coefficients, is, in
    this order: shrinkage of the coupling towards the diagonal, the inflation, the lower bound
    relative to the Laplace covariance (the eigenvalues of $L^{-1/2}\Sigma L^{-1/2}$ are raised to
    the minimum ratio, $L = \operatorname{diag}(1/(1+\lambda_i))$), and the upper bound at the
    prior variance.

    Args:
        approximation (LaplaceApproximation): The Laplace approximation the moments refer to, i.e.
            the eigenvectors `moments` was computed with.
        reference (ReferenceCovariance): The prior covariance $\Gamma_0$ of the Gaussian.
        moments (SubspaceMoments): Moments of the pilot chains; its dimension $k$ is the number of
            calibrated coefficients.
        settings (CalibrationSettings): The regularization and `recenter`.
        seed (int): Seed of the sample generator of the new Gaussian.

    Returns:
        CalibratedApproximation: The calibrated approximation and diagnostics.

    Raises:
        ValueError: If the moments have more coefficients than the approximation has eigenpairs.
    """
    eigenvalues = approximation.eigenpairs.eigenvalues
    eigenvectors = approximation.eigenpairs.eigenvectors
    num_coefficients = moments.coefficient_mean.size
    if num_coefficients > eigenvalues.size:
        raise ValueError(
            f"The moments have {num_coefficients} coefficients, the approximation "
            f"{eigenvalues.size} eigenpairs."
        )
    laplace_variances = 1.0 / (1.0 + eigenvalues[:num_coefficients])
    empirical = moments.coefficient_covariance
    empirical = 0.5 * (empirical + empirical.T)
    covariance = (1.0 - settings.coupling_shrinkage) * empirical
    covariance = covariance + settings.coupling_shrinkage * np.diag(np.diag(empirical))
    covariance = settings.variance_inflation * covariance
    if settings.minimum_variance_ratio > 0.0:
        scale = np.sqrt(laplace_variances)
        whitened = covariance / np.outer(scale, scale)
        values, vectors = np.linalg.eigh(0.5 * (whitened + whitened.T))
        values = np.maximum(values, settings.minimum_variance_ratio)
        covariance = (vectors * values) @ vectors.T * np.outer(scale, scale)
    variances, rotation = np.linalg.eigh(0.5 * (covariance + covariance.T))
    variances = np.maximum(variances, _MINIMUM_VARIANCE_FRACTION * laplace_variances.min())
    num_clipped = int(np.sum(variances > _MAXIMUM_VARIANCE))
    variances = np.minimum(variances, _MAXIMUM_VARIANCE)
    covariance = (rotation * variances) @ rotation.T

    new_eigenvalues = np.concatenate([1.0 / variances - 1.0, eigenvalues[num_coefficients:]])
    new_eigenvectors = np.concatenate(
        [eigenvectors[:, :num_coefficients] @ rotation, eigenvectors[:, num_coefficients:]], axis=1
    )
    order = np.argsort(-new_eigenvalues)
    eigenpairs = GeneralizedEigenpairs(new_eigenvalues[order], new_eigenvectors[:, order])
    mean = moments.mean if settings.recenter else approximation.gaussian.mean
    gaussian = LowRankGaussian(reference, eigenpairs, mean=mean, seed=seed)
    return CalibratedApproximation(
        approximation=LaplaceApproximation(gaussian=gaussian, eigenpairs=eigenpairs),
        laplace_variances=laplace_variances,
        empirical_variance_ratios=np.diag(empirical) / laplace_variances,
        calibrated_covariance=covariance,
        calibrated_variance_ratios=np.diag(covariance) / laplace_variances,
        num_clipped=num_clipped,
    )
