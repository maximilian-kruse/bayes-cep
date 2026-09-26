"""Gaussian activation-time likelihood for sparse point observations at mesh vertices.

Classes:
    LikelihoodSettings: Settings for observed activation times at a subset of mesh vertices.

Functions:
    build_activation_time_likelihood: Build a `GaussianLogLikelihood` from `LikelihoodSettings`.
"""

from dataclasses import dataclass

import numpy as np
from ls_bayesian.posterior import likelihood


# ==================================================================================================
@dataclass(frozen=True)
class LikelihoodSettings:
    """Settings for observed activation times at a subset of mesh vertices.

    Observations are modelled as i.i.d. Gaussian noise around the true activation time at each
    observed vertex (homoscedastic, uncorrelated).

    Attributes:
        num_vertices (int): Number of mesh vertices, i.e. the forward map's solution dimension.
        observed_vertex_indices (np.ndarray): Indices of the observed vertices, shape
            `(num_observations,)`. Each vertex may be observed at most once.
        data_vector (np.ndarray): Observed (noisy) activation times at those vertices, shape
            `(num_observations,)`.
        noise_variance (float): Variance $\\sigma^2$ of the i.i.d. Gaussian observation noise.
    """

    num_vertices: int
    observed_vertex_indices: np.ndarray[tuple[int], np.dtype[np.integer]]
    data_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    noise_variance: float


# ==================================================================================================
def build_activation_time_likelihood(
    settings: LikelihoodSettings,
) -> likelihood.GaussianLogLikelihood:
    """Build a `GaussianLogLikelihood` for sparse, homoscedastic point observations.

    Args:
        settings (LikelihoodSettings): Observation settings.

    Returns:
        likelihood.GaussianLogLikelihood: The assembled likelihood.
    """
    precision_values = np.full(
        settings.observed_vertex_indices.shape[0], 1.0 / settings.noise_variance
    )
    observation_settings = likelihood.VertexObservationSettings(
        data_vector=settings.data_vector,
        num_vertices=settings.num_vertices,
        observed_vertex_indices=settings.observed_vertex_indices,
        precision_values=precision_values,
    )
    return likelihood.GaussianLogLikelihood.from_vertex_observations(observation_settings)
