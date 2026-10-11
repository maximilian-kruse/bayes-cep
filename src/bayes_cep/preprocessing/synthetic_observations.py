"""Synthetic sparse, noisy activation-time observations for the fiber-orientation likelihood.

Classes:
    ObservationSamplingSettings: Settings for sampling synthetic observations.

Functions:
    generate_synthetic_observations: Sample sparse, noisy activation-time observations from a
        forward-solved ground-truth activation-time field.
"""

from dataclasses import dataclass

import numpy as np
from ls_bayesian.posterior.interfaces import ParameterToSolutionMap


# ==================================================================================================
@dataclass(frozen=True)
class ObservationSamplingSettings:
    r"""Settings for sampling synthetic activation-time observations.

    Attributes:
        num_observations (int): Number of vertices to observe.
        noise_variance (float): Variance $\sigma^2$ of the i.i.d. Gaussian observation noise.
        seed (int): Random seed for vertex sampling and noise.
    """

    num_observations: int
    noise_variance: float
    seed: int


# ==================================================================================================
def generate_synthetic_observations(
    forward_map: ParameterToSolutionMap,
    ground_truth_parameter: np.ndarray[tuple[int], np.dtype[np.float64]],
    settings: ObservationSamplingSettings,
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.integer]], np.ndarray[tuple[int], np.dtype[np.float64]]
]:
    """Forward-solve the ground truth and sample sparse, noisy vertex observations from it.

    Args:
        forward_map (ParameterToSolutionMap): Forward map from the fiber angle to the
            activation times; only `evaluate_forward` is used.
        ground_truth_parameter (np.ndarray): Ground-truth per-vertex fiber angle field.
        settings (ObservationSamplingSettings): Number of observations, noise level, and seed.

    Returns:
        tuple[np.ndarray, np.ndarray]: Observed vertex indices (sorted, without replacement) and
            noisy activation times at those vertices, each shape `(num_observations,)`.
    """
    activation_times = forward_map.evaluate_forward(ground_truth_parameter)
    rng = np.random.default_rng(settings.seed)
    observed_vertex_indices = rng.choice(
        activation_times.shape[0], size=settings.num_observations, replace=False
    )
    observed_vertex_indices.sort()
    noise = rng.normal(scale=np.sqrt(settings.noise_variance), size=settings.num_observations)
    noisy_data = activation_times[observed_vertex_indices] + noise
    return observed_vertex_indices, noisy_data
