"""Constant prior-mean fiber-angle field.

Functions:
    build_constant_prior_mean: Spatially constant prior mean from a ground-truth angle field.
"""

import numpy as np

from bayes_cep.preprocessing.axial_statistics import compute_axial_mean_and_variance


# ==================================================================================================
def build_constant_prior_mean(
    ground_truth_angle_field: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Build a spatially constant prior-mean field from a ground-truth angle field.

    With a single patient, the cross-patient ensemble mean the legacy pipeline used as a prior mean
    is unavailable. This uses the spatial axial (mod-$\\pi$) mean of the one ground-truth field as a
    deliberately uninformative constant prior mean instead: it preserves the same role — a prior
    mean that is wrong relative to the spatially-varying truth — without fabricating cross-patient
    variability that doesn't exist here. A plain arithmetic mean would be invalid here, the same way
    it is for `angle_field.interpolate_simplex_field_to_vertices`: `ground_truth_angle_field` is
    $\\pi$-periodic, so values near opposite ends of its branch can be physically adjacent.

    Args:
        ground_truth_angle_field (np.ndarray): Per-vertex ground-truth fiber angle field, shape
            `(num_vertices,)`.

    Returns:
        np.ndarray: Constant prior-mean field, shape `(num_vertices,)`.
    """
    axial_mean, _ = compute_axial_mean_and_variance(ground_truth_angle_field)
    return axial_mean * np.ones_like(ground_truth_angle_field)
