r"""Axial (mod-$\pi$) circular statistics for fiber-orientation angle fields.

A fiber-orientation angle is only meaningful modulo $\pi$: a fiber direction and its negation
describe the same physical orientation. These utilities treat angle fields as axial data via the
mean resultant vector of the *doubled* angles, $\frac{1}{N}\sum_j e^{2i\theta_j}$: doubling before
taking the complex exponential identifies angles that differ by $\pi$, so the ordinary complex mean
is already well-defined for axial data, with no separate branch-folding step needed afterward — its
angle (halved) is the axial mean, already on the principal branch $(-\pi/2, \pi/2]$, and its
magnitude measures concentration (`1`: all samples identical, `0`: uniformly spread).

Functions:
    compute_axial_mean_and_variance: Axial mean and variance of angle samples along an axis.
    shift_angles_to_minimize_axial_variance: Re-branch an angle field around its own axial mean.
    compute_axial_data_diff: Signed axial angle difference between two fields.
"""

import numpy as np


# ==================================================================================================
def compute_axial_mean_and_variance(
    angle_samples: np.ndarray[tuple[int, int], np.dtype[np.float64]], axis: int = 1
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.float64]], np.ndarray[tuple[int], np.dtype[np.float64]]
]:
    r"""Compute the axial (mod-$\pi$) mean and variance of angle samples, reduced over `axis`.

    Args:
        angle_samples (np.ndarray): Angle samples in radians, at least 2D.
        axis (int): Axis to reduce over. Defaults to `1`.

    Returns:
        tuple[np.ndarray, np.ndarray]: Axial mean (in $(-\pi/2, \pi/2]$) and axial variance
            $-\tfrac{1}{2}\ln R$, where $R$ is the mean resultant length of the doubled angles.
    """
    angle_samples = np.atleast_2d(angle_samples)
    mean_resultant = np.mean(np.exp(2j * angle_samples), axis=axis, keepdims=False)
    axial_mean = np.angle(mean_resultant) / 2
    axial_variance = -0.5 * np.log(np.abs(mean_resultant))
    return axial_mean, axial_variance


# --------------------------------------------------------------------------------------------------
def shift_angles_to_minimize_axial_variance(
    angle_samples: np.ndarray[tuple[int, int], np.dtype[np.float64]], axis: int = 1
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Re-branch an angle field around its own axial mean, minimizing its axial variance.

    Args:
        angle_samples (np.ndarray): Angle samples in radians, at least 2D.
        axis (int): Axis to compute the axial mean over. Defaults to `1`.

    Returns:
        np.ndarray: `angle_samples`, wrapped onto a single $\\pi$-branch around the axial mean.
    """
    angle_samples = np.atleast_2d(angle_samples)
    mean_resultant = np.mean(np.exp(2j * angle_samples), axis=axis, keepdims=True)
    axial_mean = np.angle(mean_resultant) / 2
    centered_angles = angle_samples - axial_mean
    wrapped_angles = (centered_angles + np.pi / 2) % np.pi - np.pi / 2
    return wrapped_angles + axial_mean


# --------------------------------------------------------------------------------------------------
def compute_axial_data_diff(
    angle_field_one: np.ndarray[tuple[int], np.dtype[np.float64]],
    angle_field_two: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    r"""Compute the signed axial angle difference $\in (-\pi/2, \pi/2]$ between two angle fields.

    Args:
        angle_field_one (np.ndarray): First angle field, in radians.
        angle_field_two (np.ndarray): Second angle field, in radians.

    Returns:
        np.ndarray: Signed axial difference `angle_field_one - angle_field_two`.
    """
    raw_diff = angle_field_one - angle_field_two
    return np.angle(np.exp(2j * raw_diff)) / 2
