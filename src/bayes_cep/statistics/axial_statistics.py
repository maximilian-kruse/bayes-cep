r"""Axial (mod-$\pi$) circular statistics for fiber-orientation angle fields.

A fiber-orientation angle is only meaningful modulo $\pi$: a fiber direction and its negation
describe the same physical orientation. These utilities treat angle fields as axial data via the
mean resultant vector of the *doubled* angles, $\frac{1}{N}\sum_j e^{2i\theta_j}$: doubling before
taking the complex exponential identifies angles that differ by $\pi$, so the ordinary complex mean
is already well-defined for axial data, with no separate branch-folding step needed afterward — its
angle (halved) is the axial mean, already on the principal branch $(-\pi/2, \pi/2]$, and its
magnitude measures concentration (`1`: all samples identical, `0`: uniformly spread).

Functions:
    compute_axial_mean: Axial mean of an in-memory array of angles.
    compute_axial_mean_and_variance: Axial mean and variance of angle samples, read block by block.
    compute_axial_sine_deviation: Sine of the doubled deviation from an axial mean.
    shift_angles_to_minimize_axial_variance: Re-branch an angle field around its own axial mean.
    compute_axial_data_diff: Signed axial angle difference between two fields.
    wrap_axial_angles: Wrap angles onto the principal branch $(-\pi/2, \pi/2]$.
"""

import numpy as np

from bayes_cep.statistics.sample_blocks import SampleStore, iterate_sample_blocks


# ==================================================================================================
def compute_axial_mean_and_variance(
    samples: SampleStore, burn_in: int = 0, block_size: int = 100
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.float64]], np.ndarray[tuple[int], np.dtype[np.float64]]
]:
    r"""Compute the pointwise axial (mod-$\pi$) mean and variance of angle samples.

    Reads the samples block by block, so e.g. a Zarr store never has to fit in memory; an in-memory
    array works as well.

    Args:
        samples (SampleStore): Angle samples in radians, shape `(num_samples, num_components)`,
            e.g. a `numpy` or `zarr` array. Reduced over the first axis.
        burn_in (int): Number of leading samples to discard. Defaults to `0`.
        block_size (int): Number of samples read at once. Defaults to `100`.

    Returns:
        tuple[np.ndarray, np.ndarray]: Axial mean (in $(-\pi/2, \pi/2]$) and axial variance
            $-\tfrac{1}{2}\ln R$, where $R$ is the length of the mean resultant of the doubled
            angles, each of shape `(num_components,)`.
    """
    resultant_sum = np.zeros(samples.shape[1], dtype=np.complex128)
    for block in iterate_sample_blocks(samples, burn_in, block_size):
        resultant_sum += np.exp(2j * block).sum(axis=0)
    mean_resultant = resultant_sum / (samples.shape[0] - burn_in)
    return np.angle(mean_resultant) / 2, -0.5 * np.log(np.abs(mean_resultant))


# --------------------------------------------------------------------------------------------------
def compute_axial_mean(
    angles: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    axis: int | None = None,
    keepdims: bool = False,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    r"""Compute the axial (mod-$\pi$) mean of an in-memory array of angles.

    Args:
        angles (np.ndarray): Angles in radians.
        axis (int | None): Axis to average over; all entries if `None`. Defaults to `None`.
        keepdims (bool): Whether to keep the averaged axis with length one. Defaults to `False`.

    Returns:
        np.ndarray: Axial mean in $(-\pi/2, \pi/2]$.
    """
    mean_resultant = np.mean(np.exp(2j * angles), axis=axis, keepdims=keepdims)
    return np.angle(mean_resultant) / 2


# --------------------------------------------------------------------------------------------------
def compute_axial_sine_deviation(
    angles: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    axial_mean: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    r"""Compute $\sin(2(\theta - \mu))$, the deviation of angles from their axial mean.

    The result is invariant under $\theta \to \theta + \pi$ and, for a concentrated field, close
    to $2(\theta - \mu)$, so it can be treated like an ordinary real variable (autocorrelation,
    correlation).

    Args:
        angles (np.ndarray): Angles in radians.
        axial_mean (np.ndarray): Axial mean $\mu$ in radians, broadcastable against `angles`.

    Returns:
        np.ndarray: The sine deviation, same shape as `angles`.
    """
    return np.sin(2.0 * (angles - axial_mean))


# --------------------------------------------------------------------------------------------------
def shift_angles_to_minimize_axial_variance(
    angles: np.ndarray[tuple[int, ...], np.dtype[np.float64]], axis: int | None = None
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    r"""Re-branch angles around their axial mean, minimizing their axial variance.

    Args:
        angles (np.ndarray): Angles in radians, any shape.
        axis (int | None): Axis to compute the axial mean over; all entries if `None`. Defaults to
            `None`.

    Returns:
        np.ndarray: `angles`, same shape, each shifted by a multiple of $\pi$ onto the branch
            $(\mu - \pi/2, \mu + \pi/2]$ around its axial mean $\mu$.
    """
    axial_mean = compute_axial_mean(angles, axis=axis, keepdims=True)
    centered_angles = angles - axial_mean
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
    return wrap_axial_angles(angle_field_one - angle_field_two)


# --------------------------------------------------------------------------------------------------
def wrap_axial_angles(
    angles: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    r"""Wrap angles onto $(-\pi/2, \pi/2]$, the principal branch of axial data.

    Args:
        angles (np.ndarray): Angles in radians.

    Returns:
        np.ndarray: The angles modulo $\pi$, in $(-\pi/2, \pi/2]$.
    """
    return np.angle(np.exp(2j * angles)) / 2
