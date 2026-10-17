r"""Error metrics of an axial fiber-angle estimate against a reference field.

All metrics work on the signed axial difference $\Delta \in (-\pi/2, \pi/2]$ of two fields (see
`axial_statistics.compute_axial_data_diff`), so that a shift by $\pi$ is no error. Angular errors
are reported in degrees. The functions take the difference per simplex, the resolution of the
piecewise-constant fields the forward model works with.

Constants:
    ERROR_PERCENTILES: Percentiles of the absolute error that are reported.
    ERROR_THRESHOLDS_DEGREES: Thresholds of the fractions of simplices with a smaller error.

Functions:
    compute_axial_error_metrics: Scalar summary of the absolute axial error.
    compute_skill_score: Improvement of an error over the error of a baseline.
    compute_error_by_distance: Quantiles of the error in bins of the distance to the data.
"""

import numpy as np

ERROR_PERCENTILES = (90, 95, 99)
ERROR_THRESHOLDS_DEGREES = (5, 10, 20)


# ==================================================================================================
def compute_axial_error_metrics(
    axial_difference: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> dict[str, float]:
    r"""Summarize an axial difference field by scalar metrics, angles in degrees.

    The keys are `error_mean_deg`, `error_median_deg`, `error_p<q>_deg` for the percentiles in
    `ERROR_PERCENTILES`, `error_max_deg`, `fraction_below_<t>_deg` for the thresholds in
    `ERROR_THRESHOLDS_DEGREES`, and `axial_concentration`: the mean of $\cos(2\Delta)$, which is
    one for a perfect match and zero for random orientations. It is bounded, so that a few
    perpendicular simplices cannot dominate it.

    Args:
        axial_difference (np.ndarray): Signed axial difference in radians, shape
            `(num_simplices,)`.

    Returns:
        dict[str, float]: The metrics.
    """
    absolute_error_degrees = np.degrees(np.abs(axial_difference))
    metrics = {
        "error_mean_deg": float(absolute_error_degrees.mean()),
        "error_median_deg": float(np.median(absolute_error_degrees)),
    }
    for percentile in ERROR_PERCENTILES:
        metrics[f"error_p{percentile}_deg"] = float(
            np.percentile(absolute_error_degrees, percentile)
        )
    metrics["error_max_deg"] = float(absolute_error_degrees.max())
    for threshold in ERROR_THRESHOLDS_DEGREES:
        metrics[f"fraction_below_{threshold}_deg"] = float(
            np.mean(absolute_error_degrees < threshold)
        )
    metrics["axial_concentration"] = float(np.mean(np.cos(2.0 * axial_difference)))
    return metrics


# --------------------------------------------------------------------------------------------------
def compute_skill_score(error: float, baseline_error: float) -> float:
    r"""Compute $1 - \text{error}/\text{baseline error}$: one for no error, zero for no gain.

    Args:
        error (float): Error of the estimate.
        baseline_error (float): Error of the baseline, e.g. of the prior mean.

    Returns:
        float: The skill score; negative if the estimate is worse than the baseline.

    Raises:
        ValueError: If `baseline_error` is not positive.
    """
    if baseline_error <= 0.0:
        raise ValueError(f"The baseline error must be positive, got {baseline_error}.")
    return 1.0 - error / baseline_error


# --------------------------------------------------------------------------------------------------
def compute_error_by_distance(
    distances: np.ndarray[tuple[int], np.dtype[np.float64]],
    absolute_errors: np.ndarray[tuple[int], np.dtype[np.float64]],
    num_bins: int,
    quantiles: tuple[float, ...] = (0.5, 0.9),
) -> tuple[np.ndarray[tuple[int], np.dtype[np.float64]], np.ndarray[tuple[int, int], np.dtype]]:
    """Compute quantiles of the error in bins of the distance, with equally many simplices per bin.

    Args:
        distances (np.ndarray): Distance of each simplex to the nearest observation, shape
            `(num_simplices,)`.
        absolute_errors (np.ndarray): Absolute error of each simplex, same shape.
        num_bins (int): Number of bins.
        quantiles (tuple[float, ...]): Quantiles of the error to compute. Defaults to the median
            and the 90th percentile.

    Returns:
        tuple[np.ndarray, np.ndarray]: Mean distance of each bin, shape `(num_bins,)`, and the
            error quantiles, shape `(num_bins, len(quantiles))`.

    Raises:
        ValueError: If the inputs differ in shape or there are fewer simplices than bins.
    """
    if distances.shape != absolute_errors.shape:
        raise ValueError(
            f"Got distances of shape {distances.shape} and errors of shape {absolute_errors.shape}."
        )
    if distances.shape[0] < num_bins:
        raise ValueError(f"Cannot fill {num_bins} bins with {distances.shape[0]} simplices.")
    order = np.argsort(distances)
    bins = np.array_split(order, num_bins)
    bin_distances = np.array([distances[bin_indices].mean() for bin_indices in bins])
    bin_quantiles = np.array(
        [np.quantile(absolute_errors[bin_indices], quantiles) for bin_indices in bins]
    )
    return bin_distances, bin_quantiles
