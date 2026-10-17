"""Line plots written to image files, with matplotlib's object-oriented API.

The format follows the suffix of the output path: use `.pdf` (vector graphics) for publications.

`matplotlib.figure.Figure` is used directly, not `pyplot`: no global figure registry, no GUI
backend, so the functions are safe on a headless machine and in parallel.

Functions:
    plot_autocorrelation: Autocorrelation functions of a few chains.
    plot_error_cdf: Cumulative distributions of absolute angular errors, with reference thresholds.
    plot_error_by_distance: Error quantiles against the distance to the nearest observation.
    plot_correlation_curve: Binned correlation against distance, with the fitted curve.
    plot_correlation_histogram: Histogram of the pooled pair correlations, with the noise floor.
    plot_histogram: Histogram of the finite values of one array.
    plot_optimization_history: Loss and gradient norm over the iterations of an optimizer.
    plot_optimization_histories: The same for several optimizations in one figure.
    plot_traces: Traces of a few chain components, with the burn-in marked.
"""

import textwrap
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

FIGURE_DPI = 150


# ==================================================================================================
def plot_autocorrelation(
    autocorrelation: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    path: Path,
    labels: Sequence[str] | None = None,
    title: str = "",
) -> None:
    """Plot autocorrelation functions against the lag.

    Args:
        autocorrelation (np.ndarray): One function per column, shape `(num_lags, num_functions)`,
            for the lags `0, ..., num_lags - 1`, e.g. columns of `compute_axial_autocorrelation`.
        path (Path): Output file; the format follows the suffix, e.g. `.pdf`.
        labels (Sequence[str] | None): Legend entry of each function. Defaults to no legend.
        title (str): Title of the plot. Defaults to none.

    Raises:
        ValueError: If `autocorrelation` is not 2D or `labels` does not match its columns.
    """
    if autocorrelation.ndim != 2:
        raise ValueError(f"autocorrelation must be 2D, got shape {autocorrelation.shape}.")
    if labels is not None and len(labels) != autocorrelation.shape[1]:
        raise ValueError(
            f"Got {len(labels)} labels for {autocorrelation.shape[1]} autocorrelation functions."
        )
    figure = Figure(figsize=(6, 4))
    axis = figure.subplots()
    axis.axhline(0.0, color="gray", linewidth=0.8)
    for column_index, function in enumerate(autocorrelation.T):
        axis.plot(function, label=None if labels is None else labels[column_index])
    if labels is not None:
        axis.legend()
    axis.set_xlabel("lag")
    axis.set_ylabel("autocorrelation")
    axis.set_title(title)
    _save(figure, path)


# --------------------------------------------------------------------------------------------------
def plot_correlation_curve(
    bin_distances: np.ndarray[tuple[int], np.dtype[np.float64]],
    bin_correlations: np.ndarray[tuple[int], np.dtype[np.float64]],
    fit_mask: np.ndarray[tuple[int], np.dtype[np.bool_]],
    fitted_correlations: np.ndarray[tuple[int], np.dtype[np.float64]],
    fitted_length: float,
    correlation_threshold: float,
    distance_threshold: float,
    path: Path,
    title: str = "",
    fit_failure_reason: str | None = None,
) -> None:
    """Plot the binned correlation against the distance, with the fit and the threshold.

    Args:
        bin_distances (np.ndarray): Mean pair distance per bin, shape `(num_bins,)`.
        bin_correlations (np.ndarray): Mean correlation per bin, shape `(num_bins,)`.
        fit_mask (np.ndarray): Which bins entered the fit, shape `(num_bins,)`.
        fitted_correlations (np.ndarray): The fitted correlation at `bin_distances`.
        fitted_length (float): Correlation length of the fit, for the legend.
        correlation_threshold (float): Correlation level of the horizontal reference line.
        distance_threshold (float): Distance where the correlation drops below the level; no
            vertical line is drawn if it is not finite.
        path (Path): Output file; the format follows the suffix, e.g. `.pdf`.
        title (str): Title of the plot. Defaults to none.
        fit_failure_reason (str | None): Why the fit failed, or `None` if it succeeded. The fit is
            then not drawn, and the failure is marked in red on the plot. Defaults to `None`.
    """
    figure = Figure(figsize=(6, 4))
    axis = figure.subplots()
    axis.plot(bin_distances, bin_correlations, "o", label="binned correlation")
    if fit_failure_reason is None:
        axis.plot(
            bin_distances[fit_mask],
            fitted_correlations[fit_mask],
            "-",
            label=f"Matern fit, length {fitted_length:.3g}",
        )
    else:
        _mark_failure(axis, "CORRELATION FIT FAILED", fit_failure_reason)
    axis.axhline(correlation_threshold, color="gray", linestyle=":")
    if np.isfinite(distance_threshold):
        axis.axvline(
            distance_threshold,
            color="gray",
            linestyle="--",
            label=f"threshold at {distance_threshold:.3g}",
        )
    axis.set_xlabel("distance")
    axis.set_ylabel("axial correlation")
    axis.set_title(title)
    axis.legend()
    _save(figure, path)


# --------------------------------------------------------------------------------------------------
def plot_correlation_histogram(
    histogram_counts: np.ndarray[tuple[int], np.dtype[np.int64]],
    histogram_edges: np.ndarray[tuple[int], np.dtype[np.float64]],
    noise_floor: float,
    path: Path,
    title: str = "",
    fit_failure_reason: str | None = None,
) -> None:
    """Plot the histogram of the pooled pair correlations, with the sampling noise floor.

    Needs no fit, so it is available also if the correlation length could not be estimated.

    Args:
        histogram_counts (np.ndarray): Number of pairs per bin, shape `(num_bins,)`.
        histogram_edges (np.ndarray): Bin edges, shape `(num_bins + 1,)`.
        noise_floor (float): Correlation level below which the correlation is dominated by
            sampling noise, marked by a vertical line.
        path (Path): Output file; the format follows the suffix, e.g. `.pdf`.
        title (str): Title of the plot. Defaults to none.
        fit_failure_reason (str | None): Why the correlation length fit failed, or `None` if it
            succeeded. The failure is then marked in red on the plot. Defaults to `None`.

    Raises:
        ValueError: If `histogram_edges` does not have one more entry than `histogram_counts`.
    """
    if histogram_edges.shape[0] != histogram_counts.shape[0] + 1:
        raise ValueError(
            f"Got {histogram_counts.shape[0]} counts for {histogram_edges.shape[0]} bin edges."
        )
    figure = Figure(figsize=(6, 4))
    axis = figure.subplots()
    axis.stairs(histogram_counts, histogram_edges, fill=True)
    axis.axvline(noise_floor, color="red", linestyle="--", label=f"noise floor {noise_floor:.3g}")
    axis.set_yscale("log")
    axis.set_xlabel("pair correlation")
    axis.set_ylabel("number of vertex pairs")
    axis.set_title(title)
    axis.legend()
    if fit_failure_reason is not None:
        _mark_failure(axis, "CORRELATION FIT FAILED", fit_failure_reason)
    _save(figure, path)


# --------------------------------------------------------------------------------------------------
def plot_error_cdf(
    absolute_errors_degrees: Sequence[np.ndarray[tuple[int], np.dtype[np.float64]]],
    labels: Sequence[str],
    path: Path,
    thresholds_degrees: Sequence[float] = (),
    title: str = "",
) -> None:
    """Plot the cumulative distribution of absolute angular errors, one curve per error array.

    The curve at an error `e` is the fraction of simplices with an error below `e`, so a curve that
    rises early is better. Overlaying the runs of several initial guesses shows their spread.

    Args:
        absolute_errors_degrees (Sequence[np.ndarray]): Absolute errors in degrees, each of shape
            `(num_simplices,)`.
        labels (Sequence[str]): Legend entry of each curve.
        path (Path): Output file; the format follows the suffix, e.g. `.pdf`.
        thresholds_degrees (Sequence[float]): Errors marked by vertical lines. Defaults to none.
        title (str): Title of the plot. Defaults to none.

    Raises:
        ValueError: If `labels` does not match `absolute_errors_degrees`.
    """
    if len(labels) != len(absolute_errors_degrees):
        raise ValueError(f"Got {len(labels)} labels for {len(absolute_errors_degrees)} curves.")
    figure = Figure(figsize=(6, 4))
    axis = figure.subplots()
    for errors, label in zip(absolute_errors_degrees, labels, strict=True):
        sorted_errors = np.sort(errors)
        fractions = np.arange(1, sorted_errors.shape[0] + 1) / sorted_errors.shape[0]
        axis.plot(sorted_errors, fractions, label=label)
    for threshold in thresholds_degrees:
        axis.axvline(threshold, color="gray", linestyle=":")
    axis.set_xlabel("absolute axial error [deg]")
    axis.set_ylabel("fraction of simplices below")
    axis.set_ylim(0.0, 1.0)
    axis.set_title(title)
    axis.legend()
    _save(figure, path)


# --------------------------------------------------------------------------------------------------
def plot_error_by_distance(
    bin_distances: np.ndarray[tuple[int], np.dtype[np.float64]],
    bin_quantiles: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    quantile_labels: Sequence[str],
    path: Path,
    title: str = "",
) -> None:
    """Plot error quantiles against the distance to the nearest observation.

    Args:
        bin_distances (np.ndarray): Mean distance per bin, shape `(num_bins,)`.
        bin_quantiles (np.ndarray): Error quantiles in degrees per bin, shape
            `(num_bins, num_quantiles)`, e.g. from `compute_error_by_distance` in degrees.
        quantile_labels (Sequence[str]): Legend entry of each quantile.
        path (Path): Output file; the format follows the suffix, e.g. `.pdf`.
        title (str): Title of the plot. Defaults to none.

    Raises:
        ValueError: If `quantile_labels` does not match the columns of `bin_quantiles`.
    """
    if len(quantile_labels) != bin_quantiles.shape[1]:
        raise ValueError(
            f"Got {len(quantile_labels)} labels for {bin_quantiles.shape[1]} quantiles."
        )
    figure = Figure(figsize=(6, 4))
    axis = figure.subplots()
    for quantiles, label in zip(bin_quantiles.T, quantile_labels, strict=True):
        axis.plot(bin_distances, quantiles, "o-", label=label)
    axis.set_xlabel("distance to the nearest observation")
    axis.set_ylabel("absolute axial error [deg]")
    axis.set_title(title)
    axis.legend()
    _save(figure, path)


# --------------------------------------------------------------------------------------------------
def plot_optimization_history(
    loss_history: np.ndarray[tuple[int], np.dtype[np.float64]],
    gradient_norm_history: np.ndarray[tuple[int], np.dtype[np.float64]],
    path: Path,
) -> None:
    """Plot the loss and the gradient norm, both on a log scale, over the optimizer iterations.

    Args:
        loss_history (np.ndarray): Loss per iteration, shape `(num_iterations,)`.
        gradient_norm_history (np.ndarray): Gradient norm per iteration, same shape.
        path (Path): Output file; the format follows the suffix, e.g. `.pdf`.
    """
    figure = Figure(figsize=(10, 4))
    loss_axis, gradient_axis = figure.subplots(1, 2)
    loss_axis.semilogy(loss_history)
    loss_axis.set_xlabel("iteration")
    loss_axis.set_ylabel("loss")
    gradient_axis.semilogy(gradient_norm_history)
    gradient_axis.set_xlabel("iteration")
    gradient_axis.set_ylabel("gradient norm")
    _save(figure, path)


# --------------------------------------------------------------------------------------------------
def plot_optimization_histories(
    loss_histories: Sequence[np.ndarray[tuple[int], np.dtype[np.float64]]],
    gradient_norm_histories: Sequence[np.ndarray[tuple[int], np.dtype[np.float64]]],
    labels: Sequence[str],
    path: Path,
    title: str = "",
) -> None:
    """Plot the losses and gradient norms of several optimizations over the iterations.

    Both are drawn on a log scale, as the loss of different initial guesses spans orders of
    magnitude at the start.

    Args:
        loss_histories (Sequence[np.ndarray]): Loss per iteration, one array per optimization.
        gradient_norm_histories (Sequence[np.ndarray]): Gradient norm per iteration, same layout.
        labels (Sequence[str]): Legend entry of each optimization.
        path (Path): Output file; the format follows the suffix, e.g. `.pdf`.
        title (str): Title of the plot. Defaults to none.

    Raises:
        ValueError: If the sequences differ in length.
    """
    if not len(loss_histories) == len(gradient_norm_histories) == len(labels):
        raise ValueError(
            f"Got {len(loss_histories)} losses, {len(gradient_norm_histories)} gradient norms and "
            f"{len(labels)} labels."
        )
    figure = Figure(figsize=(10, 4))
    loss_axis, gradient_axis = figure.subplots(1, 2)
    for loss, gradient_norm, label in zip(
        loss_histories, gradient_norm_histories, labels, strict=True
    ):
        loss_axis.semilogy(loss, label=label)
        gradient_axis.semilogy(gradient_norm, label=label)
    loss_axis.set_xlabel("iteration")
    loss_axis.set_ylabel("loss")
    gradient_axis.set_xlabel("iteration")
    gradient_axis.set_ylabel("gradient norm")
    gradient_axis.legend()
    figure.suptitle(title)
    _save(figure, path)


# --------------------------------------------------------------------------------------------------
def plot_histogram(
    values: np.ndarray[tuple[int], np.dtype[np.float64]],
    path: Path,
    xlabel: str,
    title: str = "",
) -> None:
    """Plot a histogram of the finite entries of `values`.

    Args:
        values (np.ndarray): Values, shape `(num_values,)`; non-finite entries are ignored.
        path (Path): Output file; the format follows the suffix, e.g. `.pdf`.
        xlabel (str): Label of the horizontal axis.
        title (str): Title of the plot. Defaults to none.

    Raises:
        ValueError: If `values` has no finite entry.
    """
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("values has no finite entry to plot.")
    figure = Figure(figsize=(6, 4))
    axis = figure.subplots()
    axis.hist(finite_values, bins=50)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("number of vertices")
    axis.set_title(title)
    _save(figure, path)


# --------------------------------------------------------------------------------------------------
def plot_traces(
    traces: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    path: Path,
    labels: Sequence[str],
    burn_in: int,
) -> None:
    """Plot chain traces in vertically stacked panels, one per component.

    Args:
        traces (np.ndarray): One chain per column, shape `(num_samples, num_components)`.
        path (Path): Output file; the format follows the suffix, e.g. `.pdf`.
        labels (Sequence[str]): Vertical-axis label of each component.
        burn_in (int): Number of leading samples, marked by a vertical line.

    Raises:
        ValueError: If `traces` is not 2D or has no column, or `labels` does not match its columns.
    """
    if traces.ndim != 2 or traces.shape[1] == 0:
        raise ValueError(f"traces must be 2D with at least one column, got shape {traces.shape}.")
    if len(labels) != traces.shape[1]:
        raise ValueError(f"Got {len(labels)} labels for {traces.shape[1]} traces.")
    figure = Figure(figsize=(8, 2.2 * traces.shape[1] + 0.8))
    axes = np.atleast_1d(figure.subplots(traces.shape[1], 1, sharex=True))
    for axis, trace, label in zip(axes, traces.T, labels, strict=True):
        axis.plot(trace, linewidth=0.8)
        axis.axvline(burn_in, color="gray", linestyle="--", label="end of burn-in")
        axis.set_ylabel(f"{label} [rad]")
    axes[0].legend()
    axes[-1].set_xlabel("sample")
    _save(figure, path)


# ==================================================================================================
def _mark_failure(axis: Axes, headline: str, reason: str) -> None:
    """Mark a failed computation on `axis`: a red frame and a red message box in its center."""
    for spine in axis.spines.values():
        spine.set_edgecolor("red")
        spine.set_linewidth(2.0)
    axis.text(
        0.5,
        0.5,
        f"{headline}\n{textwrap.fill(reason, width=40)}",
        transform=axis.transAxes,
        ha="center",
        va="center",
        color="red",
        fontsize=9,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "red", "alpha": 0.85},
    )


# --------------------------------------------------------------------------------------------------
def _save(figure: Figure, path: Path) -> None:
    """Write `figure` to `path` with a tight layout."""
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI)
