"""Line plots written to PNG files, with matplotlib's object-oriented API.

`matplotlib.figure.Figure` is used directly, not `pyplot`: no global figure registry, no GUI
backend, so the functions are safe on a headless machine and in parallel.

Functions:
    plot_autocorrelation: Autocorrelation functions of a few chains.
    plot_correlation_curve: Binned correlation against distance, with the fitted curve.
    plot_optimization_history: Loss and gradient norm over the iterations of an optimizer.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
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
        path (Path): Output PNG file.
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
    axis.plot(autocorrelation)
    if labels is not None:
        axis.legend(labels)
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
        path (Path): Output PNG file.
        title (str): Title of the plot. Defaults to none.
    """
    figure = Figure(figsize=(6, 4))
    axis = figure.subplots()
    axis.plot(bin_distances, bin_correlations, "o", label="binned correlation")
    axis.plot(
        bin_distances[fit_mask],
        fitted_correlations[fit_mask],
        "-",
        label=f"Matern fit, length {fitted_length:.3g}",
    )
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
def plot_optimization_history(
    loss_history: np.ndarray[tuple[int], np.dtype[np.float64]],
    gradient_norm_history: np.ndarray[tuple[int], np.dtype[np.float64]],
    path: Path,
) -> None:
    """Plot the loss and, on a log scale, the gradient norm over the optimizer iterations.

    Args:
        loss_history (np.ndarray): Loss per iteration, shape `(num_iterations,)`.
        gradient_norm_history (np.ndarray): Gradient norm per iteration, same shape.
        path (Path): Output PNG file.
    """
    figure = Figure(figsize=(10, 4))
    loss_axis, gradient_axis = figure.subplots(1, 2)
    loss_axis.plot(loss_history)
    loss_axis.set_xlabel("iteration")
    loss_axis.set_ylabel("loss")
    gradient_axis.semilogy(gradient_norm_history)
    gradient_axis.set_xlabel("iteration")
    gradient_axis.set_ylabel("gradient norm")
    _save(figure, path)


# ==================================================================================================
def _save(figure: Figure, path: Path) -> None:
    """Write `figure` to `path` with a tight layout."""
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI)
