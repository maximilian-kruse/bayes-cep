"""Plots that summarize a whole study, written to image files.

Functions:
    plot_parameter_heatmap: A value over a grid of two study parameters, cells without a value
        marked.
    plot_loss_maps: Loss over the $(\\kappa, \\tau)$ plane, one panel per loss surface.
    plot_estimates_versus_truth: Estimated against true values of several quantities, one panel
        each.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

FIGURE_DPI = 150


# ==================================================================================================
def plot_parameter_heatmap(
    values: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    row_values: list[float],
    column_values: list[float],
    row_name: str,
    column_name: str,
    colorbar_label: str,
    path: Path,
    title: str = "",
    log_scale: bool = False,
) -> None:
    """Plot a value over the grid of two parameters, e.g. an error over `kappa` and `tau`.

    Cells without a finite value (a parameter pair that was not run, or whose runs failed) are
    hatched and marked "n/a". Every other cell shows its value.

    Args:
        values (np.ndarray): Values, shape `(len(row_values), len(column_values))`; `nan` marks a
            missing value.
        row_values (list[float]): Parameter values of the rows, plotted bottom to top.
        column_values (list[float]): Parameter values of the columns, plotted left to right.
        row_name (str): Name of the row parameter.
        column_name (str): Name of the column parameter.
        colorbar_label (str): Label of the color bar.
        path (Path): Output file; the format follows the suffix, e.g. `.pdf`.
        title (str): Title of the plot. Defaults to none.
        log_scale (bool): Whether the color scale is logarithmic; needs positive values. Defaults
            to `False`.

    Raises:
        ValueError: If the shape of `values` does not match the parameter values.
    """
    if values.shape != (len(row_values), len(column_values)):
        raise ValueError(
            f"Got values of shape {values.shape} for {len(row_values)} rows and "
            f"{len(column_values)} columns."
        )
    valid = np.isfinite(values)
    figure = Figure(figsize=(1.0 * len(column_values) + 2.5, 0.8 * len(row_values) + 1.5))
    axis = figure.subplots()
    norm = None
    if log_scale and valid.any():
        norm = LogNorm(vmin=values[valid].min(), vmax=values[valid].max())
    image = axis.imshow(
        np.ma.masked_invalid(values), origin="lower", cmap="viridis", norm=norm, aspect="auto"
    )
    for row, column in np.argwhere(~valid):
        axis.add_patch(Rectangle((column - 0.5, row - 0.5), 1, 1, fill=False, hatch="//"))
        axis.text(column, row, "n/a", ha="center", va="center", fontsize=8)
    for row, column in np.argwhere(valid):
        axis.text(
            column,
            row,
            f"{values[row, column]:.3g}",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
        )
    axis.set_xticks(range(len(column_values)), [f"{value:g}" for value in column_values])
    axis.set_yticks(range(len(row_values)), [f"{value:g}" for value in row_values])
    axis.set_xlabel(column_name)
    axis.set_ylabel(row_name)
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label=colorbar_label)
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI)


# ==================================================================================================
def plot_loss_maps(
    kappas: np.ndarray[tuple[int], np.dtype[np.float64]],
    taus: np.ndarray[tuple[int], np.dtype[np.float64]],
    losses: Sequence[np.ndarray[tuple[int, int], np.dtype[np.float64]]],
    labels: Sequence[str],
    path: Path,
    title: str = "",
    max_num_columns: int = 4,
    true_parameters: Sequence[tuple[float, float]] | None = None,
) -> None:
    r"""Plot loss surfaces over the $(\kappa, \tau)$ plane, one panel per surface.

    Both axes and the color are logarithmic. Every panel has its own color range, from its minimum
    to its largest finite value, so the valley of small loss is visible whatever the scale of the
    loss. The minimum is marked, and the dashed curve is the line of constant $\kappa\tau$ through
    it, which the valley follows if the variance is what is identified best.

    Args:
        kappas (np.ndarray): Values of $\kappa$ (rows of the losses), shape `(num_kappas,)`.
        taus (np.ndarray): Values of $\tau$ (columns of the losses), shape `(num_taus,)`.
        losses (Sequence[np.ndarray]): Positive loss surfaces, each of shape
            `(num_kappas, num_taus)`.
        labels (Sequence[str]): Title of each panel.
        path (Path): Output file; the format follows the suffix, e.g. `.pdf`.
        title (str): Title of the figure. Defaults to none.
        max_num_columns (int): Panels per row at most. Defaults to 4.
        true_parameters (Sequence[tuple[float, float]] | None): The known $(\kappa, \tau)$ of
            each surface, marked by a white cross. Defaults to none.

    Raises:
        ValueError: If there are no surfaces, the labels do not match them, or a surface does not
            match the parameter values.
    """
    if len(losses) == 0 or len(labels) != len(losses):
        raise ValueError(f"Got {len(labels)} labels for {len(losses)} loss surfaces.")
    if true_parameters is not None and len(true_parameters) != len(losses):
        raise ValueError(f"Got {len(true_parameters)} true parameters for {len(losses)} surfaces.")
    for loss in losses:
        if loss.shape != (len(kappas), len(taus)):
            raise ValueError(
                f"Got a loss of shape {loss.shape} for {len(kappas)} kappas and {len(taus)} taus."
            )
    num_columns = min(len(losses), max_num_columns)
    num_rows = -(-len(losses) // num_columns)
    figure = Figure(figsize=(4.2 * num_columns, 3.6 * num_rows), layout="constrained")
    axes = np.atleast_1d(figure.subplots(num_rows, num_columns, squeeze=False)).ravel()
    kappa_grid = np.geomspace(kappas.min(), kappas.max(), 200)
    for panel, (axis, loss, label) in enumerate(zip(axes, losses, labels, strict=False)):
        finite = np.isfinite(loss) & (loss > 0.0)
        image = axis.pcolormesh(
            taus,
            kappas,
            np.ma.masked_where(~finite, loss),
            shading="nearest",
            cmap="viridis",
            norm=LogNorm(vmin=loss[finite].min(), vmax=loss[finite].max()),
        )
        best_row, best_column = np.unravel_index(
            np.argmin(np.where(finite, loss, np.inf)), loss.shape
        )
        axis.plot(taus[best_column], kappas[best_row], "r*", markersize=12)
        if true_parameters is not None:
            axis.plot(
                true_parameters[panel][1], true_parameters[panel][0], "wx", markersize=9, mew=2
            )
        axis.plot(
            taus[best_column] * kappas[best_row] / kappa_grid,
            kappa_grid,
            "r--",
            linewidth=0.8,
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(taus.min(), taus.max())
        axis.set_ylim(kappas.min(), kappas.max())
        axis.set_xlabel("tau")
        axis.set_ylabel("kappa")
        axis.set_title(
            f"{label}\nmin {loss[best_row, best_column]:.3g} at "
            f"kappa {kappas[best_row]:.3g}, tau {taus[best_column]:.3g}",
            fontsize=9,
        )
        figure.colorbar(image, ax=axis, label="loss")
    for axis in axes[len(losses) :]:
        axis.set_visible(False)
    if title:
        figure.suptitle(title)
    figure.savefig(path, dpi=FIGURE_DPI)


# ==================================================================================================
def plot_estimates_versus_truth(
    quantity_names: Sequence[str],
    truths: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    estimates: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    group_labels: Sequence[str],
    path: Path,
    title: str = "",
) -> None:
    """Plot estimated against true values on logarithmic axes, one panel per quantity.

    Args:
        quantity_names (Sequence[str]): Name of each quantity, e.g. `kappa`, `tau`, `kappa * tau`.
        truths (np.ndarray): Positive true values, shape `(num_points, num_quantities)`.
        estimates (np.ndarray): Positive estimates, same shape.
        group_labels (Sequence[str]): Group of each point, shape `(num_points,)`; every group has
            its own color and legend entry.
        path (Path): Output file; the format follows the suffix, e.g. `.pdf`.
        title (str): Title of the figure. Defaults to none.

    Raises:
        ValueError: If the shapes do not match.
    """
    if truths.shape != estimates.shape or truths.shape != (len(group_labels), len(quantity_names)):
        raise ValueError(
            f"Got truths of shape {truths.shape}, estimates of shape {estimates.shape}, "
            f"{len(group_labels)} group labels and {len(quantity_names)} quantities."
        )
    labels = np.asarray(group_labels)
    figure = Figure(figsize=(4.2 * len(quantity_names), 4.0), layout="constrained")
    axes = np.atleast_1d(figure.subplots(1, len(quantity_names)))
    for column, (axis, name) in enumerate(zip(axes, quantity_names, strict=True)):
        low = min(truths[:, column].min(), estimates[:, column].min())
        high = max(truths[:, column].max(), estimates[:, column].max())
        axis.plot([low, high], [low, high], "k--", linewidth=0.8)
        for group in dict.fromkeys(labels):
            chosen = labels == group
            axis.scatter(
                truths[chosen, column], estimates[chosen, column], s=14, alpha=0.7, label=group
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(f"true {name}")
        axis.set_ylabel(f"estimated {name}")
        axis.set_aspect("equal")
    axes[0].legend(fontsize=8)
    if title:
        figure.suptitle(title)
    figure.savefig(path, dpi=FIGURE_DPI)
