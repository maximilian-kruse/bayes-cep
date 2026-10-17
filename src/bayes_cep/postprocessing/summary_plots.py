"""Plots that summarize a whole study, written to image files.

Functions:
    plot_parameter_heatmap: A value over a grid of two study parameters, cells without a value
        marked.
"""

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
