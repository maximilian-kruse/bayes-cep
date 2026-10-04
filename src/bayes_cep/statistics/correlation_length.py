r"""Correlation length of a fiber-angle field, estimated from samples.

The fiber angle is only defined modulo $\pi$, so correlations are computed for axial data: with
$\mu_x$ the axial mean at vertex $x$ (see `bayes_cep.statistics.axial_statistics`), the field
$s_x = \sin(2(\theta_x - \mu_x))$ is invariant under $\theta_x \to \theta_x + \pi$, and the
correlation $\rho(x, y)$ is the Pearson correlation of $s_x$ and $s_y$ over the samples (the
circular correlation coefficient of Jammalamadaka and Sarma, for doubled angles). For small angular
spread $s_x \approx 2(\theta_x - \mu_x)$, so this reduces to the ordinary correlation.

$\rho$ is pooled over pairs of vertices $(b, y)$ with $b$ among a random set of base vertices,
binned by the graph distance $d(b, y)$ along the mesh (see `bayes_cep.mesh.geodesic`). The primary
output is the model-free correlation distance threshold: the distance where the binned mean
correlation first drops below a fixed threshold. It assumes no covariance model, which matters on a
curved surface, where a Matérn correlation in geodesic distance is in general not a valid
covariance. As a secondary summary, the binned mean correlation is also fitted by the Matérn
correlation with smoothness $\nu = 1$ (that of the 2D Bilaplacian SPDE prior),
$\rho(d) = r K_1(r)$ with $r = \sqrt{8}\, d / \ell$, which has correlation $\approx 0.14$ at
$d = \ell$; read $\ell$ as an effective length, not as the exact parameter of the field.

Classes:
    CorrelationLengthSettings: Parameters of the estimation.
    CorrelationLengthResult: Correlation distance threshold, fitted length and the binned curve.

Functions:
    estimate_correlation_length: Estimate the correlation distance from samples.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.optimize import curve_fit
from scipy.special import kv

from bayes_cep.mesh.geodesic import (
    assemble_edge_length_graph,
    compute_graph_distances,
    find_boundary_vertices,
)
from bayes_cep.statistics.axial_statistics import compute_axial_mean_and_variance
from bayes_cep.statistics.sample_blocks import SampleStore, iterate_sample_blocks


# ==================================================================================================
@dataclass(frozen=True)
class CorrelationLengthSettings:
    """Parameters of the correlation length estimation.

    Attributes:
        max_distance (float): Largest pair distance considered, in mesh units; a few times the
            expected correlation length.
        num_base_vertices (int): Number of randomly drawn base vertices. Defaults to `200`.
        num_bins (int): Number of distance bins in `[0, max_distance]`. Defaults to `40`.
        boundary_margin (float): Base vertices closer than this to the mesh boundary are excluded,
            since boundary effects distort the correlation there. Defaults to `0.0`.
        noise_floor_scale (float): The fit uses only the bins before the binned correlation first
            drops below `noise_floor_scale / sqrt(num_samples)`, beyond which it is dominated by
            sampling noise. Defaults to `2.0`.
        correlation_threshold (float): Correlation value whose crossing distance is reported as the
            model-free `correlation_distance_threshold`. Defaults to `1/e`.
        burn_in (int): Number of leading samples to discard. Defaults to `0`.
        block_size (int): Number of samples read at once. Defaults to `100`.
        seed (int): Seed for drawing the base vertices. Defaults to `0`.
    """

    max_distance: float
    num_base_vertices: int = 200
    num_bins: int = 40
    boundary_margin: float = 0.0
    noise_floor_scale: float = 2.0
    correlation_threshold: float = float(np.exp(-1.0))
    burn_in: int = 0
    block_size: int = 100
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate the settings."""
        if self.max_distance <= 0:
            raise ValueError(f"max_distance must be positive, got {self.max_distance}.")
        if self.num_base_vertices <= 0:
            raise ValueError(f"num_base_vertices must be positive, got {self.num_base_vertices}.")
        if self.num_bins < 2:
            raise ValueError(f"num_bins must be at least 2, got {self.num_bins}.")
        if self.boundary_margin < 0:
            raise ValueError(f"boundary_margin must be non-negative, got {self.boundary_margin}.")
        if self.noise_floor_scale <= 0:
            raise ValueError(f"noise_floor_scale must be positive, got {self.noise_floor_scale}.")
        if not 0 < self.correlation_threshold < 1:
            raise ValueError(
                f"correlation_threshold must be in (0, 1), got {self.correlation_threshold}."
            )
        if self.burn_in < 0:
            raise ValueError(f"burn_in must be non-negative, got {self.burn_in}.")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}.")


# ==================================================================================================
@dataclass(frozen=True)
class CorrelationLengthResult:
    """Estimated correlation distance and length, and the binned correlation curve behind them.

    Attributes:
        correlation_distance_threshold (float): Distance where the binned correlation first drops
            below `correlation_threshold`, linearly interpolated; `nan` if it never does. The
            primary, model-free estimate.
        length (float): Secondary summary: correlation length $\\ell$ of the fitted Matérn
            ($\\nu = 1$) correlation, an effective length rather than the field's exact parameter.
        bin_distances (np.ndarray): Mean pair distance per bin, shape `(num_bins,)`; `nan` for
            empty bins.
        bin_correlations (np.ndarray): Mean correlation per bin, shape `(num_bins,)`.
        bin_counts (np.ndarray): Number of pairs per bin, shape `(num_bins,)`.
        fitted_correlations (np.ndarray): The fitted correlation at `bin_distances`.
        fit_mask (np.ndarray): Which bins entered the fit, shape `(num_bins,)`.
        base_vertices (np.ndarray): The base vertices used.
    """

    correlation_distance_threshold: float
    length: float
    bin_distances: np.ndarray[tuple[int], np.dtype[np.float64]]
    bin_correlations: np.ndarray[tuple[int], np.dtype[np.float64]]
    bin_counts: np.ndarray[tuple[int], np.dtype[np.int64]]
    fitted_correlations: np.ndarray[tuple[int], np.dtype[np.float64]]
    fit_mask: np.ndarray[tuple[int], np.dtype[np.bool_]]
    base_vertices: np.ndarray[tuple[int], np.dtype[np.int64]]


# ==================================================================================================
def _matern_correlation(
    distance: np.ndarray[tuple[int], np.dtype[np.float64]], length: float
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    r"""Matérn ($\nu = 1$) correlation $r K_1(r)$, $r = \sqrt{8}\, d / \ell$."""
    scaled_distance = np.sqrt(8.0) * np.asarray(distance, dtype=np.float64) / length
    safe_distance = np.maximum(scaled_distance, 1e-12)
    return np.where(scaled_distance > 1e-12, safe_distance * kv(1, safe_distance), 1.0)


# --------------------------------------------------------------------------------------------------
def _accumulate_axial_correlation_moments(
    samples: SampleStore,
    axial_mean: np.ndarray[tuple[int], np.dtype[np.float64]],
    base_vertices: np.ndarray[tuple[int], np.dtype[np.int64]],
    settings: CorrelationLengthSettings,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    r"""Correlation of $s = \sin(2(\theta - \mu))$ between base vertices and all vertices.

    Returns:
        np.ndarray: Correlations, shape `(num_base_vertices, num_components)`; `nan` where either
            vertex has zero variance.
    """
    num_samples, num_components = samples.shape
    num_kept = num_samples - settings.burn_in
    sum_sine_deviation = np.zeros(num_components)
    sum_sine_deviation_squared = np.zeros(num_components)
    sum_base_cross_products = np.zeros((base_vertices.size, num_components))
    for block in iterate_sample_blocks(samples, settings.burn_in, settings.block_size):
        sine_deviation = np.sin(2.0 * (block - axial_mean))
        sum_sine_deviation += sine_deviation.sum(axis=0)
        sum_sine_deviation_squared += (sine_deviation**2).sum(axis=0)
        sum_base_cross_products += sine_deviation[:, base_vertices].T @ sine_deviation
    mean_sine_deviation = sum_sine_deviation / num_kept
    variance_sine_deviation = sum_sine_deviation_squared / num_kept - mean_sine_deviation**2
    covariance = (
        sum_base_cross_products / num_kept
        - mean_sine_deviation[base_vertices, None] * mean_sine_deviation[None, :]
    )
    standard_deviation = np.sqrt(np.maximum(variance_sine_deviation, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        return covariance / (standard_deviation[base_vertices, None] * standard_deviation[None, :])


# --------------------------------------------------------------------------------------------------
def _find_crossing(
    distances: np.ndarray[tuple[int], np.dtype[np.float64]],
    correlations: np.ndarray[tuple[int], np.dtype[np.float64]],
    level: float,
) -> float:
    """Distance where the binned correlation first drops below `level` (linear interpolation)."""
    valid = np.flatnonzero(np.isfinite(distances) & np.isfinite(correlations))
    below = np.flatnonzero(correlations[valid] < level)
    if below.size == 0 or below[0] == 0:
        return float("nan")
    after = valid[below[0]]
    before = valid[below[0] - 1]
    fraction = (correlations[before] - level) / (correlations[before] - correlations[after])
    return float(distances[before] + fraction * (distances[after] - distances[before]))


# --------------------------------------------------------------------------------------------------
def _select_base_vertices(
    graph: sp.csr_array,
    boundary_vertices: np.ndarray[tuple[int], np.dtype[np.int64]],
    settings: CorrelationLengthSettings,
) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
    """Draw the base vertices at random among those at least `boundary_margin` from the boundary.

    Raises:
        ValueError: If no vertex is farther than `settings.boundary_margin` from the boundary.
    """
    num_vertices = graph.shape[0]
    if boundary_vertices.size > 0 and settings.boundary_margin > 0:
        boundary_distance = compute_graph_distances(
            graph, boundary_vertices, settings.boundary_margin, min_over_sources=True
        )
        # The search is limited to `boundary_margin`, so vertices farther away than that are left
        # at `inf`: the finite entries are exactly those within the margin of the boundary
        candidates = np.flatnonzero(~np.isfinite(boundary_distance))
    else:
        candidates = np.arange(num_vertices)
    if candidates.size == 0:
        raise ValueError(
            f"No vertex is farther than boundary_margin={settings.boundary_margin} from the "
            "boundary."
        )
    rng = np.random.default_rng(settings.seed)
    num_base_vertices = min(settings.num_base_vertices, candidates.size)
    return np.sort(rng.choice(candidates, size=num_base_vertices, replace=False))


# --------------------------------------------------------------------------------------------------
def _bin_pair_correlations(
    distance: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    correlation: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    settings: CorrelationLengthSettings,
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.int64]],
    np.ndarray[tuple[int], np.dtype[np.float64]],
    np.ndarray[tuple[int], np.dtype[np.float64]],
]:
    """Pool all vertex pairs (excluding each base vertex with itself) by distance bin.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Pair count, mean pair distance and mean
            correlation per bin, each of shape `(num_bins,)`; the means are `nan` for empty bins.
    """
    pair_mask = np.isfinite(distance) & np.isfinite(correlation) & (distance > 0)
    pair_distance = distance[pair_mask]
    pair_correlation = correlation[pair_mask]
    bin_index = np.minimum(
        (pair_distance / settings.max_distance * settings.num_bins).astype(np.int64),
        settings.num_bins - 1,
    )
    bin_counts = np.bincount(bin_index, minlength=settings.num_bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        bin_distances = (
            np.bincount(bin_index, weights=pair_distance, minlength=settings.num_bins) / bin_counts
        )
        bin_correlations = (
            np.bincount(bin_index, weights=pair_correlation, minlength=settings.num_bins)
            / bin_counts
        )
    return bin_counts, bin_distances, bin_correlations


# --------------------------------------------------------------------------------------------------
def _select_fit_bins(
    bin_counts: np.ndarray[tuple[int], np.dtype[np.int64]],
    bin_correlations: np.ndarray[tuple[int], np.dtype[np.float64]],
    num_kept: int,
    settings: CorrelationLengthSettings,
) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
    """Select the leading bins whose correlation is above the sampling noise floor.

    Raises:
        ValueError: If fewer than 2 bins are selected.
    """
    noise_floor = settings.noise_floor_scale / np.sqrt(num_kept)
    above_floor = (bin_counts > 0) & (bin_correlations > noise_floor)
    num_leading = above_floor.size if above_floor.all() else int(np.argmin(above_floor))
    fit_mask = np.zeros_like(above_floor)
    fit_mask[:num_leading] = above_floor[:num_leading]
    if fit_mask.sum() < 2:
        raise ValueError(
            "Fewer than 2 distance bins have correlation above the noise floor "
            f"({noise_floor:.3f}); increase the number of samples or reduce max_distance."
        )
    return fit_mask


# --------------------------------------------------------------------------------------------------
def _fit_matern_length(
    bin_distances: np.ndarray[tuple[int], np.dtype[np.float64]],
    bin_correlations: np.ndarray[tuple[int], np.dtype[np.float64]],
    bin_counts: np.ndarray[tuple[int], np.dtype[np.int64]],
    fit_mask: np.ndarray[tuple[int], np.dtype[np.bool_]],
    initial_length: float,
    max_distance: float,
) -> float:
    """Fit the Matérn length to the selected bins, weighting each by its pair count."""
    (length,), _ = curve_fit(
        _matern_correlation,
        bin_distances[fit_mask],
        bin_correlations[fit_mask],
        p0=[initial_length],
        sigma=1.0 / np.sqrt(bin_counts[fit_mask]),
        bounds=(1e-12 * max_distance, np.inf),
    )
    return float(length)


# --------------------------------------------------------------------------------------------------
def estimate_correlation_length(
    vertex_coordinates: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    connectivity: np.ndarray[tuple[int, int], np.dtype[np.integer]],
    samples: SampleStore,
    settings: CorrelationLengthSettings,
) -> CorrelationLengthResult:
    r"""Estimate the correlation distance of a fiber-angle field from samples.

    Args:
        vertex_coordinates (np.ndarray): Mesh vertex coordinates, shape `(num_vertices, dim)`.
        connectivity (np.ndarray): Triangle vertex indices, shape `(num_simplices, 3)`.
        samples (SampleStore): Angle samples in radians, shape `(num_samples, num_vertices)`, e.g.
            prior samples or an MCMC Zarr store. Read twice, block by block.
        settings (CorrelationLengthSettings): Estimation parameters.

    Raises:
        ValueError: If `samples` does not match the mesh, `settings.burn_in` leaves no samples, no
            vertex is farther than `settings.boundary_margin` from the boundary, or too few bins
            are above the noise floor to fit.

    Returns:
        CorrelationLengthResult: Model-free correlation distance threshold and fitted Matérn
            length, with the binned curve.
    """
    num_samples, num_vertices = samples.shape
    if num_vertices != vertex_coordinates.shape[0]:
        raise ValueError(
            f"samples have {num_vertices} components, but the mesh has "
            f"{vertex_coordinates.shape[0]} vertices."
        )

    graph = assemble_edge_length_graph(vertex_coordinates, connectivity)
    base_vertices = _select_base_vertices(graph, find_boundary_vertices(connectivity), settings)

    axial_mean, _ = compute_axial_mean_and_variance(samples, settings.burn_in, settings.block_size)
    correlation = _accumulate_axial_correlation_moments(
        samples, axial_mean, base_vertices, settings
    )
    distance = compute_graph_distances(graph, base_vertices, settings.max_distance)

    bin_counts, bin_distances, bin_correlations = _bin_pair_correlations(
        distance, correlation, settings
    )
    fit_mask = _select_fit_bins(
        bin_counts, bin_correlations, num_samples - settings.burn_in, settings
    )

    correlation_distance_threshold = _find_crossing(
        bin_distances, bin_correlations, settings.correlation_threshold
    )
    initial_length = (
        correlation_distance_threshold * np.sqrt(8.0) / 2.0
        if np.isfinite(correlation_distance_threshold)
        else settings.max_distance
    )
    length = _fit_matern_length(
        bin_distances, bin_correlations, bin_counts, fit_mask, initial_length, settings.max_distance
    )
    return CorrelationLengthResult(
        correlation_distance_threshold=correlation_distance_threshold,
        length=length,
        bin_distances=bin_distances,
        bin_correlations=bin_correlations,
        bin_counts=bin_counts,
        fitted_correlations=_matern_correlation(bin_distances, length),
        fit_mask=fit_mask,
        base_vertices=base_vertices,
    )
