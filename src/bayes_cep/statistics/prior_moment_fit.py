r"""Estimation of the prior parameters $\kappa$ and $\tau$ from one observed fiber-angle field.

The fiber angle is only known modulo $\pi$, and the prior is a Gaussian on real-valued angles whose
branch is unknown, so the likelihood of the field is not available. The parameters are instead
fitted by simulated moments: summary statistics that do not change under $\theta \to \theta + \pi$
are computed for the observed field and for samples of the prior on the same mesh, and the pair
$(\kappa, \tau)$ on a grid is scored by how far the statistics of its samples are from the observed
ones.

The statistics are, for the doubled angle $e^{2i\theta}$ of the simplices,

- the correlation $C_k$ = mean of $\cos(2(\theta_i - \theta_j))$ over random simplex pairs whose
  centroid distance (Euclidean or geodesic, in mesh units) is in the distance bin $k$, which is one
  for equal fibers and zero for independent ones, and
- the concentration $R = |\operatorname{mean} e^{2i\theta}|$ over all simplices.

$C_k$ as a function of the distance mainly tells the correlation length (set by $\kappa$), and $R$
the variance (set by $\kappa\tau$). The loss is

$$\sum_{k < K} (C_k^{\text{sim}} - C_k^{\text{obs}})^2 + w\,(R^{\text{sim}} - R^{\text{obs}})^2,$$

with the samples averaged before the comparison, over the first $K$ distance bins (the outer bins
are noisy and affected by the mesh boundary). A field smoothed by `smooth_axial_angle_field` is
compared with prior samples smoothed in the same way (`smoothing_scale`), which removes the
roughness a Bilaplacian prior cannot produce.

A single field has to be treated as many realizations (stationarity). The distance is only a label
of the pairs (observed and simulated statistics are binned alike, so no closed-form correlation
function is assumed): the Euclidean distance mislabels pairs that are close in space but far along
the surface, the geodesic one is meaningful only over short distances. The grid is discrete: the
loss is flat along a valley of nearly constant $\kappa\tau$, so look at the whole loss surface, not
only its minimum.

Classes:
    DistanceMetric: Distance used to bin the simplex pairs.
    MomentFitSettings: Parameters of the statistics and the simulation.
    AxialMoments: Correlation per distance bin and concentration of a field.
    MomentFitResult: Loss and statistics on the parameter grid, with I/O.

Functions:
    sample_simplex_pairs: Random simplex pairs.
    compute_pair_distances: Distance of simplex pairs.
    sample_distance_pairs: Random simplex pairs and their distance bins.
    compute_axial_moments: The statistics of a per-simplex angle field.
    fit_prior_parameters: Score a grid of $(\kappa, \tau)$ against an observed field.
    pool_losses: Sum the losses of several fields, e.g. the other patients of an ensemble.
"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Self

import numpy as np
import pyvista as pv

from bayes_cep.mesh.geodesic import assemble_centroid_edge_length_graph, compute_graph_distances
from bayes_cep.mesh.interpolation import InterpolationStrategy
from bayes_cep.mesh.io import create_dolfinx_mesh
from bayes_cep.mesh.topology import get_triangle_connectivity
from bayes_cep.posterior.prior import PriorParameters, build_fiber_angle_prior
from bayes_cep.statistics.axial_smoothing import smooth_axial_angle_field


# ==================================================================================================
class DistanceMetric(StrEnum):
    """Distance between the centroids of two simplices.

    Attributes:
        EUCLIDEAN: Straight line in space.
        GEODESIC: Shortest path along the surface (mesh graph with centroid nodes, see
            [`assemble_centroid_edge_length_graph`][bayes_cep.mesh.geodesic.assemble_centroid_edge_length_graph]).
    """

    EUCLIDEAN = "euclidean"
    GEODESIC = "geodesic"


# ==================================================================================================
@dataclass(frozen=True)
class MomentFitSettings:
    """Parameters of the statistics and of the simulation.

    There are no defaults for the scientific parameters: the reference values are in
    `single_runs.reference`.

    Attributes:
        distance_bin_edges (tuple[float, ...]): Increasing edges of the distance bins, in mesh
            units; the last one should exceed the extent of the mesh.
        num_fitted_bins (int): Number of leading distance bins in the loss.
        num_pairs (int): Number of random simplex pairs for the correlation.
        distance_metric (DistanceMetric): Distance that bins the pairs.
        num_source_simplices (int): Only for the geodesic distance: the pairs are
            `num_pairs // num_source_simplices` random partners of each of this many random
            simplices, since one shortest-path search per source gives the distance to all partners.
        concentration_weight (float): Weight $w$ of the concentration in the loss.
        num_prior_samples (int): Number of prior samples per parameter pair.
        seed (int): Seed of the pairs and of the prior samples.
        smoothing_scale (float): Scale of the smoothing applied to every prior sample before the
            statistics (`0` for none); the observed field must be smoothed in the same way.
    """

    distance_bin_edges: tuple[float, ...]
    num_fitted_bins: int
    num_pairs: int
    distance_metric: DistanceMetric
    num_source_simplices: int
    concentration_weight: float
    num_prior_samples: int
    seed: int
    smoothing_scale: float


# ==================================================================================================
@dataclass(frozen=True)
class AxialMoments:
    """The statistics of a per-simplex angle field.

    Attributes:
        correlation (np.ndarray): Mean $\\cos 2\\Delta$ per distance bin, shape `(num_bins,)`; `nan`
            for a bin without pairs.
        concentration (float): The concentration $R$.
    """

    correlation: np.ndarray
    concentration: float


# ==================================================================================================
@dataclass(frozen=True)
class MomentFitResult:
    """The loss and the statistics on the parameter grid.

    Attributes:
        kappas (np.ndarray): Values of $\\kappa$ (rows), shape `(num_kappas,)`.
        taus (np.ndarray): Values of $\\tau$ (columns), shape `(num_taus,)`.
        loss (np.ndarray): Loss, shape `(num_kappas, num_taus)`.
        correlation (np.ndarray): Mean correlation of the samples, shape
            `(num_kappas, num_taus, num_bins)`.
        concentration (np.ndarray): Mean concentration of the samples, shape
            `(num_kappas, num_taus)`.
        observed_correlation (np.ndarray): Correlation of the observed field, shape `(num_bins,)`.
        observed_concentration (float): Concentration of the observed field.
        distance_bin_edges (np.ndarray): Edges of the distance bins, shape `(num_bins + 1,)`.
    """

    kappas: np.ndarray
    taus: np.ndarray
    loss: np.ndarray
    correlation: np.ndarray
    concentration: np.ndarray
    observed_correlation: np.ndarray
    observed_concentration: float
    distance_bin_edges: np.ndarray

    # ----------------------------------------------------------------------------------------------
    def best_parameters(self) -> tuple[float, float]:
        """The $(\\kappa, \\tau)$ of the smallest loss."""
        row, column = np.unravel_index(np.argmin(self.loss), self.loss.shape)
        return float(self.kappas[row]), float(self.taus[column])

    # ----------------------------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        """Write the result as one `.npz` file."""
        np.savez(
            path,
            kappas=self.kappas,
            taus=self.taus,
            loss=self.loss,
            correlation=self.correlation,
            concentration=self.concentration,
            observed_correlation=self.observed_correlation,
            observed_concentration=self.observed_concentration,
            distance_bin_edges=self.distance_bin_edges,
        )

    # ----------------------------------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> Self:
        """Read a result written by `save`."""
        with np.load(path) as data:
            return cls(
                kappas=data["kappas"],
                taus=data["taus"],
                loss=data["loss"],
                correlation=data["correlation"],
                concentration=data["concentration"],
                observed_correlation=data["observed_correlation"],
                observed_concentration=float(data["observed_concentration"]),
                distance_bin_edges=data["distance_bin_edges"],
            )


# ==================================================================================================
def sample_simplex_pairs(
    mesh: pv.UnstructuredGrid, settings: MomentFitSettings
) -> tuple[np.ndarray, np.ndarray]:
    """Draw random simplex pairs.

    Args:
        mesh (pv.UnstructuredGrid): Triangular surface mesh.
        settings (MomentFitSettings): Number of pairs, metric, number of sources and seed.

    Returns:
        tuple[np.ndarray, np.ndarray]: The two simplex indices of every pair, each of shape
            `(num_pairs,)` (`num_pairs` rounded down to a multiple of `num_source_simplices` for the
            geodesic distance).

    Raises:
        ValueError: If the geodesic distance is used with more sources than pairs or simplices.
    """
    num_simplices = get_triangle_connectivity(mesh).shape[0]
    rng = np.random.default_rng(settings.seed)
    if settings.distance_metric is DistanceMetric.EUCLIDEAN:
        return (
            rng.integers(0, num_simplices, settings.num_pairs),
            rng.integers(0, num_simplices, settings.num_pairs),
        )
    num_sources = settings.num_source_simplices
    if not 0 < num_sources <= min(settings.num_pairs, num_simplices):
        raise ValueError(
            f"num_source_simplices must be in (0, min(num_pairs, num_simplices)], got {num_sources}"
        )
    sources = rng.choice(num_simplices, num_sources, replace=False)
    partners_per_source = settings.num_pairs // num_sources
    return (
        np.repeat(sources, partners_per_source),
        rng.integers(0, num_simplices, num_sources * partners_per_source),
    )


# ==================================================================================================
def compute_pair_distances(
    mesh: pv.UnstructuredGrid,
    first: np.ndarray[tuple[int], np.dtype[np.integer]],
    second: np.ndarray[tuple[int], np.dtype[np.integer]],
    metric: DistanceMetric,
    sources_per_search_batch: int = 50,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Compute the distance between the centroids of simplex pairs.

    Args:
        mesh (pv.UnstructuredGrid): Triangular surface mesh.
        first (np.ndarray): First simplex of every pair, shape `(num_pairs,)`.
        second (np.ndarray): Second simplex of every pair, shape `(num_pairs,)`.
        metric (DistanceMetric): The distance. The geodesic one runs one shortest-path search per
            distinct simplex in `first`, so `first` should contain few distinct simplices.
        sources_per_search_batch (int): Searches run together, which bounds the memory of the
            distance matrix to `sources_per_search_batch * (num_vertices + num_simplices)`.

    Returns:
        np.ndarray: Distances in mesh units, shape `(num_pairs,)`; `inf` for unconnected pairs.
    """
    connectivity = get_triangle_connectivity(mesh)
    if metric is DistanceMetric.EUCLIDEAN:
        centroids = mesh.points[connectivity].mean(axis=1)
        return np.linalg.norm(centroids[first] - centroids[second], axis=1)
    graph = assemble_centroid_edge_length_graph(mesh.points, connectivity)
    centroid_node_offset = mesh.n_points
    distances = np.empty(first.shape[0])
    unique_sources, inverse = np.unique(first, return_inverse=True)
    for start in range(0, unique_sources.shape[0], sources_per_search_batch):
        batch = slice(start, start + sources_per_search_batch)
        from_batch = compute_graph_distances(graph, centroid_node_offset + unique_sources[batch])
        in_batch = np.flatnonzero((inverse >= batch.start) & (inverse < batch.stop))
        distances[in_batch] = from_batch[
            inverse[in_batch] - batch.start, centroid_node_offset + second[in_batch]
        ]
    return distances


# ==================================================================================================
def sample_distance_pairs(
    mesh: pv.UnstructuredGrid, settings: MomentFitSettings
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw random simplex pairs and assign them to distance bins.

    Args:
        mesh (pv.UnstructuredGrid): Triangular surface mesh.
        settings (MomentFitSettings): Number of pairs, metric, bin edges and seed.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The two simplex indices of every pair and the
            index of its distance bin (centroid distance), each of shape `(num_pairs,)`.
    """
    first, second = sample_simplex_pairs(mesh, settings)
    distances = compute_pair_distances(mesh, first, second, settings.distance_metric)
    bins = np.digitize(distances, settings.distance_bin_edges) - 1
    return first, second, bins


# ==================================================================================================
def compute_axial_moments(
    simplex_angles: np.ndarray[tuple[int], np.dtype[np.float64]],
    pairs: tuple[np.ndarray, np.ndarray, np.ndarray],
    num_bins: int,
) -> AxialMoments:
    """Compute the statistics of a per-simplex angle field.

    Args:
        simplex_angles (np.ndarray): Angles in radians, shape `(num_simplices,)`.
        pairs (tuple[np.ndarray, np.ndarray, np.ndarray]): Output of `sample_distance_pairs`.
        num_bins (int): Number of distance bins.

    Returns:
        AxialMoments: Correlation per bin and concentration.
    """
    first, second, bins = pairs
    cosine = np.cos(2.0 * (simplex_angles[first] - simplex_angles[second]))
    in_range = (bins >= 0) & (bins < num_bins)
    sums = np.bincount(bins[in_range], weights=cosine[in_range], minlength=num_bins)
    counts = np.bincount(bins[in_range], minlength=num_bins)
    correlation = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    concentration = float(np.abs(np.mean(np.exp(2j * simplex_angles))))
    return AxialMoments(correlation, concentration)


# ==================================================================================================
def fit_prior_parameters(
    mesh: pv.UnstructuredGrid,
    observed_simplex_angles: np.ndarray[tuple[int], np.dtype[np.float64]],
    kappas: tuple[float, ...],
    taus: tuple[float, ...],
    settings: MomentFitSettings,
    interpolation: InterpolationStrategy,
) -> MomentFitResult:
    """Score a grid of $(\\kappa, \\tau)$ against an observed per-simplex angle field.

    Args:
        mesh (pv.UnstructuredGrid): Triangular surface mesh of the field.
        observed_simplex_angles (np.ndarray): The observed field in radians, shape
            `(num_simplices,)`. With `settings.smoothing_scale > 0` it must have been smoothed with
            that scale already, from vertex angles moved to the simplices with `interpolation`.
        kappas (tuple[float, ...]): Values of $\\kappa$ to try.
        taus (tuple[float, ...]): Values of $\\tau$ to try.
        settings (MomentFitSettings): Statistics and simulation parameters.
        interpolation (InterpolationStrategy): Vertex-to-simplex map of the samples; use the one
            of the forward model.

    Returns:
        MomentFitResult: The loss and the statistics of the samples on the grid.
    """
    num_bins = len(settings.distance_bin_edges) - 1
    connectivity = get_triangle_connectivity(mesh)
    vertex_to_simplex = interpolation.assemble_matrix(mesh.points, connectivity)
    pairs = sample_distance_pairs(mesh, settings)
    observed = compute_axial_moments(observed_simplex_angles, pairs, num_bins)
    dolfinx_mesh = create_dolfinx_mesh(mesh)

    fitted = slice(0, settings.num_fitted_bins)
    loss = np.zeros((len(kappas), len(taus)))
    correlation = np.zeros((len(kappas), len(taus), num_bins))
    concentration = np.zeros((len(kappas), len(taus)))
    for row, kappa in enumerate(kappas):
        for column, tau in enumerate(taus):
            prior = build_fiber_angle_prior(
                dolfinx_mesh,
                np.zeros(mesh.n_points),
                PriorParameters(kappa=kappa, tau=tau, seed=settings.seed),
            )
            sample_moments = []
            for _ in range(settings.num_prior_samples):
                sample = prior.generate_sample()
                if settings.smoothing_scale > 0.0:
                    sample, _ = smooth_axial_angle_field(mesh, sample, settings.smoothing_scale)
                sample_moments.append(
                    compute_axial_moments(vertex_to_simplex @ sample, pairs, num_bins)
                )
            correlation[row, column] = np.nanmean([m.correlation for m in sample_moments], axis=0)
            concentration[row, column] = np.mean([m.concentration for m in sample_moments])
            loss[row, column] = (
                np.nansum((correlation[row, column, fitted] - observed.correlation[fitted]) ** 2)
                + settings.concentration_weight
                * (concentration[row, column] - observed.concentration) ** 2
            )
    return MomentFitResult(
        kappas=np.array(kappas),
        taus=np.array(taus),
        loss=loss,
        correlation=correlation,
        concentration=concentration,
        observed_correlation=observed.correlation,
        observed_concentration=observed.concentration,
        distance_bin_edges=np.array(settings.distance_bin_edges),
    )


# ==================================================================================================
def pool_losses(
    results: list[MomentFitResult],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Sum the loss surfaces of several fields on their common grid.

    The minimum of the sum is the single pair that fits all fields best, e.g. the estimate from
    the other patients of an ensemble for a held-out patient.

    Args:
        results (list[MomentFitResult]): Results on the same $(\\kappa, \\tau)$ grid.

    Returns:
        np.ndarray: The summed loss, shape `(num_kappas, num_taus)`.

    Raises:
        ValueError: If the grids differ.
    """
    for result in results[1:]:
        if not (
            np.array_equal(result.kappas, results[0].kappas)
            and np.array_equal(result.taus, results[0].taus)
        ):
            raise ValueError("The results are on different parameter grids.")
    return np.sum([result.loss for result in results], axis=0)
