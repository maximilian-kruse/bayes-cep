"""Compare Euclidean and geodesic centroid distances of the pairs used by the prior moment fit.

For random simplex pairs (geodesic design of `sample_simplex_pairs`), prints per Euclidean distance
bin the share of pairs whose geodesic distance exceeds the Euclidean one by more than a factor
`--ratio-threshold`, and the median and 90th percentile of the ratio geodesic / Euclidean. A ratio
far above one means the Euclidean label mixes pairs that are far apart along the surface.

Usage: python scripts/check_distance_metrics.py [--patient-data DIR] [--patients patient_01 ...]
    [--ratio-threshold 1.25]
"""

import argparse
from pathlib import Path

import numpy as np

from bayes_cep.mesh.io import load_pyvista_mesh
from bayes_cep.statistics.prior_moment_fit import (
    DistanceMetric,
    compute_pair_distances,
    sample_simplex_pairs,
)
from single_runs.reference import MOMENT_FIT_DISTANCE_BIN_EDGES, reference_moment_fit_settings


# ==================================================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--patient-data", type=Path, default=Path("patient_data"))
    parser.add_argument("--patients", nargs="+", default=["patient_01"])
    parser.add_argument("--ratio-threshold", type=float, default=1.25)
    arguments = parser.parse_args()

    settings = reference_moment_fit_settings(0.0, DistanceMetric.GEODESIC)
    edges = np.array(MOMENT_FIT_DISTANCE_BIN_EDGES)
    for patient in arguments.patients:
        mesh = load_pyvista_mesh(arguments.patient_data / patient / "mesh.vtu")
        first, second = sample_simplex_pairs(mesh, settings)
        euclidean = compute_pair_distances(mesh, first, second, DistanceMetric.EUCLIDEAN)
        geodesic = compute_pair_distances(mesh, first, second, DistanceMetric.GEODESIC)
        keep = euclidean > 0.0
        ratio = geodesic[keep] / euclidean[keep]
        euclidean_bin = np.digitize(euclidean[keep], edges) - 1
        geodesic_bin = np.digitize(geodesic[keep], edges) - 1
        print(f"{patient}: {first.size} pairs, {np.isinf(geodesic).sum()} unconnected")
        print("  Euclidean bin   pairs   median ratio   p90 ratio   share ratio > threshold")
        for k in range(len(edges) - 1):
            in_bin = euclidean_bin == k
            if not in_bin.any():
                continue
            print(
                f"  [{edges[k]:5.1f},{edges[k + 1]:5.1f})  {in_bin.sum():6d}   "
                f"{np.median(ratio[in_bin]):8.2f}   {np.percentile(ratio[in_bin], 90):8.2f}   "
                f"{np.mean(ratio[in_bin] > arguments.ratio_threshold):8.2f}"
            )
        changes_bin = euclidean_bin != geodesic_bin
        fitted = geodesic_bin < settings.num_fitted_bins
        print(
            f"  share of pairs changing bin: {np.mean(changes_bin):.3f} "
            f"(within the fitted bins: {np.mean(changes_bin[fitted]):.3f})"
        )


if __name__ == "__main__":
    main()
