"""Estimate kappa and tau of the SPDE prior from patient fiber fields by simulated moments.

For every patient directory in `--patient-data` (`patient_01`, ... with `mesh.vtu`, `basis_vecs.npy`
and `fiber_field.npy`), scores the grid `MOMENT_FIT_KAPPAS` x `MOMENT_FIT_TAUS` of
`single_runs.reference` (see `bayes_cep.statistics.prior_moment_fit`) and writes
`<out-dir>/<patient>.npz` (a `MomentFitResult`).

With `--smoothing-scale 0` the raw fiber field is fitted. With a positive scale, the smoothed field
of `<patient>/smoothed_scale_<scale>/angle_field.npy` (see `scripts/smooth_patient_fibers.py`) is
fitted against prior samples smoothed with the same scale.

Usage: python scripts/fit_prior_parameters.py [--patient-data DIR] [--smoothing-scale 0]
    [--distance-metric euclidean|geodesic] [--out-dir DIR] [--patients patient_01 ...]

The default output directory is `<patient-data>/prior_fits/raw` or `.../smoothed_scale_<scale>`,
with the suffix `_geodesic` for the geodesic distance.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from bayes_cep.mesh.interpolation import NearestNeighborInterpolationStrategy
from bayes_cep.mesh.io import load_pyvista_mesh
from bayes_cep.mesh.topology import get_triangle_connectivity
from bayes_cep.preprocessing.angle_field import compute_angle_from_fiber_vectors
from bayes_cep.statistics.prior_moment_fit import DistanceMetric, fit_prior_parameters
from single_runs.reference import MOMENT_FIT_KAPPAS, MOMENT_FIT_TAUS, reference_moment_fit_settings


# ==================================================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--patient-data", type=Path, default=Path("patient_data"))
    parser.add_argument("--smoothing-scale", type=float, default=0.0)
    parser.add_argument(
        "--distance-metric", type=DistanceMetric, choices=list(DistanceMetric), required=True
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--patients", nargs="+", default=[f"patient_0{i}" for i in range(1, 8)])
    arguments = parser.parse_args()

    scale = arguments.smoothing_scale
    version = "raw" if scale == 0.0 else f"smoothed_scale_{scale:g}"
    suffix = "_geodesic" if arguments.distance_metric is DistanceMetric.GEODESIC else ""
    out_dir = arguments.out_dir or arguments.patient_data / "prior_fits" / (version + suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = reference_moment_fit_settings(scale, arguments.distance_metric)
    interpolation = NearestNeighborInterpolationStrategy()

    for patient in arguments.patients:
        start = time.time()
        patient_dir = arguments.patient_data / patient
        mesh = load_pyvista_mesh(patient_dir / "mesh.vtu")
        if scale == 0.0:
            basis_vectors = np.load(patient_dir / "basis_vecs.npy")
            observed = compute_angle_from_fiber_vectors(
                np.load(patient_dir / "fiber_field.npy"),
                basis_vectors[..., 0],
                basis_vectors[..., 1],
            )
        else:
            vertex_angles = np.load(patient_dir / version / "angle_field.npy")
            observed = interpolation.assemble_matrix(
                mesh.points, get_triangle_connectivity(mesh)
            ) @ (vertex_angles)
        result = fit_prior_parameters(
            mesh, observed, MOMENT_FIT_KAPPAS, MOMENT_FIT_TAUS, settings, interpolation
        )
        result.save(out_dir / f"{patient}.npz")
        kappa, tau = result.best_parameters()
        print(
            f"{patient}: best kappa {kappa:g}, tau {tau:g}, loss {result.loss.min():.3f} "
            f"({time.time() - start:.0f} s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
