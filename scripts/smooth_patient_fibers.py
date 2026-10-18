"""Smooth patient fiber fields and store the result.

For every given fiber field (`.npy`, per-simplex 3D vectors) of the mesh in `--mesh-dir`
(`mesh.vtu` and `basis_vecs.npy`), writes into `--out-dir` (default: `smoothed_scale_<scale>` next
to the fiber field):

- the smoothed fiber field, under the name of the input (a drop-in replacement of it),
- `angle_` + the name with `fiber_field` replaced: the smoothed per-vertex angles [rad],
- `coherence_` + the same: the per-vertex coherence in [0, 1]; the angle is not meaningful where it
  is near zero.

Usage: python scripts/smooth_patient_fibers.py --mesh-dir DIR [--scale 3] [--out-dir DIR]
    FIBER_FIELD [FIBER_FIELD ...]

The scale is the standard deviation of the Gaussian smoothing, in the units of the mesh coordinates.
"""

import argparse
from pathlib import Path

import numpy as np

from bayes_cep.mesh.io import load_pyvista_mesh
from bayes_cep.statistics.axial_smoothing import smooth_fiber_field


# ==================================================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fiber_fields", nargs="+", type=Path)
    parser.add_argument("--mesh-dir", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--out-dir", type=Path, default=None)
    arguments = parser.parse_args()

    mesh = load_pyvista_mesh(arguments.mesh_dir / "mesh.vtu")
    basis_vectors = np.load(arguments.mesh_dir / "basis_vecs.npy")
    for path in arguments.fiber_fields:
        out_dir = arguments.out_dir or path.parent / f"smoothed_scale_{arguments.scale:g}"
        out_dir.mkdir(parents=True, exist_ok=True)
        smoothed, angles, coherence = smooth_fiber_field(
            mesh, np.load(path), basis_vectors, arguments.scale
        )
        np.save(out_dir / path.name, smoothed)
        np.save(out_dir / path.name.replace("fiber_field", "angle_field"), angles)
        np.save(out_dir / path.name.replace("fiber_field", "coherence"), coherence)
        print(f"{path} -> {out_dir} (median coherence {np.median(coherence):.2f})")


if __name__ == "__main__":
    main()
