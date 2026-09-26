"""Mesh loading and format conversion utilities.

Functions:
    load_pyvista_mesh: Load and validate a triangular surface mesh from disk.
    create_dolfinx_mesh: Convert a pyvista mesh to a dolfinx mesh, in matching vertex order.
"""

from pathlib import Path

import basix
import dolfinx as dlx
import numpy as np
import pyvista as pv
import ufl
from mpi4py import MPI


# ==================================================================================================
def load_pyvista_mesh(mesh_path: Path) -> pv.UnstructuredGrid:
    """Load and validate a triangular surface mesh from disk.

    Args:
        mesh_path (Path): Path to the mesh file (e.g. `.vtu`), readable by pyvista.

    Raises:
        ValueError: If the mesh contains non-triangular cells.

    Returns:
        pv.UnstructuredGrid: Loaded mesh.
    """
    mesh = pv.read(str(mesh_path))
    if not np.all(mesh.celltypes == pv.CellType.TRIANGLE):
        raise ValueError("Only triangular meshes are supported.")
    return mesh


# --------------------------------------------------------------------------------------------------
def create_dolfinx_mesh(pv_mesh: pv.UnstructuredGrid) -> dlx.mesh.Mesh:
    """Convert a pyvista mesh to a dolfinx mesh, in matching (input-node) vertex order.

    Assumes `pv_mesh` has already been validated as purely triangular, e.g. via
    [`load_pyvista_mesh`][bayes_cep.mesh.io.load_pyvista_mesh].

    Args:
        pv_mesh (pv.UnstructuredGrid): Triangular surface mesh.

    Returns:
        dlx.mesh.Mesh: Equivalent dolfinx mesh, for use with `ls_bayesian.spde_prior`.
    """
    points = pv_mesh.points
    cells = pv_mesh.cells.reshape(-1, 4)[:, 1:]
    ufl_cell_type = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(3,)))
    return dlx.mesh.create_mesh(MPI.COMM_WORLD, cells, points, ufl_cell_type)
