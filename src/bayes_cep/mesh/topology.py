"""Topology of triangular surface meshes.

Functions:
    get_triangle_connectivity: Vertex indices of the triangles of a triangular surface mesh.
"""

import numpy as np
import pyvista as pv


# ==================================================================================================
def get_triangle_connectivity(
    pv_mesh: pv.UnstructuredGrid,
) -> np.ndarray[tuple[int, int], np.dtype[np.integer]]:
    """Vertex indices of the triangles of a triangular surface mesh.

    Assumes `pv_mesh` has already been validated as purely triangular, e.g. via
    [`load_pyvista_mesh`][bayes_cep.mesh.io.load_pyvista_mesh].

    Args:
        pv_mesh (pv.UnstructuredGrid): Triangular surface mesh.

    Returns:
        np.ndarray: Connectivity, shape `(num_simplices, 3)`.
    """
    return pv_mesh.cells.reshape(-1, 4)[:, 1:]
