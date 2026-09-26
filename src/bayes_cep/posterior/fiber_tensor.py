r"""Anisotropic conductivity tensor field for cardiac fiber orientation.

Classes:
    FiberTensorSettings: Settings for `FiberTensor`.
    FiberTensor: Per-simplex conductivity tensor from a fiber-orientation angle.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from eikonax import tensorfield


# ==================================================================================================
@dataclass
class FiberTensorSettings:
    """Settings for `FiberTensor`.

    Attributes:
        dimension (int): Spatial dimension of the tensor (3 for a surface mesh embedded in 3D).
        basis_vectors_one (jax.Array | npt.NDArray): First local tangent-plane basis vector per
            simplex, shape `(num_simplices, dimension)`.
        basis_vectors_two (jax.Array | npt.NDArray): Second local tangent-plane basis vector per
            simplex, shape `(num_simplices, dimension)`.
        longitudinal_velocities (jax.Array | npt.NDArray): Conduction velocity along the fiber
            direction, per simplex, shape `(num_simplices,)`.
        transversal_velocities (jax.Array | npt.NDArray): Conduction velocity transverse to the
            fiber direction, per simplex, shape `(num_simplices,)`.
    """

    dimension: int
    basis_vectors_one: jax.Array | npt.NDArray[np.float64]
    basis_vectors_two: jax.Array | npt.NDArray[np.float64]
    longitudinal_velocities: jax.Array | npt.NDArray[np.float64]
    transversal_velocities: jax.Array | npt.NDArray[np.float64]


# ==================================================================================================
class FiberTensor(tensorfield.AbstractSimplexTensor):
    r"""Per-simplex conductivity tensor from a fiber-orientation angle.

    The parameter for each simplex is a scalar rotation angle $\theta$ within the simplex's local
    tangent-plane basis $(e_1, e_2)$. The fiber direction is
    $f = \cos(\theta) e_1 + \sin(\theta) e_2$, and the transverse in-plane direction is
    $f^\perp = -\sin(\theta) e_1 + \cos(\theta) e_2$. The (metric, i.e. inverse-conductivity) tensor
    eikonax expects is

    $$
    \mathbf{M} = \frac{1}{v_\parallel^2} f f^T + \frac{1}{v_\perp^2} f^\perp {f^\perp}^T,
    $$

    for longitudinal/transversal conduction velocities $v_\parallel, v_\perp$.
    """

    _basis_vectors_one: jax.Array
    _basis_vectors_two: jax.Array
    _longitudinal_velocities: jax.Array
    _transversal_velocities: jax.Array

    # ----------------------------------------------------------------------------------------------
    def __init__(self, settings: FiberTensorSettings) -> None:
        """Initialize the tensor field from per-simplex basis vectors and conduction velocities."""
        self.dimension = settings.dimension
        self._basis_vectors_one = jnp.array(settings.basis_vectors_one, dtype=jnp.float32)
        self._basis_vectors_two = jnp.array(settings.basis_vectors_two, dtype=jnp.float32)
        self._longitudinal_velocities = jnp.array(
            settings.longitudinal_velocities, dtype=jnp.float32
        )
        self._transversal_velocities = jnp.array(settings.transversal_velocities, dtype=jnp.float32)

    # ----------------------------------------------------------------------------------------------
    def assemble(
        self,
        simplex_ind: jax.Array,
        parameter: jax.Array,
    ) -> jax.Array:
        """Assemble the conductivity tensor for one simplex from its fiber angle."""
        e_1 = self._basis_vectors_one[simplex_ind]
        e_2 = self._basis_vectors_two[simplex_ind]
        v_long = self._longitudinal_velocities[simplex_ind]
        v_trans = self._transversal_velocities[simplex_ind]
        fiber_direction = jnp.cos(parameter) * e_1 + jnp.sin(parameter) * e_2
        transverse_direction = -jnp.sin(parameter) * e_1 + jnp.cos(parameter) * e_2
        tensor = 1 / jnp.square(v_long) * jnp.outer(
            fiber_direction, fiber_direction
        ) + 1 / jnp.square(v_trans) * jnp.outer(transverse_direction, transverse_direction)
        return tensor

    # ----------------------------------------------------------------------------------------------
    def derivative(
        self,
        simplex_ind: jax.Array,
        parameter: jax.Array,
    ) -> jax.Array:
        """Parametric derivative of `assemble`, via forward-mode automatic differentiation."""
        return jax.jacfwd(self.assemble, argnums=1)(simplex_ind, parameter)
