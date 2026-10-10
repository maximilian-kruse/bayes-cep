import numpy as np
import pytest

from bayes_cep.statistics.axial_statistics import wrap_axial_angles

pytestmark = pytest.mark.unit


# ==================================================================================================
def test_wrap_axial_angles_maps_onto_the_principal_branch_and_keeps_the_orientation() -> None:
    angles = np.array([-3.0, -1.0, 0.0, 0.7, np.pi / 2, 2.0, 7.5])

    wrapped = wrap_axial_angles(angles)

    assert np.all(wrapped > -np.pi / 2 - 1e-12)
    assert np.all(wrapped <= np.pi / 2 + 1e-12)
    # Same orientation: the difference is a multiple of pi.
    multiples = (angles - wrapped) / np.pi
    np.testing.assert_allclose(multiples, np.round(multiples), atol=1e-12)


# --------------------------------------------------------------------------------------------------
def test_wrap_axial_angles_leaves_angles_on_the_branch_unchanged() -> None:
    angles = np.linspace(-1.5, 1.5, 7)

    np.testing.assert_allclose(wrap_axial_angles(angles), angles, atol=1e-14)
