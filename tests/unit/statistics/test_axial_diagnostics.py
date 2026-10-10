from pathlib import Path

import numpy as np
import pytest
from ls_bayesian.mcmc.diagnostics import estimate_effective_sample_size
from ls_bayesian.mcmc.storage import ZarrStorage, open_zarr_samples

from bayes_cep.statistics.axial_diagnostics import compute_axial_effective_sample_size
from bayes_cep.statistics.sample_blocks import MappedSampleStore
from tests.helpers import generate_ar1_chain

pytestmark = pytest.mark.unit

AUTOREGRESSION_COEFFICIENT = 0.8
ANALYTIC_AUTOCORRELATION_TIME = (1 + AUTOREGRESSION_COEFFICIENT) / (1 - AUTOREGRESSION_COEFFICIENT)
ANGLE_STANDARD_DEVIATION = 0.1


# ==================================================================================================
def make_angle_chain(
    axial_mean: float, num_samples: int, num_vertices: int, rng: np.random.Generator
) -> np.ndarray:
    """AR(1) angle fluctuations of small amplitude around `axial_mean`, not folded in any branch."""
    return axial_mean + ANGLE_STANDARD_DEVIATION * generate_ar1_chain(
        AUTOREGRESSION_COEFFICIENT, num_samples, num_vertices, rng
    )


# ==================================================================================================
def test_mapped_sample_store_applies_the_transform_to_every_slice() -> None:
    base = np.arange(12.0).reshape(6, 2)

    store_under_test = MappedSampleStore(base, lambda block: 2 * block + 1)

    assert store_under_test.shape == (6, 2)
    np.testing.assert_array_equal(store_under_test[1:4], 2 * base[1:4] + 1)


# ==================================================================================================
def test_axial_autocorrelation_time_of_a_concentrated_ar1_angle_matches_analytic_value() -> None:
    chain = make_angle_chain(0.3, 100_000, 3, np.random.default_rng(0))

    result = compute_axial_effective_sample_size(chain, max_lag=300, block_size=7_000)

    np.testing.assert_allclose(
        result.integrated_autocorrelation_time, ANALYTIC_AUTOCORRELATION_TIME, rtol=0.1
    )
    assert not result.truncated.any()


# --------------------------------------------------------------------------------------------------
def test_effective_sample_size_is_invariant_under_shifting_samples_by_multiples_of_pi() -> None:
    rng = np.random.default_rng(1)
    chain = make_angle_chain(np.pi / 2 - 0.02, 20_000, 2, rng)
    shifted_chain = chain + np.pi * rng.integers(-2, 3, size=chain.shape)

    result = compute_axial_effective_sample_size(chain, max_lag=100)
    shifted_result = compute_axial_effective_sample_size(shifted_chain, max_lag=100)

    np.testing.assert_allclose(
        shifted_result.effective_sample_size, result.effective_sample_size, rtol=1e-8
    )


# --------------------------------------------------------------------------------------------------
def test_ordinary_effective_sample_size_of_raw_angles_is_wrong_for_axial_data() -> None:
    rng = np.random.default_rng(2)
    chain = make_angle_chain(0.0, 20_000, 1, rng)
    shifted_chain = chain + np.pi * rng.integers(0, 2, size=chain.shape)

    axial = compute_axial_effective_sample_size(shifted_chain, max_lag=100)
    naive = estimate_effective_sample_size(shifted_chain, max_lag=100)

    # The random +-pi jumps look like white noise to the ordinary estimator.
    assert naive.effective_sample_size[0] > 3 * axial.effective_sample_size[0]


# --------------------------------------------------------------------------------------------------
def test_burn_in_discards_the_leading_samples() -> None:
    chain = make_angle_chain(0.2, 4_000, 2, np.random.default_rng(3))
    chain[:300] += 0.5  # transient

    with_burn_in = compute_axial_effective_sample_size(chain, max_lag=50, burn_in=300)
    on_stationary_part = compute_axial_effective_sample_size(chain[300:], max_lag=50)

    np.testing.assert_allclose(
        with_burn_in.effective_sample_size, on_stationary_part.effective_sample_size
    )


# --------------------------------------------------------------------------------------------------
def test_zarr_store_gives_the_same_result_as_the_array_in_memory(tmp_path: Path) -> None:
    chain = make_angle_chain(-0.4, 1_500, 3, np.random.default_rng(4))
    storage = ZarrStorage(tmp_path / "samples.zarr", chunk_size=100, buffer_size=50)
    for sample in chain:
        storage.store(sample)
    storage.flush()

    from_disk = compute_axial_effective_sample_size(
        open_zarr_samples(tmp_path / "samples.zarr"), max_lag=40, burn_in=100, block_size=130
    )
    from_memory = compute_axial_effective_sample_size(chain, max_lag=40, burn_in=100)

    np.testing.assert_allclose(
        from_disk.effective_sample_size, from_memory.effective_sample_size, rtol=1e-10
    )


# --------------------------------------------------------------------------------------------------
def test_matches_arviz_on_the_sine_of_the_centered_angles() -> None:
    arviz = pytest.importorskip("arviz")
    chain = make_angle_chain(0.1, 2_000, 2, np.random.default_rng(5))
    centered_sine = np.sin(2 * (chain - np.angle(np.exp(2j * chain).mean(axis=0)) / 2))
    half = chain.shape[0] // 2
    split_chains = np.stack([centered_sine[:half], centered_sine[half:]])  # (chain, draw, vertex)

    result = compute_axial_effective_sample_size(chain, max_lag=half - 1)

    for vertex in range(chain.shape[1]):
        expected = float(arviz.ess(split_chains[:, :, vertex], method="mean"))
        np.testing.assert_allclose(result.effective_sample_size[vertex], expected, rtol=1e-6)


# --------------------------------------------------------------------------------------------------
def test_constant_vertex_gives_nan_and_does_not_affect_the_others() -> None:
    chain = make_angle_chain(0.3, 1_000, 2, np.random.default_rng(6))
    chain[:, 1] = 0.7

    result = compute_axial_effective_sample_size(chain, max_lag=20)

    assert np.isfinite(result.effective_sample_size[0])
    assert np.isnan(result.effective_sample_size[1])


# --------------------------------------------------------------------------------------------------
def test_invalid_burn_in_is_rejected() -> None:
    chain = make_angle_chain(0.3, 100, 2, np.random.default_rng(7))

    with pytest.raises(ValueError, match="burn_in"):
        compute_axial_effective_sample_size(chain, max_lag=5, burn_in=100)
