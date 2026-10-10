"""Constants and pure helper functions for the tests (fixtures live in `conftest.py` files)."""

import numpy as np


# ==================================================================================================
def generate_ar1_chain(
    autoregression_coefficient: float,
    num_samples: int,
    num_components: int,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Stationary AR(1) chains $x_t = \phi x_{t-1} + \varepsilon_t$ with unit marginal variance.

    Their integrated autocorrelation time is $(1 + \phi) / (1 - \phi)$.
    """
    innovation_scale = np.sqrt(1 - autoregression_coefficient**2)
    innovations = innovation_scale * rng.normal(size=(num_samples, num_components))
    chain = np.empty((num_samples, num_components))
    chain[0] = rng.normal(size=num_components)
    for index in range(1, num_samples):
        chain[index] = autoregression_coefficient * chain[index - 1] + innovations[index]
    return chain
