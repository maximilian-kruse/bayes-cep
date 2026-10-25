r"""Cross-validation over the observations: the folds and the score of the held-out ones.

The prior parameters of a posterior are judged by how well the MAP estimate of the training
observations predicts the held-out ones. The ground truth is not needed.

Classes:
    HeldOutSplitSettings: Which fold of a random partition of the observations is held out.

Functions:
    compute_gaussian_negative_log_predictive: Mean negative log density of held-out observations.
"""

from dataclasses import dataclass

import numpy as np


# ==================================================================================================
@dataclass(frozen=True)
class HeldOutSplitSettings:
    """Which fold of a random partition of the observations is held out.

    The partition depends only on the number of observations and `seed`, so all runs with the same
    `seed` and `num_folds` split the same data the same way, and their folds are disjoint and cover
    all observations.

    Attributes:
        num_folds (int): Number of folds, at least 2.
        fold_index (int): The held-out fold, in `[0, num_folds)`.
        seed (int): Seed of the random partition.
    """

    num_folds: int
    fold_index: int
    seed: int

    def __post_init__(self) -> None:
        if self.num_folds < 2:
            raise ValueError(f"num_folds must be at least 2, got {self.num_folds}.")
        if not 0 <= self.fold_index < self.num_folds:
            raise ValueError(f"fold_index must be in [0, {self.num_folds}), got {self.fold_index}.")

    # ----------------------------------------------------------------------------------------------
    def split(
        self, num_observations: int
    ) -> tuple[
        np.ndarray[tuple[int], np.dtype[np.intp]], np.ndarray[tuple[int], np.dtype[np.intp]]
    ]:
        """Split the positions of the observations into training and held-out positions.

        Args:
            num_observations (int): Number of observations, at least `num_folds`.

        Returns:
            tuple[np.ndarray, np.ndarray]: The sorted positions of the training observations and of
                the held-out fold.

        Raises:
            ValueError: If there are fewer observations than folds.
        """
        if num_observations < self.num_folds:
            raise ValueError(
                f"Cannot split {num_observations} observations into {self.num_folds} folds."
            )
        folds = np.array_split(
            np.random.default_rng(self.seed).permutation(num_observations), self.num_folds
        )
        held_out = np.sort(folds[self.fold_index])
        training = np.sort(np.concatenate(folds[: self.fold_index] + folds[self.fold_index + 1 :]))
        return training, held_out


# ==================================================================================================
def compute_gaussian_negative_log_predictive(
    residuals: np.ndarray[tuple[int], np.dtype[np.float64]], noise_variance: float
) -> float:
    r"""Mean negative log density of held-out observations given a point prediction.

    For residuals $r_i$ (prediction minus observation) and noise variance $\sigma^2$,
    $\frac{1}{n}\sum_i \left[\frac{r_i^2}{2\sigma^2} + \frac{1}{2}\log(2\pi\sigma^2)\right]$. It
    uses the MAP prediction and the observation noise only; the posterior uncertainty of the
    prediction is not included, so it is a lower bound of a fully Bayesian predictive score.

    Args:
        residuals (np.ndarray): Predicted minus observed activation times, shape `(n,)`.
        noise_variance (float): Variance $\sigma^2 > 0$ of the observation noise.

    Returns:
        float: The mean negative log predictive density per observation.

    Raises:
        ValueError: If there are no residuals or the noise variance is not positive.
    """
    if residuals.size == 0:
        raise ValueError("Need at least one held-out residual.")
    if noise_variance <= 0:
        raise ValueError(f"noise_variance must be positive, got {noise_variance}.")
    return float(
        0.5 * np.mean(residuals**2) / noise_variance + 0.5 * np.log(2 * np.pi * noise_variance)
    )
