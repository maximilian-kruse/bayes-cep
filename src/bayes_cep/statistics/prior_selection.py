r"""Selection of the prior parameters $(\kappa, \tau)$ from the MAP and Laplace runs of a grid.

The Laplace evidence ranks the prior parameters without the ground truth. When the truth is known
(synthetic data), the choice by the evidence can be checked against the choice by the error of the
MAP estimate.

Classes:
    SelectionValidation: How well a score picks the pair that gives the best MAP estimate.

Functions:
    compute_laplace_log_evidence: Log evidence of the prior parameters of one MAP and Laplace run.
    validate_selection: Compare the pair of highest score with the pair of highest skill.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr


# ==================================================================================================
def compute_laplace_log_evidence(
    loss_final: np.ndarray[tuple[int, ...], np.dtype[np.float64]] | float,
    log_determinant_ratio: np.ndarray[tuple[int, ...], np.dtype[np.float64]] | float,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]] | float:
    r"""Laplace approximation of the log evidence $\log p(y \mid \kappa, \tau)$, up to a constant.

    With the posterior $\propto \exp(-\Phi(m))\,\mathcal{N}(m; \bar m, \mathcal{C})$ and its
    Gaussian approximation $\mathcal{N}(m_{MAP}, \Gamma)$ at the MAP estimate,

    $$\log p(y) \approx -\left[\Phi(m_{MAP}) + \tfrac{1}{2}\|m_{MAP} - \bar m\|^2_{\mathcal{C}^{-1}}
    \right] + \tfrac{1}{2}\left(\log\det\Gamma - \log\det\mathcal{C}\right) + \text{const},$$

    where the bracket is the final loss of the MAP run (misfit plus the unnormalized prior cost) and
    the second term is the log-determinant ratio of the Laplace run (the Occam factor). The
    normalization of the prior cancels in the ratio, so the unnormalized prior cost suffices. The
    constant does not depend on $(\kappa, \tau)$ as long as the noise variance is fixed.

    Two approximations enter the second term. The Laplace run uses the Gauss-Newton Hessian, which
    leaves out the second-derivative term of the forward model, so the log evidence is only as good
    as this curvature. And the log-determinant ratio is exact only if the Laplace rank reaches the
    number of observations (the misfit Hessian has at most one nonzero eigenvalue per observation);
    every dropped direction would add $\log(1 + \lambda) > 0$ to the ratio's magnitude.
    The expansion is also taken at the MAP estimate, so the loss must be that of a stationary point.

    Args:
        loss_final (np.ndarray | float): Final loss of the MAP run, the negative log posterior up to
            a constant.
        log_determinant_ratio (np.ndarray | float): $\log\det\Gamma - \log\det\mathcal{C}$ of the
            Laplace run at that MAP estimate; non-positive.

    Returns:
        np.ndarray | float: The log evidence, same shape as the inputs.
    """
    return -loss_final + 0.5 * log_determinant_ratio


# ==================================================================================================
@dataclass(frozen=True)
class SelectionValidation:
    """How well a score picks the pair that gives the best MAP estimate, for one data set.

    Attributes:
        selected_index (int): Index of the grid cell of highest score (log evidence or
            cross-validation score).
        skill_index (int): Index of the grid cell of highest skill score.
        regret (float): Skill at `skill_index` minus skill at `selected_index`; zero if the score
            picks the best pair, otherwise the skill lost by following it.
        rank_correlation (float): Spearman rank correlation of the score and the skill over the
            grid cells.
    """

    selected_index: int
    skill_index: int
    regret: float
    rank_correlation: float


# ==================================================================================================
def validate_selection(
    score: np.ndarray[tuple[int], np.dtype[np.float64]],
    skill: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> SelectionValidation:
    """Compare the pair of highest score with the pair of highest skill score.

    Args:
        score (np.ndarray): Score of the grid cells of one data set (log evidence, or the negative
            held-out log predictive density), larger is better, shape `(num_cells,)`.
        skill (np.ndarray): Skill score of the MAP estimates of the same cells, same shape.

    Returns:
        SelectionValidation: The two chosen cells, the skill lost and the rank correlation.

    Raises:
        ValueError: If the shapes differ, there are fewer than three cells or a value is not finite
            (drop such cells first).
    """
    if score.shape != skill.shape or score.ndim != 1:
        raise ValueError(
            f"Got a score of shape {score.shape} and skill of shape {skill.shape}; "
            "both must be vectors of the same length."
        )
    if score.size < 3:
        raise ValueError(f"Need at least 3 grid cells, got {score.size}.")
    if not (np.isfinite(score).all() and np.isfinite(skill).all()):
        raise ValueError("The score and the skill must be finite in every cell.")
    selected_index = int(np.argmax(score))
    skill_index = int(np.argmax(skill))
    return SelectionValidation(
        selected_index=selected_index,
        skill_index=skill_index,
        regret=float(skill[skill_index] - skill[selected_index]),
        rank_correlation=float(spearmanr(score, skill).statistic),
    )
