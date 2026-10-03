r"""Optimizer-selection strategy for MAP estimation: plain Euclidean L-BFGS-B, or L-BFGS in the
prior's Cameron-Martin space with cautious updating and a choice of line search. Each strategy
builds both the optimizer backend and the `OptimizationModel` it requires as a matched pair.
`CustomLBFGSStrategy`'s own line search is itself selectable between `ArmijoBacktrackingLineSearch`
(sufficient decrease only) and `StrongWolfeLineSearch` (bracketing + zoom, additionally enforcing a
curvature condition).

Classes:
    OptimizerStrategy: ABC interface for building a matched `OptimizationModel`/`BaseOptimizer`
        pair.
    ScipyLBFGSBStrategy: Plain Euclidean L-BFGS-B via `scipy.optimize.minimize`.
    CustomLBFGSStrategy: L-BFGS in the prior's Cameron-Martin space, with backtracking and
        cautious updating.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import override

from ls_bayesian.common.logging import BaseLogger
from ls_bayesian.optimization.algorithms.custom_lbfgs import (
    CustomLBFGSOptimizer,
    CustomLBFGSSettings,
)
from ls_bayesian.optimization.algorithms.scipy_lbfgs_b import (
    ScipyLBFGSBOptimizer,
    ScipyLBFGSBSettings,
)
from ls_bayesian.optimization.components.cautious_update import (
    CautiousUpdateSettings,
    CautiousUpdateStrategy,
)
from ls_bayesian.optimization.components.line_search import (
    ArmijoBacktrackingLineSearch,
    ArmijoBacktrackingLineSearchSettings,
    StrongWolfeLineSearch,
    StrongWolfeLineSearchSettings,
)
from ls_bayesian.optimization.components.seed_scaling import (
    BarzilaiBorweinSeedScaling,
    BarzilaiBorweinSeedScalingSettings,
)
from ls_bayesian.optimization.model import OptimizationModel
from ls_bayesian.optimization.optimizer import BaseOptimizer
from ls_bayesian.posterior import interfaces
from ls_bayesian.posterior.posterior import LogPosterior

from bayes_cep.optimization.model import CameronMartinPosteriorModel, EuclideanPosteriorModel

# Barzilai-Borwein seed-scaling defaults. `ls_bayesian`'s defaults (clamp `[1e-2, 1e2]`, fallback
# `1.0`) assume a Hessian of order one in the Cameron-Martin metric, i.e. a prior-dominated problem.
# Here the data (many observations, tiny noise variance, broad prior) dominate and the scaling
# wants values of about `1e-7`. Clamping it to `1e-2` costs about a dozen backtracking forward
# solves per iteration, and the neutral fallback `1.0` for the very first iteration (no correction
# pair yet to estimate the scale from) costs about 20 more.
SEED_SCALING_GAMMA_MIN = 1e-10
SEED_SCALING_FALLBACK_VALUE = 1e-6


# ==================================================================================================
class OptimizerStrategy(ABC):
    """ABC interface for building a matched `OptimizationModel`/`BaseOptimizer` pair.

    Methods:
        build: Build the model and optimizer for a given `LogPosterior` and prior.
    """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def build(
        self,
        log_posterior: LogPosterior,
        prior: interfaces.GaussianPrior,
        logger: BaseLogger | None = None,
    ) -> tuple[OptimizationModel, BaseOptimizer]:
        """Build the model and optimizer.

        Args:
            log_posterior (LogPosterior): Negative log-posterior to minimize.
            prior (interfaces.GaussianPrior): The same prior `log_posterior` was built with, e.g.
                `PosteriorBuilder.prior`.
            logger (BaseLogger | None, optional): Logger for iteration-by-iteration progress
                reports, shared with any sub-components (line search, acceptance strategy).
                Defaults to `None`.

        Returns:
            tuple[OptimizationModel, BaseOptimizer]: The model and optimizer, in the same geometry.
        """


# ==================================================================================================
@dataclass(frozen=True)
class ScipyLBFGSBStrategy(OptimizerStrategy):
    """Plain Euclidean L-BFGS-B via `scipy.optimize.minimize`.

    Pairs `ScipyLBFGSBOptimizer` with `EuclideanPosteriorModel`: scipy's L-BFGS-B is hardcoded to
    Euclidean geometry, so the model must expose the unmodified `LogPosterior` gradient, not the
    Cameron-Martin representer. Simple and well-tested, but not metric-consistent with a
    Cameron-Martin prior; see `bayes_cep.optimization.model`'s docstring for the metric-consistent
    alternative.

    Attributes:
        settings (ScipyLBFGSBSettings): Settings for the L-BFGS-B optimizer.
    """

    settings: ScipyLBFGSBSettings = field(default_factory=ScipyLBFGSBSettings)

    # ----------------------------------------------------------------------------------------------
    @override
    def build(
        self,
        log_posterior: LogPosterior,
        prior: interfaces.GaussianPrior,
        logger: BaseLogger | None = None,
    ) -> tuple[EuclideanPosteriorModel, ScipyLBFGSBOptimizer]:
        """Build the Euclidean model and the L-BFGS-B optimizer. `prior` is accepted but unused:
        `EuclideanPosteriorModel` only needs the posterior."""
        model = EuclideanPosteriorModel(log_posterior)
        optimizer = ScipyLBFGSBOptimizer(self.settings, logger=logger)
        return model, optimizer


# ==================================================================================================
@dataclass(frozen=True)
class CustomLBFGSStrategy(OptimizerStrategy):
    """L-BFGS in the prior's Cameron-Martin space, with Armijo backtracking and cautious updating
    (Li & Fukushima, 2001).

    Pairs `CustomLBFGSOptimizer` with `CameronMartinPosteriorModel`: the two-loop recursion, line
    search, and cautious-update condition are all evaluated in the Cameron-Martin inner product,
    matching the gradient the model provides.

    Attributes:
        lbfgs_settings (CustomLBFGSSettings): Memory size, iteration cap, gradient-norm tolerance.
        line_search_settings (ArmijoBacktrackingLineSearchSettings |
            StrongWolfeLineSearchSettings): Line-search settings; Armijo backtracking (default,
            sufficient decrease only) or strong-Wolfe (bracketing + zoom, also enforces a
            curvature condition).
            Selected on the command line via `--line-search-settings:<strategy-name>`.
        cautious_update_settings (CautiousUpdateSettings): Cautious correction-pair acceptance
            settings.
        seed_scaling_settings (BarzilaiBorweinSeedScalingSettings): Barzilai-Borwein-style
            seed-scaling clamp bounds (`gamma_min`/`gamma_max`) and first-iteration
            `fallback_value`. Defaults to `gamma_min=1e-10` and `fallback_value=1e-6` (see
            `SEED_SCALING_GAMMA_MIN`/`SEED_SCALING_FALLBACK_VALUE`) and the library's
            `gamma_max`.
    """

    lbfgs_settings: CustomLBFGSSettings = field(default_factory=CustomLBFGSSettings)
    line_search_settings: ArmijoBacktrackingLineSearchSettings | StrongWolfeLineSearchSettings = (
        field(default_factory=ArmijoBacktrackingLineSearchSettings)
    )
    cautious_update_settings: CautiousUpdateSettings = field(default_factory=CautiousUpdateSettings)
    seed_scaling_settings: BarzilaiBorweinSeedScalingSettings = field(
        default_factory=lambda: BarzilaiBorweinSeedScalingSettings(
            gamma_min=SEED_SCALING_GAMMA_MIN, fallback_value=SEED_SCALING_FALLBACK_VALUE
        )
    )

    # ----------------------------------------------------------------------------------------------
    @override
    def build(
        self,
        log_posterior: LogPosterior,
        prior: interfaces.GaussianPrior,
        logger: BaseLogger | None = None,
    ) -> tuple[CameronMartinPosteriorModel, CustomLBFGSOptimizer]:
        """Build the Cameron-Martin model, the line search, the cautious-update strategy, the
        seed-scaling strategy, and the L-BFGS optimizer, sharing the same logger across all four
        sub-components."""
        model = CameronMartinPosteriorModel(log_posterior, prior)
        line_search: ArmijoBacktrackingLineSearch | StrongWolfeLineSearch
        match self.line_search_settings:
            case StrongWolfeLineSearchSettings():
                line_search = StrongWolfeLineSearch(self.line_search_settings, logger=logger)
            case ArmijoBacktrackingLineSearchSettings():
                line_search = ArmijoBacktrackingLineSearch(self.line_search_settings, logger=logger)
        acceptance_strategy = CautiousUpdateStrategy(self.cautious_update_settings, logger=logger)
        seed_scaling_strategy = BarzilaiBorweinSeedScaling(
            self.seed_scaling_settings, logger=logger
        )
        optimizer = CustomLBFGSOptimizer(
            self.lbfgs_settings,
            line_search,
            acceptance_strategy,
            seed_scaling_strategy,
            logger=logger,
        )
        return model, optimizer
