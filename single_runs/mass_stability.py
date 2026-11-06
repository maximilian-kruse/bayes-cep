r"""Stability limit of the step angle of PHMC for candidate mass matrices, at states of chains.

Preconditioned HMC on function space
([`PHMCAlgorithm`][ls_bayesian.mcmc.algorithms.phmc.PHMCAlgorithm]) integrates the Gaussian
reference $\mathcal N(0, \Sigma_M)$ exactly (a rotation by the step angle $\varepsilon$) and
applies the rest of the potential as kicks. In the coordinates of the Laplace
eigenvectors the precision of the posterior at a state $u_s$ is $I + K_s$ with $K_s =
V^TH(u_s)V$, and the kicks act with the difference to the precision $\Sigma_M^{-1}$ of the mass
matrix. Along an eigenvector of $\Sigma_M^{1/2}(I + K_s)\Sigma_M^{1/2}$ with eigenvalue $\omega^2$
the extra stiffness is $\gamma = \omega^2 - 1$, and the Strang splitting (half kick, rotation,
half kick) of that mode is stable if and only if $|2\cos\varepsilon - \gamma\varepsilon\sin
\varepsilon| < 2$, which for $\gamma > 0$ is $\gamma\,\varepsilon\tan(\varepsilon/2) < 2$, about
$\varepsilon < 2/\sqrt\gamma$. The limit of a state is that of its largest $\omega^2$. For a
Gaussian target this is exact; the real posterior is not Gaussian, so the limit is an estimate
from the local curvature, and a condition that is necessary, not sufficient, for a good
acceptance rate.

The run computes $K_s$ (784 products per state, a few seconds) at states of the chains and at the
saved states of a calibration run (the states of
[`HmcTrajectoryRun`][single_runs.hmc_trajectory.HmcTrajectoryRun]), and the spectrum of $\omega^2$
for three mass matrices (their covariances in the coefficients of the $K$ eigenvectors):

- `laplace`: $\operatorname{diag}(1/(1+\lambda_i))$,
- `calibrated`: the covariance of the calibration for the leading $k$ coefficients (its
  `calibrated_covariance`) and the Laplace variances beyond,
- `averaged_covariance`: the average of $(I + K_s)^{-1}$ over the chain states for the leading
  $k$ coefficients (floored at the Laplace covariance, as the calibration does) and the Laplace
  variances beyond.

The step limits of the calibrated mass matrix at the saved states can be compared with the
trajectories of the earlier study, which were stable at the step angle 0.1 and unstable at 0.15.

Classes:
    MassStabilityRunConfig: The posterior, the chains, the calibration and the states.
    MassStabilityResults: The arrays the run writes.
    MassStabilityRun: The stability run.

Functions:
    step_angle_limit: The largest stable step angle for an eigenvalue $\omega^2$.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, override

import numpy as np
from ls_bayesian.common.logging import BaseLogger
from ls_bayesian.mcmc.storage import open_zarr_samples
from scipy.optimize import brentq

from bayes_cep.laplace.builder import LaplaceBuilder, LaplaceSettings
from bayes_cep.laplace.calibration import regularize_coefficient_covariance
from bayes_cep.run.config import RunConfig
from bayes_cep.run.directories import ArrayRecord, resolve_repository_path
from bayes_cep.run.progress import StepReporter, describe_array
from bayes_cep.run.template import Metrics, Run
from single_runs.calibrate import CalibrationResults, PilotStates
from single_runs.hessian_average import project_gauss_newton_hessian
from single_runs.inference import InferenceProblemConfig, assemble_posterior_components

CANDIDATE_NAMES = ("laplace", "calibrated", "averaged_covariance")
ORIGIN_PILOT = -1  # `state_origin` of the saved states of the calibration run
_MAXIMUM_ANGLE = np.pi  # the limit never exceeds a half period


# ==================================================================================================
def step_angle_limit(omega_squared: float) -> float:
    r"""The largest step angle for which the splitting of a mode is stable.

    With $\gamma = \omega^2 - 1$: $\pi$ (no limit below a half period) for $\gamma \le 0$, else the
    root of $\gamma\,\varepsilon\tan(\varepsilon/2) = 2$ in $(0, \pi)$.

    Args:
        omega_squared (float): Eigenvalue $\omega^2$ of $\Sigma_M^{1/2}(I + K)\Sigma_M^{1/2}$.

    Returns:
        float: The limit of the step angle, at most $\pi$.
    """
    gamma = omega_squared - 1.0
    if gamma <= 0.0:
        return _MAXIMUM_ANGLE

    def residual(angle: float) -> float:
        return gamma * angle * np.tan(0.5 * angle) - 2.0

    upper = _MAXIMUM_ANGLE * (1.0 - 1e-9)
    if residual(upper) < 0.0:
        return _MAXIMUM_ANGLE
    return float(brentq(residual, 1e-12, upper))


# ==================================================================================================
@dataclass(frozen=True)
class MassStabilityRunConfig(RunConfig):
    """The posterior, the chains, the calibration and the states.

    Attributes:
        problem (InferenceProblemConfig): The posterior the chains sample.
        map_estimate_path (Path): `.npy` file with the MAP estimate the Laplace approximation was
            centered at.
        laplace (LaplaceSettings): Eigensolver and rank of the eigenproblem at the MAP estimate.
        chain_paths (tuple[Path, ...]): The Zarr stores (`samples.zarr`) of finished MCMC runs of
            `problem`, whose states carry the Hessians and the averaged covariance.
        burn_in (int): Number of samples discarded from the start of each chain.
        states_per_chain (int): Number of states of each chain, evenly spaced after the burn-in.
        calibration_path (Path): The `results` directory of the calibration run of these chains:
            its `calibrated_covariance` is a mass matrix and its saved states are further states.
    """

    problem: InferenceProblemConfig
    map_estimate_path: Path
    laplace: LaplaceSettings
    chain_paths: tuple[Path, ...]
    burn_in: int
    states_per_chain: int
    calibration_path: Path

    def __post_init__(self) -> None:
        if self.burn_in < 0 or self.states_per_chain < 1:
            raise ValueError(
                f"burn_in must be non-negative and states_per_chain positive, got "
                f"{self.burn_in} and {self.states_per_chain}."
            )
        if len(self.chain_paths) == 0:
            raise ValueError("At least one chain is needed.")


# ==================================================================================================
@dataclass(frozen=True)
class MassStabilityResults(ArrayRecord):
    """The arrays a stability run writes; $S$ is the number of states, $C$ the number of candidates
    (`CANDIDATE_NAMES`), $K$ the rank.

    Attributes:
        eigenvalues (np.ndarray): Eigenvalues of the Laplace approximation, shape `(K,)`.
        state_origin (np.ndarray): Position of the chain of each state among `chain_paths`, or
            `ORIGIN_PILOT` for the saved states of the calibration, shape `(S,)`.
        state_indices (np.ndarray): Sample index in the chain of each state (position in the saved
            states for the calibration's), shape `(S,)`.
        omega_squared (np.ndarray): The eigenvalues $\\omega^2$ of each state and candidate,
            descending, shape `(S, C, K)`.
        step_limits (np.ndarray): The stable step angle limit of the largest $\\omega^2$, shape
            `(S, C)`.
        averaged_covariance (np.ndarray): The leading block of the average of $(I + K_s)^{-1}$ over
            the chain states, before the floor, shape `(k, k)`.
    """

    FILE_NAMES: ClassVar[Mapping[str, str]] = {
        "eigenvalues": "hessian_eigenvalues.npy",
        "state_origin": "state_origin.npy",
        "state_indices": "state_indices.npy",
        "omega_squared": "omega_squared.npy",
        "step_limits": "step_limits.npy",
        "averaged_covariance": "averaged_covariance.npy",
    }

    eigenvalues: np.ndarray
    state_origin: np.ndarray
    state_indices: np.ndarray
    omega_squared: np.ndarray
    step_limits: np.ndarray
    averaged_covariance: np.ndarray


_FILES = MassStabilityResults.FILE_NAMES  # the names of the result files


# ==================================================================================================
class MassStabilityRun(Run[MassStabilityRunConfig]):
    """Stability limit of the PHMC step angle for candidate mass matrices at chain states."""

    config_type = MassStabilityRunConfig
    name = "mass-stability"
    outputs = {
        f"results/{_FILES['eigenvalues']}": "Eigenvalues of the Laplace approximation.",
        f"results/{_FILES['state_origin']}": "Chain (or the calibration) of each state.",
        f"results/{_FILES['state_indices']}": "Sample index of each state.",
        f"results/{_FILES['omega_squared']}": "Spectrum of the preconditioned precision, per "
        "state and candidate mass matrix.",
        f"results/{_FILES['step_limits']}": "Stable step angle limit per state and candidate.",
        f"results/{_FILES['averaged_covariance']}": "Leading block of the average of the "
        "inverse precisions.",
    }

    # ----------------------------------------------------------------------------------------------
    @override
    def input_files(self) -> list[Path]:
        # The chains are directories (Zarr stores) and not hashed.
        calibration_dir = resolve_repository_path(self.config.calibration_path)
        return [
            *self.config.problem.input_files(),
            resolve_repository_path(self.config.map_estimate_path),
            *CalibrationResults.file_paths(calibration_dir),
            *PilotStates.file_paths(calibration_dir),
        ]

    # ----------------------------------------------------------------------------------------------
    @override
    def report(self, run_dir: Path) -> None:
        """No plots: the arrays are analysed in a notebook."""

    # ----------------------------------------------------------------------------------------------
    @override
    def _execute(self, run_dir: Path, logger: BaseLogger) -> Metrics:
        config = self.config
        steps = StepReporter(logger)
        assembled = assemble_posterior_components(config.problem, steps, logger)
        components = assembled.components
        map_estimate = np.load(resolve_repository_path(config.map_estimate_path))
        calibration_dir = resolve_repository_path(config.calibration_path)

        with steps.step(f"Eigenproblem of rank {config.laplace.num_eigenvalues} at the MAP"):
            pairs = LaplaceBuilder(config.laplace, components).build(map_estimate).eigenpairs
        eigenvalues, eigenvectors = pairs.eigenvalues, pairs.eigenvectors
        rank = eigenvalues.size
        logger.info(f"      {describe_array('eigenvalues', eigenvalues)}")
        laplace_variances = 1.0 / (1.0 + eigenvalues)
        calibration = CalibrationResults.load(calibration_dir)
        if not np.array_equal(calibration.eigenvalues, eigenvalues):
            raise ValueError(
                f"The eigenvalues of the calibration in {calibration_dir} differ from those of "
                "this run: the coefficients refer to different eigenvectors."
            )
        calibrated_block = calibration.calibrated_covariance
        block_size = calibrated_block.shape[0]
        if block_size > rank:
            raise ValueError(
                f"The calibrated covariance has {block_size} coefficients, the eigenproblem {rank}."
            )

        states, origin, indices = [], [], []
        for position, chain_path in enumerate(config.chain_paths):
            chain = open_zarr_samples(resolve_repository_path(chain_path))
            chain_indices = np.linspace(
                config.burn_in, chain.shape[0] - 1, config.states_per_chain
            ).astype(int)
            states.extend(np.asarray(chain[int(index)]) for index in chain_indices)
            origin.extend([position] * chain_indices.size)
            indices.extend(chain_indices)
        num_chain_states = len(states)
        pilot = PilotStates.load(calibration_dir)
        states.extend(pilot.states)
        origin.extend([ORIGIN_PILOT] * pilot.states.shape[0])
        indices.extend(range(pilot.states.shape[0]))

        identity = np.eye(rank)
        precisions = np.empty((len(states), rank, rank))
        covariance_sum = np.zeros((rank, rank))
        for position, state in enumerate(states):
            with steps.step(f"Hessian at state {position + 1} of {len(states)}"):
                precisions[position] = identity + project_gauss_newton_hessian(
                    components, state, eigenvectors
                )
            if position < num_chain_states:
                covariance_sum += np.linalg.inv(precisions[position])
        averaged = covariance_sum / num_chain_states

        averaged_block = regularize_coefficient_covariance(
            averaged[:block_size, :block_size], laplace_variances[:block_size], 0.0, 1.0, 1.0
        )
        covariances = []
        for leading in (None, calibrated_block, averaged_block):
            covariance = np.diag(laplace_variances)
            if leading is not None:
                covariance[:block_size, :block_size] = leading
            covariances.append(np.linalg.cholesky(0.5 * (covariance + covariance.T)))

        omega_squared = np.empty((len(states), len(CANDIDATE_NAMES), rank))
        step_limits = np.empty((len(states), len(CANDIDATE_NAMES)))
        with steps.step("Spectra of the preconditioned precisions"):
            for position in range(len(states)):
                for column, factor in enumerate(covariances):
                    preconditioned = factor.T @ precisions[position] @ factor
                    values = np.linalg.eigvalsh(0.5 * (preconditioned + preconditioned.T))[::-1]
                    omega_squared[position, column] = values
                    step_limits[position, column] = step_angle_limit(float(values[0]))

        MassStabilityResults(
            eigenvalues=eigenvalues,
            state_origin=np.asarray(origin),
            state_indices=np.asarray(indices),
            omega_squared=omega_squared,
            step_limits=step_limits,
            averaged_covariance=averaged[:block_size, :block_size],
        ).save(self.results_dir(run_dir))

        metrics: Metrics = {
            "num_chain_states": int(num_chain_states),
            "num_pilot_states": int(pilot.states.shape[0]),
        }
        origin_array = np.asarray(origin)
        for column, name in enumerate(CANDIDATE_NAMES):
            for label, mask in (
                ("chain", origin_array != ORIGIN_PILOT),
                ("pilot", origin_array == ORIGIN_PILOT),
            ):
                limits = step_limits[mask, column]
                metrics[f"{name}_{label}_step_limit_min"] = float(limits.min())
                metrics[f"{name}_{label}_step_limit_p10"] = float(np.percentile(limits, 10))
                metrics[f"{name}_{label}_step_limit_median"] = float(np.median(limits))
        return metrics
