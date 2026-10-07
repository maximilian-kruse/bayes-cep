"""Metadata of a run and of a study: what code, inputs and machine produced the results.

Functions:
    collect_run_metadata: Metadata of one run.
    collect_environment: Code and package versions, for the study description.
"""

import hashlib
import os
import platform
import socket
import subprocess
from collections.abc import Sequence
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from bayes_cep.run.config import REPOSITORY_ROOT

TRACKED_PACKAGES = ("bayes_cep", "ls_bayesian", "eikonax", "numpy", "scipy", "jax", "tyro")


# ==================================================================================================
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------------------
def _git(*arguments: str) -> str | None:
    """Output of a git command in the repository, or `None` if git is unavailable."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), *arguments],
            capture_output=True,
            text=True,
            check=True,
        )
    except OSError, subprocess.CalledProcessError:
        return None
    return completed.stdout.strip()


# --------------------------------------------------------------------------------------------------
def _git_state() -> dict[str, Any]:
    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


# --------------------------------------------------------------------------------------------------
def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package.replace("_", "-"))
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


# ==================================================================================================
def collect_environment() -> dict[str, Any]:
    """Collect git state, the `pixi.lock` hash and key package versions.

    Returns:
        dict[str, Any]: JSON-serializable environment description.
    """
    lock_path = REPOSITORY_ROOT / "pixi.lock"
    return {
        "git": _git_state(),
        "pixi_lock_sha256": _sha256(lock_path) if lock_path.exists() else None,
        "python": platform.python_version(),
        "packages": _package_versions(),
    }


# ==================================================================================================
def collect_run_metadata(input_files: Sequence[Path]) -> dict[str, Any]:
    """Collect the metadata of a run about to start.

    Includes the environment, content hashes of the input files, the machine, and the SLURM job
    identifiers if the run is a cluster task.

    Args:
        input_files (Sequence[Path]): Files the run reads; missing ones are left out.

    Returns:
        dict[str, Any]: JSON-serializable metadata.
    """
    return {
        "started": datetime.now().astimezone().isoformat(timespec="seconds"),
        "environment": collect_environment(),
        "input_sha256": {path.name: _sha256(path) for path in input_files if path.exists()},
        "host": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
