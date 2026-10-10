"""Metadata of a run and of a study: what code, inputs and machine produced the results.

The environment covers everything that determines the code that ran: the Python version, the
pixi environment and the hash of `pixi.lock` (which pins all conda and PyPI packages), and the git
state of the repository and of every editable (path) dependency, which `pixi.lock` does not pin. A
study additionally archives the specification behind these fingerprints, see `EnvironmentArchive`.
The records are frozen dataclasses; `dataclasses.asdict` gives their JSON form.

Classes:
    GitState: Commit and uncommitted changes of one git repository.
    GitRepository: Read the state and the uncommitted changes of one git repository.
    EditablePackage: An editable installed package and the git state of its source directory.
    Environment: What code the process runs, for the study description and each run.
    RunMetadata: Environment, inputs and machine of one run.
    EnvironmentArchive: The specification behind an `Environment`, copied for the study directory.
"""

import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Self
from urllib.parse import unquote, urlparse

from bayes_cep.run.directories import REPOSITORY_ROOT

LOCK_FILE_NAME = "pixi.lock"
PROJECT_FILE_NAME = "pyproject.toml"
PIXI_ENVIRONMENT_VARIABLE = "PIXI_ENVIRONMENT_NAME"
CONDA_SPECIFICATION_PATTERN = "*_conda_spec.txt"
PATCH_FILE_NAME = "source.patch"
GIT_TIMEOUT_SECONDS = 10
EXPORT_TIMEOUT_SECONDS = 120


# ==================================================================================================
@dataclass(frozen=True)
class GitState:
    """The state of one git repository.

    Attributes:
        commit (str | None): Hash of the checked-out commit; `None` if not a git repository or git
            is unavailable.
        dirty (bool | None): Whether there are uncommitted changes or untracked files; `None` if
            unknown.
        diff_sha256 (str | None): Hash of the uncommitted changes of tracked files (see
            `GitRepository.read_uncommitted_changes`) if there are any, else `None`, so that
            different uncommitted states can be told apart. Untracked files only set `dirty`.
    """

    commit: str | None
    dirty: bool | None
    diff_sha256: str | None


# ==================================================================================================
class GitRepository:
    """A git repository, read through the `git` command line.

    Every query returns `None` entries instead of raising if git fails, hangs or is unavailable,
    since metadata must never stop a run.
    """

    def __init__(self, directory: Path) -> None:
        """Wrap the repository containing `directory`."""
        self._directory = directory

    # ----------------------------------------------------------------------------------------------
    def read_state(self) -> GitState:
        """The commit, and whether and how the working tree differs from it.

        Returns:
            GitState: The state; its entries are `None` where git cannot tell, e.g. outside of a
                repository.
        """
        status = self._run("status", "--porcelain")
        dirty = None if status is None else bool(status.strip())
        changes = self.read_uncommitted_changes() if dirty else None
        commit = self._run("rev-parse", "HEAD")
        return GitState(
            commit=None if commit is None else commit.strip(),
            dirty=dirty,
            diff_sha256=hashlib.sha256(changes.encode()).hexdigest() if changes else None,
        )

    # ----------------------------------------------------------------------------------------------
    def read_uncommitted_changes(self) -> str | None:
        """The uncommitted changes of tracked files as an applicable patch.

        `pixi.lock` is left out: git treats it as binary (`-diff`), and it is archived and hashed
        on its own. Binary files are included in full, so that the patch always applies.

        Returns:
            str | None: The patch, empty if there are no changes; `None` if git fails.
        """
        return self._run("diff", "--binary", "HEAD", "--", ".", f":(exclude){LOCK_FILE_NAME}")

    # ----------------------------------------------------------------------------------------------
    def _run(self, *arguments: str) -> str | None:
        """Output of a git command, or `None` if git fails, hangs or is unavailable.

        The output is returned as it is (undecodable bytes are replaced), including the trailing
        newline, so that a patch can be written out unchanged.
        """
        try:
            completed = subprocess.run(
                ["git", "-C", str(self._directory), *arguments],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True,
                timeout=GIT_TIMEOUT_SECONDS,
            )
        except OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired:
            return None
        return completed.stdout


# ==================================================================================================
@dataclass(frozen=True)
class EditablePackage:
    """An editable installed package.

    Attributes:
        path (str): The source directory the package is installed from.
        git (GitState): The git state of that directory.
    """

    path: str
    git: GitState

    @classmethod
    def collect_installed(cls) -> dict[str, Self]:
        """The editable packages installed in the current environment, found through PEP 610.

        The package of this repository itself is left out.

        Returns:
            dict[str, Self]: The packages by name, sorted.
        """
        packages: dict[str, Self] = {}
        for distribution in importlib_metadata.distributions():
            name = distribution.metadata["Name"]
            source_directory = cls._find_source_directory(distribution)
            if (
                name is not None
                and source_directory is not None
                and source_directory.resolve() != REPOSITORY_ROOT
            ):
                packages[name] = cls(
                    path=str(source_directory),
                    git=GitRepository(source_directory).read_state(),
                )
        return dict(sorted(packages.items()))

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _find_source_directory(distribution: importlib_metadata.Distribution) -> Path | None:
        """The source directory of an editable installed distribution, else `None`."""
        text = distribution.read_text("direct_url.json")
        if text is None:
            return None
        try:
            direct_url = json.loads(text)
            url = urlparse(direct_url["url"])
            editable = direct_url.get("dir_info", {}).get("editable", False)
        except ValueError, KeyError, AttributeError:
            return None  # malformed metadata of one package must not stop the collection
        if url.scheme != "file" or not editable:
            return None
        return Path(unquote(url.path))


# ==================================================================================================
@dataclass(frozen=True)
class Environment:
    """What environment the current process runs in.

    Attributes:
        python (str): Python version.
        pixi_environment (str | None): Name of the pixi environment the process runs in; `None`
            outside of pixi.
        pixi_lock_sha256 (str | None): Hash of `pixi.lock`; `None` if there is none.
        git (GitState): The git state of this repository.
        editable_packages (dict[str, EditablePackage]): The editable packages by name, except this
            repository (see `git`), with the git state of their source, since `pixi.lock` does not
            pin path dependencies.
    """

    python: str
    pixi_environment: str | None
    pixi_lock_sha256: str | None
    git: GitState
    editable_packages: dict[str, EditablePackage]

    @classmethod
    def collect_from_current_process(cls) -> Self:
        """Collect the environment of the current process."""
        lock_path = REPOSITORY_ROOT / LOCK_FILE_NAME
        return cls(
            python=platform.python_version(),
            pixi_environment=os.environ.get(PIXI_ENVIRONMENT_VARIABLE),
            pixi_lock_sha256=(
                hashlib.sha256(lock_path.read_bytes()).hexdigest() if lock_path.exists() else None
            ),
            git=GitRepository(REPOSITORY_ROOT).read_state(),
            editable_packages=EditablePackage.collect_installed(),
        )

    # ----------------------------------------------------------------------------------------------
    def find_differences(self, recorded: dict[str, Any]) -> list[str]:
        """Describe how this environment differs from a recorded one.

        Args:
            recorded (dict[str, Any]): The JSON form of an environment (`dataclasses.asdict`), e.g.
                the one stored in the study description.

        Returns:
            list[str]: One line per differing entry, by dotted path, e.g. `git.commit: recorded
                'abc', now 'def'`; empty if the environments agree. An entry present in only one
                of them counts as `None` in the other.
        """
        before = self._flatten(recorded)
        now = self._flatten(asdict(self))
        return [
            f"{key}: recorded {before.get(key)!r}, now {now.get(key)!r}"
            for key in sorted(before.keys() | now.keys())
            if before.get(key) != now.get(key)
        ]

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        """Nested dicts as one dict with dotted keys."""
        flat: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, dict):
                flat.update(Environment._flatten(value, f"{prefix}{key}."))
            else:
                flat[f"{prefix}{key}"] = value
        return flat


# ==================================================================================================
@dataclass(frozen=True)
class RunMetadata:
    """Data for a run.

    Attributes:
        started (str): Start time, ISO 8601 with time zone.
        environment (Environment): The code the run executed.
        input_sha256 (dict[str, str | None]): Content hash of every input file, by path relative
            to the repository (absolute outside of it); `None` for a file that does not exist.
        host (str): Name of the machine.
        cpu_count (int | None): Number of cores of the machine.
        slurm_job_id (str | None): SLURM job id if the run is a cluster task.
        slurm_array_task_id (str | None): SLURM array task id if the run is a cluster task.
    """

    started: str
    environment: Environment
    input_sha256: dict[str, str | None]
    host: str
    cpu_count: int | None
    slurm_job_id: str | None
    slurm_array_task_id: str | None

    @classmethod
    def collect_for_run(cls, input_files: Sequence[Path], environment: Environment) -> Self:
        """Collect the metadata of a run about to start.

        Args:
            input_files (Sequence[Path]): Files the run reads; a missing file is recorded as
                `None`.
            environment (Environment): The environment the run executes in. Collected once by the
                submitter instead of per run, since it costs several git calls and a scan of all
                installed packages.

        Returns:
            Self: The environment, the content hashes of the input files, the machine, and the
                SLURM job identifiers if the run is a cluster task.
        """
        return cls(
            started=datetime.now().astimezone().isoformat(timespec="seconds"),
            environment=environment,
            input_sha256=cls._hash_input_files(input_files),
            host=socket.gethostname(),
            cpu_count=os.cpu_count(),
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
        )

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _hash_input_files(input_files: Sequence[Path]) -> dict[str, str | None]:
        """Content hash of each file by repository-relative path; `None` for a missing file."""
        hashes: dict[str, str | None] = {}
        for path in input_files:
            resolved = path.resolve()
            if resolved.is_relative_to(REPOSITORY_ROOT):
                key = str(resolved.relative_to(REPOSITORY_ROOT))
            else:
                key = str(resolved)
            if not resolved.exists():
                hashes[key] = None
                continue
            with resolved.open("rb") as file:
                hashes[key] = hashlib.file_digest(file, "sha256").hexdigest()
        return hashes


# ==================================================================================================
class EnvironmentArchive:
    """The content behind an `Environment`: its specification, copied for the study directory.

    The `Environment` records fingerprints (the hash of `pixi.lock`, the hash of the uncommitted
    changes); the archive stores the files they stand for and checks that they match. Together, the
    archived files rebuild the environment (`pixi install --locked`) without the original machine:

    - `pixi.lock` and `pyproject.toml`: the complete, pinned specification of all environments.
    - `<environment>_<platform>_conda_spec.txt`: the exact conda packages of the pixi environment in
      use, installable with conda or mamba. PyPI and path dependencies are not part of it (they are
      in `pixi.lock` and in the git states of the environment). Skipped, with a warning, if the
      process does not run in a pixi environment or the export fails.
    - `source.patch`: the uncommitted changes of tracked files of this repository, if any (without
      `pixi.lock`, which is archived itself); `git apply` restores them on top of the commit.
      Untracked files are not included.
    """

    def __init__(self, environment: Environment, target_dir: Path) -> None:
        """Archive the specification behind `environment` into `target_dir`.

        Args:
            environment (Environment): The environment collected from the current process.
            target_dir (Path): Directory to write to; created if missing, existing files stay.
        """
        self._environment = environment
        self._target_dir = target_dir

    # ----------------------------------------------------------------------------------------------
    def write_specification(self) -> list[str]:
        """Write the archive.

        A warning is issued if an archived file does not match the hash recorded in the
        environment, i.e. if it changed since the environment was collected.

        Returns:
            list[str]: The names of the files in the target directory.
        """
        self._target_dir.mkdir(parents=True, exist_ok=True)
        for name in (LOCK_FILE_NAME, PROJECT_FILE_NAME):
            if (REPOSITORY_ROOT / name).exists():
                shutil.copy(REPOSITORY_ROOT / name, self._target_dir / name)
        self._check_hash(
            LOCK_FILE_NAME,
            self._environment.pixi_lock_sha256,
            self._read_archived(LOCK_FILE_NAME),
        )
        if self._environment.pixi_environment is None:
            warnings.warn(
                "Not in a pixi environment: no conda specification archived.", stacklevel=2
            )
        else:
            self._export_conda_specification(self._environment.pixi_environment)
        patch = GitRepository(REPOSITORY_ROOT).read_uncommitted_changes()
        if patch:
            (self._target_dir / PATCH_FILE_NAME).write_text(patch, encoding="utf-8")
        self._check_hash(
            PATCH_FILE_NAME, self._environment.git.diff_sha256, self._read_archived(PATCH_FILE_NAME)
        )
        return sorted(path.name for path in self._target_dir.iterdir())

    # ----------------------------------------------------------------------------------------------
    def _read_archived(self, name: str) -> bytes | None:
        """The content of an archived file, or `None` if it was not written."""
        path = self._target_dir / name
        return path.read_bytes() if path.exists() else None

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _check_hash(name: str, recorded_sha256: str | None, content: bytes | None) -> None:
        """Warn if the archived content does not have the hash recorded in the environment."""
        archived_sha256 = None if content is None else hashlib.sha256(content).hexdigest()
        if archived_sha256 != recorded_sha256:
            warnings.warn(
                f"The archived {name} does not match the environment collected before: it "
                "changed in between.",
                stacklevel=2,
            )

    # ----------------------------------------------------------------------------------------------
    def _export_conda_specification(self, pixi_environment: str) -> None:
        """Write the explicit conda specification of a pixi environment into the target directory.

        The specification is optional: if `pixi` is unavailable or the export fails, a warning is
        issued and nothing is written.
        """
        if shutil.which("pixi") is None:
            warnings.warn("pixi not found: no conda specification archived.", stacklevel=2)
            return
        command = [
            "pixi",
            "workspace",
            "export",
            "conda-explicit-spec",
            str(self._target_dir),
            f"--environment={pixi_environment}",
            "--ignore-pypi-errors",
            "--ignore-source-errors",
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
                check=False,
                timeout=EXPORT_TIMEOUT_SECONDS,
            )
            reason = completed.stderr.strip()
        except OSError, subprocess.TimeoutExpired:
            reason = "the export failed to run or timed out."
        if not any(self._target_dir.glob(CONDA_SPECIFICATION_PATTERN)):
            warnings.warn(f"No conda specification archived. {reason}", stacklevel=2)
