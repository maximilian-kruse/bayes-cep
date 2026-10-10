"""The environment a process runs in: what code, packages and git states determine its results.

The environment covers the Python version, the pixi environment and the hash of `pixi.lock`
(which pins all conda and PyPI packages), and the git state of the repository and of every editable
(path) dependency, which `pixi.lock` does not pin. The records are frozen dataclasses;
`dataclasses.asdict` gives their JSON form. The specification behind these fingerprints is archived
by [`EnvironmentArchive`][bayes_cep.run.environment_archive.EnvironmentArchive].

Constants:
    LOCK_FILE_NAME: Name of the pixi lock file in the repository root.

Classes:
    GitState: Commit and uncommitted changes of one git repository.
    GitRepository: Read the state and the uncommitted changes of one git repository.
    EditablePackage: An editable installed package and the git state of its source directory.
    Environment: What code the process runs, for the study description and each run.
"""

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Self
from urllib.parse import unquote, urlparse

from bayes_cep.run.directories import REPOSITORY_ROOT

LOCK_FILE_NAME = "pixi.lock"
PIXI_ENVIRONMENT_VARIABLE = "PIXI_ENVIRONMENT_NAME"
GIT_TIMEOUT_SECONDS = 10


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
