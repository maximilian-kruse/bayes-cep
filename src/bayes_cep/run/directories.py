"""The directories of runs and studies on disk, and the state of a run.

```
<study root>/                 StudyDirectory
  study/       study.json, environment.json,
               environment/ (pixi.lock, pyproject.toml, conda spec, source.patch)
  runs/<id>/                  RunDirectory: config.json, metadata.json, status.json,
                              run.log, metrics.json and the result files of the run
  summary/     written by `write_run_table`
  jobs/        submitit job folders and logs (SLURM only)
```

A run that was never submitted or started has no directory and counts as pending.

This module owns the layout above, the atomic writing of the JSON records, the repository root
against which configured paths are resolved, and the format of recorded times. The result files
that only a run itself produces are known to the run.

Constants:
    REPOSITORY_ROOT: Root of the repository.

Classes:
    RunState: The states of a run.
    RunDirectory: One run's directory: its log, its JSON records, its configuration and state.
    StudyDirectory: A study's directory: its description and its run directories.

Functions:
    resolve_repository_path: Resolve a configured path against the repository root.
    format_current_time: The current time as an ISO 8601 string with time zone.
    write_json_record: Write a JSON file atomically.
"""

import json
import os
import shutil
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from bayes_cep.run.config import JsonValue, RunConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CONFIG_RECORD = "config.json"
METADATA_RECORD = "metadata.json"
METRICS_RECORD = "metrics.json"
STATUS_RECORD = "status.json"
LOG_FILE_NAME = "run.log"
RESULTS_DIR_NAME = "results"


# ==================================================================================================
class RunState(StrEnum):
    """The state of a run; also its value in `status.json`.

    `SUBMITTED` is written by the submitter before a job may start, so that a queued run is told
    apart from one that was never submitted. `RUNNING` and `SUBMITTED` are the *active* states.
    """

    PENDING = "pending"
    SUBMITTED = "submitted"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"

    @property
    def is_active(self) -> bool:
        """Whether a job may currently be working on the run (or has died while doing so)."""
        return self in (RunState.SUBMITTED, RunState.RUNNING)


# ==================================================================================================
def resolve_repository_path(path: Path) -> Path:
    """Resolve a relative path against the repository root, so runs do not depend on the working
    directory; absolute paths stay as they are."""
    return path if path.is_absolute() else REPOSITORY_ROOT / path


# --------------------------------------------------------------------------------------------------
def format_current_time() -> str:
    """The current time, ISO 8601 with time zone and second resolution."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


# --------------------------------------------------------------------------------------------------
def write_json_record(path: Path, data: JsonValue) -> None:
    """Write `data` as JSON via a temporary file and a rename, so readers never see a partial file.

    Missing parent folders are created. The temporary file is named after the process, so that
    concurrent writers of the same record do not clobber each other's temporary file.

    Args:
        path (Path): Target file.
        data (JsonValue): Plain JSON data; numpy scalars and the like are not accepted.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temporary, path)


# ==================================================================================================
class RunDirectory:
    """The directory of one run; its records are directly in it.

    Attributes:
        path (Path): The directory; it need not exist yet.
    """

    def __init__(self, path: Path) -> None:
        """Wrap the directory `path`."""
        self.path = path

    # ----------------------------------------------------------------------------------------------
    @property
    def log_path(self) -> Path:
        """The log file of the run."""
        return self.path / LOG_FILE_NAME

    # ----------------------------------------------------------------------------------------------
    def clear_for_new_attempt(self) -> None:
        """Remove everything of a previous attempt except the status record.

        The status is kept so that a submitted run stays recognizable as active until its job
        records the start.
        """
        if not self.path.exists():
            return
        for entry in self.path.iterdir():
            if entry.name == STATUS_RECORD:
                continue
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()

    # ----------------------------------------------------------------------------------------------
    def record_submitted(self) -> None:
        """Record that a job for the run is queued and may start any time, as `SUBMITTED`."""
        self._write_state(RunState.SUBMITTED)

    # ----------------------------------------------------------------------------------------------
    def record_start(self, config: RunConfig, metadata: dict[str, Any]) -> None:
        """Record the configuration and metadata of a run that starts now, as `RUNNING`.

        The status starts afresh, and the metrics of any earlier attempt are removed, so that
        nothing of an earlier attempt can be mistaken for the result of this one.
        """
        (self.path / METRICS_RECORD).unlink(missing_ok=True)
        write_json_record(self.path / CONFIG_RECORD, config.to_json_dict())
        write_json_record(self.path / METADATA_RECORD, metadata)
        self._write_state(RunState.RUNNING, started=format_current_time())

    # ----------------------------------------------------------------------------------------------
    def record_success(self, metrics: dict[str, Any]) -> None:
        """Record the metrics and the end of a run that finished, as `DONE`."""
        write_json_record(self.path / METRICS_RECORD, metrics)
        self._write_state(
            RunState.DONE,
            started=self._read_state().get("started"),
            finished=format_current_time(),
        )

    # ----------------------------------------------------------------------------------------------
    def record_failure(self, error: BaseException, formatted_traceback: str) -> None:
        """Record that, and why, a run failed or was interrupted, as `FAILED`."""
        self._write_state(
            RunState.FAILED,
            started=self._read_state().get("started"),
            finished=format_current_time(),
            error=f"{type(error).__name__}: {error}",
            traceback=formatted_traceback,
        )

    # ----------------------------------------------------------------------------------------------
    def read_metrics(self) -> dict[str, Any]:
        """The recorded metrics; empty unless the run is `DONE`."""
        if self.read_state() != RunState.DONE:
            return {}
        return json.loads((self.path / METRICS_RECORD).read_text())

    # ----------------------------------------------------------------------------------------------
    def read_state(self) -> RunState:
        """The state of the run; `PENDING` if it has no status file."""
        status = self._read_state()
        return RunState(status["state"]) if status else RunState.PENDING

    # ----------------------------------------------------------------------------------------------
    def _read_state(self) -> dict[str, Any]:
        """The content of `status.json`; empty if there is none."""
        path = self.path / STATUS_RECORD
        return json.loads(path.read_text()) if path.exists() else {}

    # ----------------------------------------------------------------------------------------------
    def _write_state(self, state: RunState, **entries: Any) -> None:
        """Replace `status.json` by `state` and the entries that are set."""
        status = {"state": state.value}
        status.update({name: value for name, value in entries.items() if value is not None})
        write_json_record(self.path / STATUS_RECORD, status)


# ==================================================================================================
class StudyDirectory:
    """The directory of one study.

    Attributes:
        path (Path): The directory.
        study_record_path (Path): Description of the study and its run ids (`study/study.json`).
        environment_path (Path): The environment recorded at creation (`study/environment.json`).
        environment_dir (Path): The archived environment specification (`study/environment/`).
        summary_dir (Path): Where the cross-run summary is written.
        job_dir (Path): Where submitit writes its job folders and logs (SLURM jobs only).
    """

    def __init__(self, path: Path) -> None:
        """Wrap the study directory `path`."""
        self.path = path
        self.study_record_path = path / "study" / "study.json"
        self.environment_path = path / "study" / "environment.json"
        self.environment_dir = path / "study" / "environment"
        self.summary_dir = path / "summary"
        self.job_dir = path / "jobs"

    # ----------------------------------------------------------------------------------------------
    def run_directory(self, run_id: str) -> RunDirectory:
        """The directory of the run with this id."""
        return RunDirectory(self.path / "runs" / run_id)
