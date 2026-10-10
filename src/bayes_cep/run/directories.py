"""The directories of runs and studies on disk, and the state of a run.

```
<study_name>/                 StudyDirectory
  study/       study.json, runs.json, environment.json, definition.py,
               environment/ (pixi.lock, pyproject.toml, conda spec, source.patch)
  runs/<id>/                  RunDirectory: config.json, metadata.json, status.json,
                              run.log, metrics.json and the result files of the run
  summary/     written by the collector
  jobs/        submitit job folders and logs (local and SLURM)
```

A run that has not started has no directory and counts as pending.

This module owns the layout above, the repository root and the atomic reading and writing of the
JSON records. The result files that only a run itself produces are known to the run.

Constants:
    REPOSITORY_ROOT: Root of the repository, against which configured paths are resolved.

Classes:
    RunState: The states of a run.
    JsonRecords: The JSON records in one folder, written atomically.
    RunDirectory: One run's directory: its log, its JSON records, its configuration and state.
    StudyDirectory: A study's directory: its description and its run directories.

Functions:
    resolve_repository_path: Resolve a configured path against the repository root.
"""

import json
import os
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from bayes_cep.run.config import RunConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CONFIG_RECORD = "config.json"
METADATA_RECORD = "metadata.json"
METRICS_RECORD = "metrics.json"
STATUS_RECORD = "status.json"
LOG_FILE_NAME = "run.log"


# ==================================================================================================
class RunState(StrEnum):
    """The state of a run; also its value in `status.json`."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


# ==================================================================================================
def resolve_repository_path(path: Path) -> Path:
    """Resolve a relative path against the repository root, so runs do not depend on the working
    directory; absolute paths stay as they are."""
    return path if path.is_absolute() else REPOSITORY_ROOT / path


# ==================================================================================================
class JsonRecords:
    """The JSON records in one folder, written atomically so that readers never see a partial file.

    Attributes:
        path (Path): The folder; it need not exist yet.
    """

    def __init__(self, path: Path) -> None:
        """Wrap the folder `path`."""
        self.path = path

    # ----------------------------------------------------------------------------------------------
    def write_record(self, name: str, data: Any) -> None:
        """Write the record `name`, e.g. `"status.json"`, via a temporary file and a rename.

        Args:
            name (str): File name of the record; missing parent folders are created.
            data (Any): JSON-serializable data.
        """
        target = self.path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        os.replace(temporary, target)

    # ----------------------------------------------------------------------------------------------
    def read_record(self, name: str) -> Any:
        """Read the record `name`."""
        return json.loads((self.path / name).read_text())

    # ----------------------------------------------------------------------------------------------
    def has_record(self, name: str) -> bool:
        """Whether the record `name` exists."""
        return (self.path / name).exists()


# ==================================================================================================
class RunDirectory:
    """The directory of one run; its records are directly in it.

    Attributes:
        path (Path): The directory; it need not exist yet.
        records (JsonRecords): The JSON records of the run.
    """

    def __init__(self, path: Path) -> None:
        """Wrap the directory `path`."""
        self.path = path
        self.records = JsonRecords(path)

    # ----------------------------------------------------------------------------------------------
    @property
    def log_path(self) -> Path:
        """The log file of the run."""
        return self.path / LOG_FILE_NAME

    # ----------------------------------------------------------------------------------------------
    def write_config(self, config: RunConfig) -> None:
        """Record the configuration of the run in `config.json`."""
        self.records.write_record(CONFIG_RECORD, config.to_json_dict())

    # ----------------------------------------------------------------------------------------------
    def read_config[ConfigT: RunConfig](self, config_type: type[ConfigT]) -> ConfigT:
        """Read the recorded configuration, which must be of type `config_type`."""
        return config_type.from_json_dict(self.records.read_record(CONFIG_RECORD))

    # ----------------------------------------------------------------------------------------------
    def write_metadata(self, metadata: dict[str, Any]) -> None:
        """Record what produced the results (see `bayes_cep.run.metadata`) in `metadata.json`."""
        self.records.write_record(METADATA_RECORD, metadata)

    # ----------------------------------------------------------------------------------------------
    def write_metrics(self, metrics: dict[str, Any]) -> None:
        """Record the scalar metrics of the run in `metrics.json`."""
        self.records.write_record(METRICS_RECORD, metrics)

    # ----------------------------------------------------------------------------------------------
    def record_start(self, config: RunConfig, metadata: dict[str, Any]) -> None:
        """Record the configuration and metadata of a run that is about to start, as `RUNNING`."""
        self.write_config(config)
        self.write_metadata(metadata)
        self.write_state(RunState.RUNNING)

    # ----------------------------------------------------------------------------------------------
    def record_success(self, metrics: dict[str, Any]) -> None:
        """Record the metrics and the end of a run that finished, as `DONE`."""
        self.write_metrics(metrics)
        self.write_state(RunState.DONE)

    # ----------------------------------------------------------------------------------------------
    def record_failure(self, error: Exception, formatted_traceback: str) -> None:
        """Record that, and why, a run failed, as `FAILED`."""
        self.write_state(
            RunState.FAILED,
            error=f"{type(error).__name__}: {error}",
            traceback=formatted_traceback,
        )

    # ----------------------------------------------------------------------------------------------
    def read_metrics(self) -> dict[str, Any]:
        """The recorded metrics; empty if the run has none (pending or failed)."""
        if not self.records.has_record(METRICS_RECORD):
            return {}
        return self.records.read_record(METRICS_RECORD)

    # ----------------------------------------------------------------------------------------------
    def read_state(self) -> RunState:
        """The state of the run; `PENDING` if it has no status file."""
        if not self.records.has_record(STATUS_RECORD):
            return RunState.PENDING
        return RunState(self.records.read_record(STATUS_RECORD)["state"])

    # ----------------------------------------------------------------------------------------------
    def write_state(self, state: RunState, **entries: Any) -> None:
        """Record `state`, the current time and any further entries in `status.json`.

        Entries of an existing status file are kept, so the start time survives the final update.
        """
        status = (
            self.records.read_record(STATUS_RECORD)
            if self.records.has_record(STATUS_RECORD)
            else {}
        )
        status.update(
            state=state.value, updated=datetime.now().astimezone().isoformat(timespec="seconds")
        )
        status.update(entries)
        self.records.write_record(STATUS_RECORD, status)


# ==================================================================================================
class StudyDirectory:
    """The directory of one study.

    Attributes:
        path (Path): The directory.
        description_records (JsonRecords): The records describing the study, in `study/`.
        summary_dir (Path): Where the collector writes.
        job_dir (Path): Where submitit writes its job folders and logs (local and SLURM jobs).
        definition_path (Path): The archived study module.
        environment_dir (Path): The archived environment specification.
    """

    def __init__(self, path: Path) -> None:
        """Wrap the study directory `path`."""
        self.path = path
        self.description_records = JsonRecords(path / "study")
        self.summary_dir = path / "summary"
        self.job_dir = path / "jobs"
        self.definition_path = path / "study" / "definition.py"
        self.environment_dir = path / "study" / "environment"

    # ----------------------------------------------------------------------------------------------
    def run_directory(self, run_id: str) -> RunDirectory:
        """The directory of the run with this id."""
        return RunDirectory(self.path / "runs" / run_id)

    # ----------------------------------------------------------------------------------------------
    def read_run_index(self) -> list[dict[str, Any]]:
        """The run list recorded at creation: `index`, `id`, `overrides` and `config` per run."""
        return self.description_records.read_record("runs.json")
