"""Base class of run configurations: JSON round trip, content-hash run id and text overview.

A configuration is a frozen dataclass holding paths and settings only, never arrays. Its JSON form
tags every nested dataclass with a `__type__` entry (the class name), so that e.g. two different
optimizer strategies never serialize identically. Decoding is driven by the type hints of the
fields, with no registry: a union field is resolved through the `__type__` tag.

Classes:
    RunConfig: Base class of all run configurations.

Constants:
    REPOSITORY_ROOT: Root of the repository, against which configured paths are resolved.

Functions:
    resolve_path: Resolve a configured path against the repository root.
    to_jsonable: Convert a (nested) configuration value to JSON-compatible data.
    canonical_json: Deterministic JSON string of a configuration value.
    format_tree: Indented text overview of JSON-compatible configuration data.
"""

import hashlib
import json
import numbers
import types
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Self, Union, get_args, get_origin, get_type_hints

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RUN_ID_LENGTH = 8
TYPE_KEY = "__type__"


# ==================================================================================================
def resolve_path(path: Path) -> Path:
    """Resolve a relative path against the repository root, so runs do not depend on the working
    directory; absolute paths stay as they are."""
    return path if path.is_absolute() else REPOSITORY_ROOT / path


# ==================================================================================================
def to_jsonable(value: Any) -> Any:
    """Convert a configuration value to JSON-compatible data.

    Args:
        value (Any): Dataclass, path, enum, number, string, `None`, or tuple, list or dict of
            these.

    Returns:
        Any: JSON-compatible data; dataclasses become dicts of their fields plus `__type__`.

    Raises:
        TypeError: For arrays and any other unsupported type; arrays do not belong in a
            configuration.
    """
    if is_dataclass(value) and not isinstance(value, type):
        entries = {field.name: to_jsonable(getattr(value, field.name)) for field in fields(value)}
        return {TYPE_KEY: type(value).__name__, **entries}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, tuple | list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"Cannot serialize {type(value).__name__} in a configuration.")


# ==================================================================================================
def canonical_json(value: Any) -> str:
    """Deterministic JSON string: sorted keys, no whitespace."""
    return json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"))


# ==================================================================================================
def _decode_dataclass(data: Any, cls: type) -> Any:
    if not isinstance(data, dict) or data.get(TYPE_KEY) != cls.__name__:
        raise ValueError(f"Expected a {cls.__name__} entry, got {data!r}.")
    hints = get_type_hints(cls)
    names = {field.name for field in fields(cls) if field.init}
    unknown = set(data) - names - {TYPE_KEY}
    if unknown:
        raise ValueError(f"{cls.__name__} has no fields {sorted(unknown)}.")
    return cls(**{name: _decode(data[name], hints[name]) for name in names if name in data})


# --------------------------------------------------------------------------------------------------
def _decode_union(value: Any, members: tuple[Any, ...]) -> Any:
    if value is None:
        if type(None) in members:
            return None
        raise ValueError(f"None is not allowed in {members}.")
    if isinstance(value, dict) and TYPE_KEY in value:
        for member in members:
            if is_dataclass(member) and member.__name__ == value[TYPE_KEY]:
                return _decode_dataclass(value, member)
        raise ValueError(f"No member of {members} is named {value[TYPE_KEY]!r}.")
    for member in members:
        if member is type(None) or is_dataclass(member):
            continue
        try:
            return _decode(value, member)
        except TypeError, ValueError, KeyError:
            continue
    raise ValueError(f"Cannot decode {value!r} as any of {members}.")


# --------------------------------------------------------------------------------------------------
def _decode(value: Any, annotation: Any) -> Any:
    """Rebuild a value from its JSON form, following the type annotation of its field."""
    origin = get_origin(annotation)
    if annotation is Any or origin is Literal:
        return value
    if origin in (Union, types.UnionType):
        return _decode_union(value, get_args(annotation))
    if origin is tuple:
        args = get_args(annotation)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode(item, args[0]) for item in value)
        return tuple(_decode(item, arg) for item, arg in zip(value, args, strict=True))
    if origin is list:
        (item_type,) = get_args(annotation)
        return [_decode(item, item_type) for item in value]
    if origin is dict:
        _, value_type = get_args(annotation)
        return {key: _decode(item, value_type) for key, item in value.items()}
    if isinstance(annotation, type):
        if is_dataclass(annotation):
            return _decode_dataclass(value, annotation)
        if issubclass(annotation, Enum):
            return annotation[value]
        if annotation is Path:
            return Path(value)
        if annotation is float and isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
        if annotation is numbers.Real and isinstance(value, int | float):
            return value
        if annotation in (int, str, bool) and type(value) is annotation:
            return value
    raise TypeError(f"Cannot decode {value!r} as {annotation}.")


# ==================================================================================================
@dataclass(frozen=True)
class RunConfig:
    """Base class of run configurations; subclasses are frozen dataclasses of paths and settings.

    The JSON form round-trips: `Config.from_dict(config.to_dict()) == config`. Equal configurations
    have equal run ids, independent of how they were constructed.
    """

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible form, with a `__type__` entry for every nested dataclass."""
        return to_jsonable(self)

    # ----------------------------------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Rebuild a configuration from `to_dict` output.

        Raises:
            ValueError: If `data` does not describe this class, or contains unknown fields.
            TypeError: If a value does not fit the type of its field.
        """
        return _decode_dataclass(data, cls)

    # ----------------------------------------------------------------------------------------------
    def write_json(self, path: Path) -> None:
        """Write the JSON form to `path`."""
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    # ----------------------------------------------------------------------------------------------
    @classmethod
    def read_json(cls, path: Path) -> Self:
        """Read a configuration written by `write_json` (or a run's `config.json`)."""
        return cls.from_dict(json.loads(path.read_text()))

    # ----------------------------------------------------------------------------------------------
    @property
    def run_id(self) -> str:
        """First `RUN_ID_LENGTH` hex digits of the sha256 of the canonical JSON."""
        return hashlib.sha256(canonical_json(self).encode()).hexdigest()[:RUN_ID_LENGTH]

    # ----------------------------------------------------------------------------------------------
    def describe(self) -> str:
        """Indented text overview of all settings."""
        return format_tree(self.to_dict())


# ==================================================================================================
def format_tree(data: dict[str, Any], level: int = 0) -> str:
    """Indented text overview of JSON-compatible configuration data.

    Nested dicts become indented blocks headed by their key and `__type__`; other values are
    aligned in one column per block.

    Args:
        data (dict[str, Any]): Configuration in JSON form (`RunConfig.to_dict` or a `config.json`).
        level (int): Indentation level of the block. Defaults to `0`.

    Returns:
        str: The overview, one line per setting.
    """
    pad = "  " * (level + 1)
    lines = [data[TYPE_KEY]] if level == 0 and TYPE_KEY in data else []
    keys = [key for key in data if key != TYPE_KEY]
    width = max((len(key) for key in keys), default=0)
    for key in keys:
        value = data[key]
        if isinstance(value, dict):
            tag = f" ({value[TYPE_KEY]})" if TYPE_KEY in value else ""
            lines.append(f"{pad}{key}{tag}:")
            lines.append(format_tree(value, level + 1))
        else:
            lines.append(f"{pad}{key:<{width}} : {value}")
    return "\n".join(line for line in lines if line)
