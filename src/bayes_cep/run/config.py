"""Base class of run configurations: JSON round trip, content-hash run id and text overview.

A configuration is a frozen dataclass holding paths and settings only, never arrays. Its JSON form
tags every nested dataclass with a `__type__` entry (the class name), so that e.g. two different
optimizer strategies never serialize identically. Decoding is driven by the type hints of the
fields, with no registry: a union field is resolved through the `__type__` tag. Class names are
therefore part of the identity of a configuration: renaming a class changes the run ids of all
configurations containing it, and two classes of the same name cannot share a union.

Classes:
    RunConfig: Base class of all run configurations.
    ConfigCodec: Conversion of configuration values to and from JSON-compatible data.

Functions:
    format_config_tree: Indented text overview of JSON-compatible configuration data.
"""

import hashlib
import json
import math
import numbers
import types
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from functools import cache, cached_property
from pathlib import Path
from typing import Any, Literal, Self, Union, get_args, get_origin, get_type_hints

import numpy as np

RUN_ID_LENGTH = 8
TYPE_KEY = "__type__"

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonDict = dict[str, JsonValue]


# ==================================================================================================
class ConfigCodec:
    """Conversion of configuration values to and from JSON-compatible data.

    Encoding follows the runtime type of a value, decoding follows the type annotation of the field
    that holds it. The two are inverse for everything a configuration may contain: dataclasses,
    paths, enums, finite numbers, strings, `None`, and tuples, lists and string-keyed dicts of
    these. Numbers are coerced to the annotated type on decoding (an `int` in a `float` field
    becomes a `float`).
    """

    @classmethod
    def encode(cls, value: object) -> JsonValue:
        """Convert a configuration value to JSON-compatible data.

        Args:
            value (object): Dataclass, path, enum, finite number, string, `None`, or tuple, list or
                dict (with string keys) of these.

        Returns:
            JsonValue: JSON-compatible data; dataclasses become dicts of their fields plus
                `__type__`.

        Raises:
            TypeError: For arrays, non-string dict keys and any other unsupported type; arrays do
                not belong in a configuration.
            ValueError: For `NaN` and infinite numbers.
        """
        if is_dataclass(value) and not isinstance(value, type):
            entries = {f.name: cls.encode(getattr(value, f.name)) for f in fields(value) if f.init}
            return {TYPE_KEY: type(value).__name__, **entries}
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Enum):
            return value.name
        if isinstance(value, tuple | list):
            return [cls.encode(item) for item in value]
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise TypeError(f"Dict keys in a configuration must be strings, got {list(value)}.")
            return {key: cls.encode(item) for key, item in value.items()}
        if isinstance(value, np.generic):
            return cls.encode(value.item())
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(
                    f"Cannot serialize the non-finite number {value} in a configuration."
                )
            return value + 0.0  # turns -0.0 into 0.0, which hash and compare equal
        if value is None or isinstance(value, bool | int | str):
            return value
        raise TypeError(f"Cannot serialize {type(value).__name__} in a configuration.")

    # ----------------------------------------------------------------------------------------------
    @classmethod
    def encode_canonical_json(cls, value: object) -> str:
        """Deterministic JSON string of a value: sorted keys, no whitespace."""
        return json.dumps(cls.encode(value), sort_keys=True, separators=(",", ":"))

    # ----------------------------------------------------------------------------------------------
    @classmethod
    def decode_dataclass(cls, data: object, dataclass_type: type) -> Any:
        """Rebuild a dataclass instance from its JSON form.

        Raises:
            ValueError: If `data` does not describe `dataclass_type`, or contains unknown fields,
                or a `Literal` or enum value is not allowed.
            TypeError: If a value does not fit the type of its field.
        """
        if not isinstance(data, dict) or data.get(TYPE_KEY) != dataclass_type.__name__:
            raise ValueError(f"Expected a {dataclass_type.__name__} entry, got {data!r}.")
        hints = cls._field_types(dataclass_type)
        names = {f.name for f in fields(dataclass_type) if f.init}
        unknown = set(data) - names - {TYPE_KEY}
        if unknown:
            raise ValueError(f"{dataclass_type.__name__} has no fields {sorted(unknown)}.")
        return dataclass_type(
            **{name: cls.decode(data[name], hints[name]) for name in names if name in data}
        )

    # ----------------------------------------------------------------------------------------------
    @classmethod
    def decode(cls, value: object, annotation: Any) -> Any:
        """Rebuild a value from its JSON form, following the type annotation of its field."""
        origin = get_origin(annotation)
        if annotation is Any:
            return value
        if origin is Literal:
            if value not in get_args(annotation):
                raise ValueError(f"{value!r} is not one of {get_args(annotation)}.")
            return value
        if origin in (Union, types.UnionType):
            return cls._decode_union(value, get_args(annotation))
        if origin is tuple:
            args = get_args(annotation)
            if len(args) == 2 and args[1] is Ellipsis:
                return tuple(cls.decode(item, args[0]) for item in value)
            return tuple(cls.decode(item, arg) for item, arg in zip(value, args, strict=True))
        if origin is list:
            (item_type,) = get_args(annotation)
            return [cls.decode(item, item_type) for item in value]
        if origin is dict:
            _, value_type = get_args(annotation)
            return {key: cls.decode(item, value_type) for key, item in value.items()}
        if isinstance(annotation, type):
            if is_dataclass(annotation):
                return cls.decode_dataclass(value, annotation)
            if issubclass(annotation, Enum):
                if value not in annotation.__members__:
                    raise ValueError(f"{value!r} is not a member of {annotation.__name__}.")
                return annotation[value]
            if annotation is Path:
                return Path(value)
            if (
                annotation is float
                and isinstance(value, int | float)
                and not isinstance(value, bool)
            ):
                return float(value)
            if annotation is numbers.Real and isinstance(value, int | float):
                return value  # used by settings of ls_bayesian, e.g. CustomLBFGSSettings
            if annotation in (int, str, bool) and type(value) is annotation:
                return value
        raise TypeError(f"Cannot decode {value!r} as {annotation}.")

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    @cache
    def _field_types(dataclass_type: type) -> dict[str, Any]:
        """Resolved type hints of a dataclass, computed once per class."""
        return get_type_hints(dataclass_type)

    # ----------------------------------------------------------------------------------------------
    @classmethod
    def _decode_union(cls, value: object, members: tuple[Any, ...]) -> Any:
        """Decode into one member of a union: by `__type__` tag for dataclasses, else by trying."""
        if value is None:
            if type(None) in members:
                return None
            raise ValueError(f"None is not allowed in {members}.")
        if isinstance(value, dict) and TYPE_KEY in value:
            matches = [m for m in members if is_dataclass(m) and m.__name__ == value[TYPE_KEY]]
            if len(matches) != 1:
                tag = value[TYPE_KEY]
                raise ValueError(f"{len(matches)} members of {members} are named {tag!r}, need 1.")
            return cls.decode_dataclass(value, matches[0])
        for member in members:
            if member is type(None) or is_dataclass(member):
                continue
            try:
                return cls.decode(value, member)
            except TypeError, ValueError:
                continue
        raise ValueError(f"Cannot decode {value!r} as any of {members}.")


# ==================================================================================================
@dataclass(frozen=True)
class RunConfig:
    """Base class of run configurations; subclasses are frozen dataclasses of paths and settings.

    The JSON form round-trips: `Config.from_json_dict(config.to_json_dict()) == config`. Equal
    configurations have equal run ids, independent of how they were constructed (the id is
    computed from the decoded form, so `1` and `1.0` in a `float` field give the same id).
    """

    def to_json_dict(self) -> JsonDict:
        """JSON-compatible form, with a `__type__` entry for every nested dataclass."""
        return ConfigCodec.encode(self)

    # ----------------------------------------------------------------------------------------------
    @classmethod
    def from_json_dict(cls, data: JsonDict) -> Self:
        """Rebuild a configuration from `to_json_dict` output.

        Raises:
            ValueError: If `data` does not describe this class, or contains unknown fields.
            TypeError: If a value does not fit the type of its field.
        """
        return ConfigCodec.decode_dataclass(data, cls)

    # ----------------------------------------------------------------------------------------------
    @cached_property
    def run_id(self) -> str:
        """First `RUN_ID_LENGTH` hex digits of the sha256 of the canonical JSON.

        The canonical JSON is taken after a decode round trip, which coerces every value to the
        type of its field. The id is computed once per configuration.
        """
        normalized = type(self).from_json_dict(self.to_json_dict())
        return hashlib.sha256(ConfigCodec.encode_canonical_json(normalized).encode()).hexdigest()[
            :RUN_ID_LENGTH
        ]

    # ----------------------------------------------------------------------------------------------
    def with_overrides(self, overrides: dict[str, Any]) -> Self:
        """A copy with nested fields replaced, addressed by dotted paths.

        Args:
            overrides (dict[str, Any]): Dotted path to new value, e.g. `{"prior.kappa": 0.1}`.

        Returns:
            Self: The updated configuration; this one is not modified.

        Raises:
            ValueError: If a path does not address a field of the configuration.
        """
        config = self
        for path, value in overrides.items():
            config = RunConfig._replace_field(config, path, path.split("."), value)
        return config

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _replace_field(obj: Any, full_path: str, parts: list[str], value: Any) -> Any:
        """Copy of the dataclass `obj` with the field at `parts` replaced."""
        head, rest = parts[0], parts[1:]
        if not is_dataclass(obj) or head not in {field.name for field in fields(obj)}:
            raise ValueError(f"{full_path!r} is not a field of the configuration (at {head!r}).")
        if rest:
            value = RunConfig._replace_field(getattr(obj, head), full_path, rest, value)
        return replace(obj, **{head: value})

    # ----------------------------------------------------------------------------------------------
    def describe(self) -> str:
        """Indented text overview of all settings."""
        return format_config_tree(self.to_json_dict())


# ==================================================================================================
def format_config_tree(data: dict[str, Any], level: int = 0) -> str:
    """Indented text overview of JSON-compatible configuration data.

    Nested dicts become indented blocks headed by their key and `__type__`; other values are
    aligned in one column per block.

    Args:
        data (dict[str, Any]): Configuration in JSON form (`RunConfig.to_json_dict` or a
            `config.json`).
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
            lines.append(format_config_tree(value, level + 1))
        else:
            lines.append(f"{pad}{key:<{width}} : {value}")
    return "\n".join(line for line in lines if line)
