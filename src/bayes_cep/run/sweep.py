"""Sweeps: how the configuration parameters of a study vary from run to run.

A sweep is a tree of nodes. Each node expands into a list of override dicts (dotted configuration
path to value), which are applied to the base configuration of the study.

Classes:
    SweepNode: A variation of configuration parameters, expanding into override dicts.
    Axis: One parameter and its values.
    Zip: Axes varied together.
    Product: All combinations of groups.
"""

import itertools
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import is_dataclass
from typing import Any, override

from bayes_cep.run.config import ConfigCodec


# ==================================================================================================
class SweepNode(ABC):
    """A variation of configuration parameters: a tree of axes, zips and products."""

    @abstractmethod
    def expand_overrides(self) -> list[dict[str, Any]]:
        """One override dict (dotted path to value) per combination."""

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def to_json_dict(self) -> dict[str, Any]:
        """JSON-compatible form."""

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def describe(self, level: int = 0) -> str:
        """Indented text overview, starting at indentation `level`."""


# ==================================================================================================
class Axis(SweepNode):
    """One parameter, addressed by dotted path, and the values it takes.

    Attributes:
        path (str): Dotted path into the base configuration, e.g. `"prior.kappa"`.
        values (tuple[Any, ...]): Values of the parameter.
    """

    def __init__(self, path: str, values: Sequence[Any]) -> None:
        """Vary the parameter at `path` over `values`.

        Raises:
            ValueError: If there are no values.
        """
        if len(values) == 0:
            raise ValueError(f"Axis {path!r} has no values.")
        self.path = path
        self.values = tuple(values)

    @override
    def expand_overrides(self) -> list[dict[str, Any]]:
        return [{self.path: value} for value in self.values]

    @override
    def to_json_dict(self) -> dict[str, Any]:
        return {"kind": "axis", "path": self.path, "values": ConfigCodec.encode(self.values)}

    @override
    def describe(self, level: int = 0) -> str:
        labels = ", ".join(self._label(value) for value in self.values)
        return f"{'  ' * level}{self.path}: {labels}"

    @staticmethod
    def _label(value: Any) -> str:
        """A value for display: scalars as they are, config objects by their class name."""
        return type(value).__name__ if is_dataclass(value) else str(value)


# ==================================================================================================
class Zip(SweepNode):
    """Axes varied together: the i-th value of every axis forms one combination."""

    def __init__(self, *axes: Axis) -> None:
        """Group axes of equal length.

        Raises:
            ValueError: If there are no axes or their lengths differ.
        """
        if not axes:
            raise ValueError("Zip needs at least one axis.")
        lengths = {len(axis.values) for axis in axes}
        if len(lengths) != 1:
            raise ValueError(f"Zip axes must have equal lengths, got {sorted(lengths)}.")
        self.axes = axes

    @override
    def expand_overrides(self) -> list[dict[str, Any]]:
        return [
            {axis.path: axis.values[position] for axis in self.axes}
            for position in range(len(self.axes[0].values))
        ]

    @override
    def to_json_dict(self) -> dict[str, Any]:
        return {"kind": "zip", "axes": [axis.to_json_dict() for axis in self.axes]}

    @override
    def describe(self, level: int = 0) -> str:
        lines = [f"{'  ' * level}zip"] + [axis.describe(level + 1) for axis in self.axes]
        return "\n".join(lines)


# ==================================================================================================
class Product(SweepNode):
    """All combinations of its groups; the last group varies fastest. No groups: a single run."""

    def __init__(self, *groups: SweepNode) -> None:
        """Combine axes, zips and further products."""
        self.groups = groups

    @override
    def expand_overrides(self) -> list[dict[str, Any]]:
        merged = []
        for parts in itertools.product(*(group.expand_overrides() for group in self.groups)):
            combination: dict[str, Any] = {}
            for part in parts:
                combination.update(part)
            merged.append(combination)
        return merged

    @override
    def to_json_dict(self) -> dict[str, Any]:
        return {"kind": "product", "groups": [group.to_json_dict() for group in self.groups]}

    @override
    def describe(self, level: int = 0) -> str:
        lines = [f"{'  ' * level}product"] + [group.describe(level + 1) for group in self.groups]
        return "\n".join(lines)
