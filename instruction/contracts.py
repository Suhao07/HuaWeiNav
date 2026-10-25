"""Immutable task semantics, independent of detector labels and execution state."""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from types import MappingProxyType
from typing import Any, Mapping


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported task value: {type(value).__name__}")


def _plain(value):
    if is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


@dataclass(frozen=True)
class TargetSpec:
    """A semantic target or anchor; detector vocabulary is deliberately absent."""

    id: str
    name: str
    role: str = "primary"
    terminal: bool = True
    min_count: int = 1
    aliases: tuple[str, ...] = ()
    attributes: Any = None

    def __post_init__(self):
        if not self.id or not self.name or self.min_count < 1:
            raise ValueError("Target requires an ID, concept and positive count")
        object.__setattr__(self, "aliases", tuple(self.aliases))
        object.__setattr__(self, "attributes", _freeze(self.attributes or {}))


@dataclass(frozen=True)
class ConstraintSpec:
    """A declared condition; runtime evidence never modifies this definition."""

    id: str
    type: str
    subject: str = ""
    relation: str = ""
    object: str = ""
    value: Any = None
    hardness: str = "hard"
    verifier: str = "planner"
    source: str = ""

    def __post_init__(self):
        object.__setattr__(self, "value", _freeze(self.value))


@dataclass(frozen=True)
class TaskSpec:
    """One immutable user-task revision, with no mutable progress or search mode."""

    raw_instruction: str
    targets: tuple[TargetSpec, ...]
    constraints: tuple[ConstraintSpec, ...] = ()
    revision: int = 1
    ordered: bool = False
    exhaustive: bool = False
    completion_mode: str = "any_target_success"
    search_priors: Any = None
    source: str = "language"

    def __post_init__(self):
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "search_priors", _freeze(self.search_priors or {}))
        if self.revision < 1 or not any(t.terminal for t in self.targets):
            raise ValueError("Task requires a positive revision and a terminal target")
        if len({t.id for t in self.targets}) != len(self.targets):
            raise ValueError("Target IDs must be unique")

    def as_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible representation."""
        return _plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskSpec:
        """Restore a serialized contract, preserving its semantic revision."""
        return cls(**{
            **data,
            "targets": tuple(TargetSpec(**t) for t in data["targets"]),
            "constraints": tuple(ConstraintSpec(**c) for c in data["constraints"]),
        })


@dataclass(frozen=True)
class CompileResult:
    """Explicit compilation outcome; failure never carries an executable task."""

    status: str
    task: TaskSpec | None = None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self):
        if (self.status == "success") != (self.task is not None):
            raise ValueError("Only successful compilation may contain a task")
