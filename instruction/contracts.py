"""Immutable task semantics, independent of detector labels and execution state."""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from types import MappingProxyType
from typing import Any, Mapping


RELATION_CONSTRAINT_TYPES = frozenset({"spatial", "relation", "object relation", "co occurrence"})


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
    demand: Any = None

    def __post_init__(self):
        if not self.id or not self.name or self.min_count < 1:
            raise ValueError("Target requires an ID, concept and positive count")
        object.__setattr__(self, "aliases", tuple(self.aliases))
        object.__setattr__(self, "attributes", _freeze(self.attributes or {}))
        object.__setattr__(self, "demand", _freeze(self.demand or {}))
        if self.demand and (not self.terminal or not self.demand.get("activity")
                            or not self.demand.get("required_resource")):
            raise ValueError("Demand requires a terminal resource target and an activity")


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


def interpretation_changes(current: TaskSpec | None, proposed: TaskSpec) -> dict[str, Any]:
    def obligations(task):
        if task is None:
            return {}
        data = task.as_dict()
        references = {target.id: f"target_{index}" for index, target in enumerate(task.targets)}
        values = {f"target_{index}": {key: value for key, value in target.items() if key != "id"}
                  for index, target in enumerate(data["targets"])}
        for index, constraint in enumerate(data["constraints"]):
            item = {key: value for key, value in constraint.items() if key not in {"id", "source"}}
            for key in ("subject", "object"):
                item[key] = references.get(item[key], item[key])
            values[f"constraint_{index}"] = item
        values.update({key: data[key] for key in ("ordered", "exhaustive", "completion_mode")})
        return values

    before, after = obligations(current), obligations(proposed)
    return {key: {"before": before.get(key), "after": after.get(key)}
            for key in dict.fromkeys([*before, *after]) if before.get(key) != after.get(key)}


@dataclass(frozen=True)
class CompileResult:
    """Explicit compilation outcome; failure never carries an executable task."""

    status: str
    task: TaskSpec | None = None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self):
        if (self.status == "success") != (self.task is not None):
            raise ValueError("Only successful compilation may contain a task")
