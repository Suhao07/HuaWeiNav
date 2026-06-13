"""Detector vocabulary metadata for real-robot grounding.

The vocabulary layer records what the detector was asked to detect and how a
raw detector label relates to the detector configuration. It does not decide
whether a label satisfies a user instruction; concept grounding and final
verification own that semantic decision.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import yaml


SYSNAV_OBJECTS_YAML_RELATIVE = "src/semantic_mapping/semantic_mapping/config/objects.yaml"


@dataclass(frozen=True)
class DetectorLabelEntry:
    """One configured detector concept and its prompt strings."""

    canonical_label: str
    prompts: Tuple[str, ...] = ()
    is_instance: bool = True
    raw_config: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly label entry."""

        return {
            "canonical_label": self.canonical_label,
            "prompts": list(self.prompts),
            "is_instance": self.is_instance,
            "raw_config": self.raw_config,
        }


@dataclass(frozen=True)
class DetectorVocabulary:
    """Detector label space and provenance helpers."""

    detector_name: str
    config_path: Optional[str] = None
    entries: Tuple[DetectorLabelEntry, ...] = ()

    @property
    def label_space(self) -> Tuple[str, ...]:
        """Return all canonical detector labels."""

        return tuple(entry.canonical_label for entry in self.entries)

    @property
    def prompt_space(self) -> Tuple[str, ...]:
        """Return all prompt strings used by the detector."""

        prompts = []
        for entry in self.entries:
            prompts.extend(entry.prompts)
        return tuple(prompts)

    def find_entry(self, label: str) -> Optional[DetectorLabelEntry]:
        """Find a config entry by raw detector label or configured prompt."""

        normalized = _normalize_label(label)
        for entry in self.entries:
            # 核心：这里只匹配 detector config 内的 canonical/prompt 字面值，不做语义 alias 推断。
            if normalized == _normalize_label(entry.canonical_label):
                return entry
            if normalized in {_normalize_label(prompt) for prompt in entry.prompts}:
                return entry
        return None

    def provenance_for(self, raw_label: str) -> Dict[str, Any]:
        """Return label provenance without changing the raw detector label."""

        entry = self.find_entry(raw_label)
        matched_by = None
        if entry is not None:
            raw_norm = _normalize_label(raw_label)
            matched_by = "canonical_label" if raw_norm == _normalize_label(entry.canonical_label) else "prompt"
        return {
            "detector_name": self.detector_name,
            "config_path": self.config_path,
            "raw_detector_label": raw_label,
            "normalized_detector_label": _normalize_label(raw_label),
            "known_in_detector_vocabulary": entry is not None,
            "matched_by": matched_by,
            "canonical_label": entry.canonical_label if entry is not None else None,
            "prompt_labels": list(entry.prompts) if entry is not None else [],
            "is_instance": entry.is_instance if entry is not None else None,
        }

    def as_context(self) -> Dict[str, Any]:
        """Return compact vocabulary context for concept grounding prompts."""

        return {
            "detector_name": self.detector_name,
            "config_path": self.config_path,
            "label_space": list(self.label_space),
            "prompt_space": list(self.prompt_space),
            "entries": [entry.as_dict() for entry in self.entries],
        }


class DetectorVocabularyAdapter:
    """Load detector vocabulary metadata from SysNav-compatible config files."""

    @staticmethod
    def from_sysnav_objects_yaml(
        config_path: str,
        detector_name: str = "sysnav_detection_node",
    ) -> DetectorVocabulary:
        """Load SysNav ``objects.yaml`` into a detector vocabulary."""

        path = Path(config_path)
        with path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}

        prompts_config = payload.get("prompts") or {}
        entries = []
        for canonical_label, config in prompts_config.items():
            config = dict(config or {})
            prompts = tuple(str(prompt).strip() for prompt in config.get("prompts", []) if str(prompt).strip())
            entries.append(
                DetectorLabelEntry(
                    canonical_label=str(canonical_label),
                    prompts=prompts,
                    is_instance=bool(config.get("is_instance", True)),
                    raw_config=config,
                )
            )

        return DetectorVocabulary(
            detector_name=detector_name,
            config_path=str(path),
            entries=tuple(entries),
        )

    @staticmethod
    def from_sysnav_root(
        sysnav_root: str,
        detector_name: str = "sysnav_detection_node",
    ) -> DetectorVocabulary:
        """Load SysNav detector vocabulary from a SysNav repository root."""

        return DetectorVocabularyAdapter.from_sysnav_objects_yaml(
            str(Path(sysnav_root) / SYSNAV_OBJECTS_YAML_RELATIVE),
            detector_name=detector_name,
        )

    @staticmethod
    def from_environment(
        detector_name: str = "sysnav_detection_node",
        config_env: str = "SYSNAV_OBJECTS_YAML",
        root_env: str = "SYSNAV_ROOT",
    ) -> DetectorVocabulary:
        """Load vocabulary from environment variables used by deployment scripts."""

        config_path = os.getenv(config_env)
        if config_path:
            return DetectorVocabularyAdapter.from_sysnav_objects_yaml(config_path, detector_name=detector_name)
        sysnav_root = os.getenv(root_env)
        if sysnav_root:
            return DetectorVocabularyAdapter.from_sysnav_root(sysnav_root, detector_name=detector_name)
        raise RuntimeError(
            f"Set {config_env} or {root_env} to load the SysNav detector vocabulary"
        )


def merge_label_provenance(metadata: Optional[Dict[str, Any]], provenance: Dict[str, Any]) -> Dict[str, Any]:
    """Return metadata with detector label provenance attached."""

    merged = dict(metadata or {})
    # 核心：保留 raw label 和 detector config 来源，让 concept grounding 显式处理词表差异。
    merged["label_provenance"] = provenance
    return merged


def vocabulary_context(vocabulary: Optional[DetectorVocabulary]) -> Dict[str, Any]:
    """Return empty context when no detector vocabulary has been configured."""

    return vocabulary.as_context() if vocabulary is not None else {}


def _normalize_label(label: str) -> str:
    """Normalize only spelling format used by detector configs."""

    return " ".join(str(label or "").strip().lower().replace("_", " ").split())
