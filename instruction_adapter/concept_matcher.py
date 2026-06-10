from __future__ import annotations

import json
import os
import base64
from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np

try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ModuleNotFoundError:
    HAS_PYDANTIC = False

    class BaseModel:
        def __init__(self, **kwargs):
            for key in getattr(self, "__annotations__", {}):
                default = getattr(type(self), key, None)
                if isinstance(default, (list, dict, set)):
                    default = default.copy()
                setattr(self, key, kwargs.get(key, default))

    def Field(default=None, default_factory=None, **_kwargs):
        return default_factory() if default_factory is not None else default

from llm_utils.cognav_llm_adapter import get_client_and_model

from .contracts import ConceptQuery
from .ontology import normalize_term
from .verifier import candidate_from_object, instruction_hash


CONCEPT_MATCH_PROMPT = """
You decide whether a mapped object instance satisfies an instruction concept.

Use the concept role carefully:
- terminal concepts may satisfy the final goal.
- anchor/support concepts are only reference objects for search or relation
  verification and must never be treated as final goal success.

Do not rely on hard-coded synonym tables. Judge whether the observed object can
play the requested role for this specific instruction. Return strict JSON.
"""


class _ParsedConceptMatch(BaseModel):
    matches_concept: bool = False
    confidence: float = 0.0
    terminal_eligible: bool = False
    reason: str = ""


class _ParsedBatchConceptItem(BaseModel):
    uid: str = ""
    matches_concept: bool = False
    confidence: float = 0.0
    terminal_eligible: bool = False
    reason: str = ""


class _ParsedBatchConceptMatch(BaseModel):
    matches: list[_ParsedBatchConceptItem] = Field(default_factory=list)


@dataclass
class ConceptMatchRecord:
    instruction_hash: str
    concept_id: str
    object_uid: str
    observed_label: str
    matches_concept: bool = False
    confidence: float = 0.0
    terminal_eligible: bool = False
    source: str = "unknown"
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class RuntimeConceptMatcher:
    """Prompt-first matcher from runtime object instances to ConceptQuery.

    这里解决 book/books、shelf/bookshelf/cabinet 这类泛化问题。
    代码只做通用缓存和精确匹配；非精确语义由 LLM/VLM 判断，并把结果落盘。
    """

    def __init__(self, vlm: str = "cognav"):
        self.vlm = vlm
        self.records: dict[tuple[str, str, str], ConceptMatchRecord] = {}
        self.label_cache: dict[tuple[str, str, str], ConceptMatchRecord] = {}
        self.stats = {
            "single_llm_calls": 0,
            "batch_llm_calls": 0,
            "batch_items_requested": 0,
            "cache_hits": 0,
            "exact_matches": 0,
        }

    def reset(self):
        self.records.clear()
        self.label_cache.clear()
        for key in self.stats:
            self.stats[key] = 0

    def match_object(
        self,
        *,
        raw_instruction: str,
        concept: ConceptQuery,
        obj: Any,
        step: int | None = None,
    ) -> ConceptMatchRecord:
        candidate = candidate_from_object(obj, canonical_label=concept.name, step=step)
        observed_label = normalize_term(getattr(obj, "tag", candidate.detector_label))
        inst_hash = instruction_hash(raw_instruction)
        key = (inst_hash, concept.id, candidate.uid)
        cached = self.records.get(key)
        if cached is not None:
            self.stats["cache_hits"] += 1
            return cached

        label_key = (inst_hash, concept.id, observed_label)
        label_cached = self.label_cache.get(label_key)
        if label_cached is not None:
            self.stats["cache_hits"] += 1
            record = ConceptMatchRecord(
                instruction_hash=inst_hash,
                concept_id=concept.id,
                object_uid=candidate.uid,
                observed_label=observed_label,
                matches_concept=label_cached.matches_concept,
                confidence=label_cached.confidence,
                terminal_eligible=label_cached.terminal_eligible,
                source=label_cached.source,
                reason=label_cached.reason,
            )
            self.records[key] = record
            return record

        record = self._compute_match(
            raw_instruction=raw_instruction,
            concept=concept,
            object_uid=candidate.uid,
            observed_label=observed_label,
            obj=obj,
        )
        self.records[key] = record
        if _should_cache_label_match(record):
            self.label_cache[label_key] = record
        return record

    def match_many(
        self,
        *,
        raw_instruction: str,
        concept: ConceptQuery,
        objects: list[Any],
        step: int | None = None,
    ) -> list[ConceptMatchRecord]:
        """Batch concept-instance grounding.

        prompt-first 不等于每个物体都单独问一次 VLM。
        这里先复用 exact/cache 再把剩余候选合并成一个结构化问题
        让 LLM 一次性返回 uid->match。代码不写领域同义词规则。
        """

        out: list[ConceptMatchRecord] = []
        pending: list[tuple[Any, Any, str, str]] = []
        inst_hash = instruction_hash(raw_instruction)
        for obj in objects or []:
            candidate = candidate_from_object(obj, canonical_label=concept.name, step=step)
            observed_label = normalize_term(getattr(obj, "tag", candidate.detector_label))
            key = (inst_hash, concept.id, candidate.uid)
            cached = self.records.get(key)
            if cached is not None:
                self.stats["cache_hits"] += 1
                out.append(cached)
                continue
            label_key = (inst_hash, concept.id, observed_label)
            label_cached = self.label_cache.get(label_key)
            if label_cached is not None:
                self.stats["cache_hits"] += 1
                record = ConceptMatchRecord(
                    instruction_hash=inst_hash,
                    concept_id=concept.id,
                    object_uid=candidate.uid,
                    observed_label=observed_label,
                    matches_concept=label_cached.matches_concept,
                    confidence=label_cached.confidence,
                    terminal_eligible=label_cached.terminal_eligible,
                    source=label_cached.source,
                    reason=label_cached.reason,
                )
                self.records[key] = record
                out.append(record)
                continue
            exact = self._exact_record(
                raw_instruction=raw_instruction,
                concept=concept,
                object_uid=candidate.uid,
                observed_label=observed_label,
            )
            if exact is not None:
                self.stats["exact_matches"] += 1
                self.records[key] = exact
                self.label_cache[label_key] = exact
                out.append(exact)
                continue
            pending.append((obj, candidate, observed_label, candidate.uid))

        if not pending:
            return out
        max_batch = _concept_match_batch_limit()
        if max_batch > 0 and len(pending) > max_batch:
            pending = sorted(
                pending,
                key=lambda item: _safe_confidence(getattr(item[0], "confidence", 0.0)),
                reverse=True,
            )
            selected = pending[:max_batch]
            deferred = pending[max_batch:]
            for _obj, _candidate, observed_label, object_uid in deferred:
                record = ConceptMatchRecord(
                    instruction_hash=inst_hash,
                    concept_id=concept.id,
                    object_uid=object_uid,
                    observed_label=observed_label,
                    source="prefiltered",
                    reason=(
                        "Skipped by generic concept-match budget. Exact matches are "
                        "handled before this filter; non-exact low-priority objects "
                        "can be reconsidered after map updates."
                    ),
                )
                self.records[(inst_hash, concept.id, object_uid)] = record
                out.append(record)
            pending = selected
        if os.getenv("LLM_OFFLINE", "0").lower() in ("1", "true", "yes", "on") or not HAS_PYDANTIC:
            for _obj, _candidate, observed_label, object_uid in pending:
                record = ConceptMatchRecord(
                    instruction_hash=inst_hash,
                    concept_id=concept.id,
                    object_uid=object_uid,
                    observed_label=observed_label,
                    source="offline",
                    reason="LLM matcher unavailable and no exact match.",
                )
                self.records[(inst_hash, concept.id, object_uid)] = record
                out.append(record)
            return out

        batch_records = self._compute_batch_match(
            raw_instruction=raw_instruction,
            concept=concept,
            pending=pending,
        )
        out.extend(batch_records)
        return out

    def _compute_match(
        self,
        *,
        raw_instruction: str,
        concept: ConceptQuery,
        object_uid: str,
        observed_label: str,
        obj: Any,
    ) -> ConceptMatchRecord:
        inst_hash = instruction_hash(raw_instruction)
        exact = self._exact_record(
            raw_instruction=raw_instruction,
            concept=concept,
            object_uid=object_uid,
            observed_label=observed_label,
        )
        if exact is not None:
            self.stats["exact_matches"] += 1
            return exact

        if os.getenv("LLM_OFFLINE", "0").lower() in ("1", "true", "yes", "on") or not HAS_PYDANTIC:
            return ConceptMatchRecord(
                instruction_hash=inst_hash,
                concept_id=concept.id,
                object_uid=object_uid,
                observed_label=observed_label,
                source="offline",
                reason="LLM matcher unavailable and no exact match.",
            )

        try:
            client, model = get_client_and_model(self.vlm)
            self.stats["single_llm_calls"] += 1
            payload = {
                "instruction": raw_instruction,
                "concept": concept.as_dict(),
                "observed_object": {
                    "uid": object_uid,
                    "label": observed_label,
                    "confidence": _safe_confidence(getattr(obj, "confidence", 0.0)),
                    "position": _safe_list(getattr(obj, "position", [])),
                    "size_hint": _safe_size(obj),
                },
            }
            content: list[dict[str, Any]] = [
                {"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)}
            ]
            for block in _object_image_blocks(obj):
                content.append(block)
            completion = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": CONCEPT_MATCH_PROMPT},
                    {"role": "user", "content": content},
                ],
                response_format=_ParsedConceptMatch,
                trace_label="concept_match_single",
            )
            parsed = completion.choices[0].message.parsed or _ParsedConceptMatch()
            return ConceptMatchRecord(
                instruction_hash=inst_hash,
                concept_id=concept.id,
                object_uid=object_uid,
                observed_label=observed_label,
                matches_concept=bool(parsed.matches_concept),
                confidence=max(0.0, min(1.0, float(parsed.confidence or 0.0))),
                terminal_eligible=bool(parsed.terminal_eligible and concept.terminal),
                source="llm",
                reason=str(parsed.reason or ""),
            )
        except Exception as exc:
            return ConceptMatchRecord(
                instruction_hash=inst_hash,
                concept_id=concept.id,
                object_uid=object_uid,
                observed_label=observed_label,
                source="error",
                reason=f"concept matcher failed: {exc}",
            )

    def as_dict(self) -> dict[str, Any]:
        return {"|".join(key): value.as_dict() for key, value in self.records.items()}

    def stats_dict(self) -> dict[str, int]:
        return dict(self.stats)

    def _exact_record(
        self,
        *,
        raw_instruction: str,
        concept: ConceptQuery,
        object_uid: str,
        observed_label: str,
    ) -> ConceptMatchRecord | None:
        inst_hash = instruction_hash(raw_instruction)
        concept_terms = {normalize_term(x) for x in concept.match_terms if normalize_term(x)}
        if observed_label and observed_label in concept_terms:
            return ConceptMatchRecord(
                instruction_hash=inst_hash,
                concept_id=concept.id,
                object_uid=object_uid,
                observed_label=observed_label,
                matches_concept=True,
                confidence=1.0,
                terminal_eligible=bool(concept.terminal),
                source="exact",
                reason="Observed label exactly matches a grounded concept term.",
            )
        return None

    def _compute_batch_match(
        self,
        *,
        raw_instruction: str,
        concept: ConceptQuery,
        pending: list[tuple[Any, Any, str, str]],
    ) -> list[ConceptMatchRecord]:
        inst_hash = instruction_hash(raw_instruction)
        try:
            client, model = get_client_and_model(self.vlm)
            self.stats["batch_llm_calls"] += 1
            self.stats["batch_items_requested"] += len(pending)
            objects_payload = []
            for obj, _candidate, observed_label, object_uid in pending:
                objects_payload.append({
                    "uid": object_uid,
                    "label": observed_label,
                    "confidence": _safe_confidence(getattr(obj, "confidence", 0.0)),
                    "position": _safe_list(getattr(obj, "position", [])),
                    "size_hint": _safe_size(obj),
                })
            payload = {
                "instruction": raw_instruction,
                "concept": concept.as_dict(),
                "observed_objects": objects_payload,
                "task": (
                    "For each observed object, decide whether it satisfies the concept "
                    "for the requested role. Return one item per uid."
                ),
            }
            completion = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": CONCEPT_MATCH_PROMPT},
                    {"role": "user", "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)}]},
                ],
                response_format=_ParsedBatchConceptMatch,
                trace_label="concept_match_batch",
            )
            parsed = completion.choices[0].message.parsed or _ParsedBatchConceptMatch()
            by_uid = {str(item.uid): item for item in list(getattr(parsed, "matches", []) or [])}
            records = []
            for _obj, _candidate, observed_label, object_uid in pending:
                item = by_uid.get(object_uid)
                if item is None:
                    record = ConceptMatchRecord(
                        instruction_hash=inst_hash,
                        concept_id=concept.id,
                        object_uid=object_uid,
                        observed_label=observed_label,
                        source="llm_batch",
                        reason="Batch matcher returned no item for this uid.",
                    )
                else:
                    record = ConceptMatchRecord(
                        instruction_hash=inst_hash,
                        concept_id=concept.id,
                        object_uid=object_uid,
                        observed_label=observed_label,
                        matches_concept=bool(item.matches_concept),
                        confidence=max(0.0, min(1.0, float(item.confidence or 0.0))),
                        terminal_eligible=bool(item.terminal_eligible and concept.terminal),
                        source="llm_batch",
                        reason=str(item.reason or ""),
                    )
                self.records[(inst_hash, concept.id, object_uid)] = record
                if _should_cache_label_match(record):
                    self.label_cache[(inst_hash, concept.id, observed_label)] = record
                records.append(record)
            return records
        except Exception as exc:
            records = []
            for _obj, _candidate, observed_label, object_uid in pending:
                record = ConceptMatchRecord(
                    instruction_hash=inst_hash,
                    concept_id=concept.id,
                    object_uid=object_uid,
                    observed_label=observed_label,
                    source="error",
                    reason=f"batch concept matcher failed: {exc}",
                )
                self.records[(inst_hash, concept.id, object_uid)] = record
                records.append(record)
            return records


@dataclass
class AnchorSearchRecord:
    instruction_hash: str
    concept_id: str
    anchor_uid: str
    status: str = "candidate_anchor"
    step: int | None = None
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class AnchorSearchLedger:
    """Instruction-scoped memory for anchor-first local search.

    It blocks one searched anchor instance, not the whole anchor concept.  This
    mirrors the target verifier ledger: one bad/empty bookshelf should not
    suppress every future shelf-like object.
    """

    def __init__(self):
        self.records: dict[tuple[str, str, str], AnchorSearchRecord] = {}

    def reset(self):
        self.records.clear()

    def _key(self, raw_instruction: str, concept_id: str, anchor_uid: str) -> tuple[str, str, str]:
        return (instruction_hash(raw_instruction), str(concept_id), str(anchor_uid))

    def get(self, raw_instruction: str, concept_id: str, anchor_uid: str) -> AnchorSearchRecord | None:
        return self.records.get(self._key(raw_instruction, concept_id, anchor_uid))

    def is_blocked(self, raw_instruction: str, concept_id: str, anchor_uid: str) -> bool:
        record = self.get(raw_instruction, concept_id, anchor_uid)
        return bool(record and record.status in ("searched_no_terminal_found", "rejected_as_wrong_anchor"))

    def mark(
        self,
        *,
        raw_instruction: str,
        concept_id: str,
        anchor_uid: str,
        status: str,
        step: int | None = None,
        reason: str = "",
        evidence: dict[str, Any] | None = None,
    ) -> AnchorSearchRecord:
        record = AnchorSearchRecord(
            instruction_hash=instruction_hash(raw_instruction),
            concept_id=str(concept_id),
            anchor_uid=str(anchor_uid),
            status=status,
            step=step,
            reason=reason,
            evidence=dict(evidence or {}),
        )
        self.records[(record.instruction_hash, record.concept_id, record.anchor_uid)] = record
        return record

    def as_dict(self) -> dict[str, Any]:
        return {"|".join(key): value.as_dict() for key, value in self.records.items()}


def _safe_confidence(value: Any) -> float:
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        elif hasattr(value, "cpu") and hasattr(value, "numpy"):
            value = value.cpu().numpy().item()
        return float(value)
    except Exception:
        return 0.0


def _safe_list(value: Any) -> list[float]:
    try:
        return [float(x) for x in list(value)[:3]]
    except Exception:
        return []


def _safe_size(obj: Any) -> list[float]:
    try:
        pts = getattr(obj, "pcd").point.positions.cpu().numpy()
        if len(pts) > 0:
            return [float(x) for x in (pts.max(axis=0)[:3] - pts.min(axis=0)[:3]).tolist()]
    except Exception:
        pass
    return []


def _concept_match_batch_limit() -> int:
    try:
        return max(0, int(os.getenv("STRIVE_CONCEPT_MATCH_MAX_BATCH", "16")))
    except ValueError:
        return 16


def _should_cache_label_match(record: ConceptMatchRecord) -> bool:
    """Cache confident prompt results at role-aware label granularity.

    The cache key includes instruction hash and concept id, so this does not
    become a global synonym table. It only prevents repeated LVLM calls when
    mapper uid churn creates multiple instances with the same observed label
    under the same instruction concept.
    """

    if not record.observed_label:
        return False
    try:
        threshold = float(os.getenv("STRIVE_CONCEPT_LABEL_CACHE_CONF", "0.75"))
    except Exception:
        threshold = 0.75
    if record.source not in ("llm", "llm_batch", "exact"):
        return False
    return float(record.confidence or 0.0) >= threshold


def _object_image_blocks(obj: Any) -> list[dict[str, Any]]:
    image = getattr(obj, "rgb", None)
    if image is None:
        return []
    try:
        img = np.asarray(image)
        if img.size == 0:
            return []
        blocks = [
            {"type": "text", "text": "observed_object_full_image"},
            _image_block(img),
        ]
        bbox = getattr(obj, "bbox", None)
        if bbox is not None:
            box = np.array(bbox, dtype=float).reshape(-1)
            if box.size >= 4:
                x1, y1, x2, y2 = [int(v) for v in box[:4]]
                h, w = img.shape[:2]
                x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w))
                y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h))
                if x2 > x1 and y2 > y1:
                    crop = img[y1:y2, x1:x2]
                    blocks.extend([
                        {"type": "text", "text": "observed_object_crop"},
                        _image_block(crop),
                    ])
        return [block for block in blocks if block is not None]
    except Exception:
        return []


def _image_block(img: np.ndarray) -> dict[str, Any] | None:
    try:
        img = _ensure_min_image_size(np.asarray(img))
        ok, encoded = cv2.imencode(".jpg", img)
        if not ok:
            return None
        data = base64.b64encode(encoded.tobytes()).decode("utf-8")
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}
    except Exception:
        return None


def _ensure_min_image_size(img: np.ndarray, min_size: int = 32) -> np.ndarray:
    """Pad/resize tiny crops before sending them to VLM providers.

    Some OpenAI-compatible providers reject images whose width/height is below
    a provider-specific threshold.  Padding preserves the crop content and avoids
    making matcher failures look like semantic rejections.
    """

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.ndim != 3 or img.shape[0] <= 0 or img.shape[1] <= 0:
        return img
    h, w = img.shape[:2]
    if h >= min_size and w >= min_size:
        return img
    scale = max(float(min_size) / max(1, h), float(min_size) / max(1, w))
    new_w = max(min_size, int(round(w * scale)))
    new_h = max(min_size, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
