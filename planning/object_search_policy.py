from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from instruction_adapter.verifier import candidate_from_object
from planning.mode_policy import is_anchor_first_relation_search


@dataclass
class ObjectSearchResult:
    found: bool
    obj: Any | None = None
    answer: str = ""
    skipped_objs: list[dict[str, Any]] = field(default_factory=list)
    concept_debug: list[dict[str, Any]] = field(default_factory=list)
    anchor_record: dict[str, Any] = field(default_factory=dict)


class InstructionObjectSearchPolicy:
    """Select terminal targets or anchor references for instruction mode.

    该策略只负责“本轮应该追哪个对象实例”。terminal target
    可以触发后续 stop 验证 anchor reference 只是局部搜索参考物，绝不能
    直接作为任务成功。最终是否满足原始指令仍由 final verifier 判断。
    """

    def select(self, mapper: Any, *, plan: Any, step: int | None = None) -> ObjectSearchResult:
        raw_instruction = mapper._raw_instruction_for_verifier()
        terminal_match_records = self._match_terminal_concepts(mapper, plan, raw_instruction, step)
        target_objs: list[Any] = []
        skipped_objs: list[dict[str, Any]] = []
        concept_debug: list[dict[str, Any]] = []

        for obj in mapper.objects:
            matched_target = None
            for target in plan.terminal_targets:
                obj_uid = candidate_from_object(obj, canonical_label=target.name, step=step).uid
                record = terminal_match_records.get((target.id, obj_uid))
                if record is None:
                    continue
                if record.matches_concept and record.terminal_eligible:
                    matched_target = target
                    concept_debug.append(record.as_dict())
                    break

            if matched_target is None:
                continue

            if mapper._is_verifier_rejected(obj, step=step):
                skipped_objs.append(candidate_from_object(obj, canonical_label=mapper.target, step=step).as_dict())
                continue

            if (
                is_anchor_first_relation_search(plan)
                and not mapper._has_unblocked_anchor_evidence(plan, matched_target, obj, step=step, debug=concept_debug)
            ):
                skipped_objs.append(
                    {
                        **candidate_from_object(obj, canonical_label=matched_target.name, step=step).as_dict(),
                        "skip_reason": "missing_unblocked_relation_anchor_evidence",
                    }
                )
                continue

            obj.tag = matched_target.name
            obj.conf_list[matched_target.name] = obj.confidence
            target_objs.append(obj)

        if target_objs:
            target_objs = sorted(target_objs, key=lambda x: x.confidence.numpy().item(), reverse=True)
            target_obj = target_objs[0]
            if target_obj.tag != mapper.target:
                target_obj.tag = mapper.target
                target_obj.conf_list[mapper.target] = target_obj.confidence
            return ObjectSearchResult(
                found=True,
                obj=target_obj,
                answer=(
                    f"Choose Obj '{target_obj.tag}' at position {target_obj.position} "
                    f"with confidence {target_obj.confidence.numpy().item()}."
                ),
                skipped_objs=skipped_objs,
                concept_debug=concept_debug,
            )

        anchor_result = self._select_anchor_reference(mapper, plan, raw_instruction, step, concept_debug)
        if anchor_result is not None:
            return anchor_result

        return ObjectSearchResult(
            found=False,
            answer=f"No target object '{mapper.target}' found; accepted aliases: {sorted(mapper._target_match_terms())}",
            skipped_objs=skipped_objs,
            concept_debug=concept_debug,
        )

    def _match_terminal_concepts(self, mapper: Any, plan: Any, raw_instruction: str, step: int | None) -> dict[tuple[str, str], Any]:
        records_by_target: dict[tuple[str, str], Any] = {}
        # terminal concept 必须批量 grounding，避免对象数线性放大 LVLM 调用。
        for target in plan.terminal_targets:
            records = mapper.concept_matcher.match_many(
                raw_instruction=raw_instruction,
                concept=target.concept_query(),
                objects=list(mapper.objects),
                step=step,
            )
            for record in records:
                records_by_target[(target.id, record.object_uid)] = record
        return records_by_target

    def _select_anchor_reference(
        self,
        mapper: Any,
        plan: Any,
        raw_instruction: str,
        step: int | None,
        concept_debug: list[dict[str, Any]],
    ) -> ObjectSearchResult | None:
        if not is_anchor_first_relation_search(plan):
            return None

        anchor_candidates = []
        for concept in mapper._anchor_concepts_for_plan():
            records = mapper.concept_matcher.match_many(
                raw_instruction=raw_instruction,
                concept=concept,
                objects=list(mapper.objects),
                step=step,
            )
            by_uid = {record.object_uid: record for record in records}
            for obj in mapper.objects:
                candidate = candidate_from_object(obj, canonical_label=concept.name, step=step)
                record = by_uid.get(candidate.uid)
                if record is None:
                    continue
                concept_debug.append(record.as_dict())
                if not record.matches_concept:
                    continue
                if mapper.anchor_search_ledger.is_blocked(raw_instruction, concept.id, candidate.uid):
                    continue
                anchor_candidates.append((record.confidence, concept, obj, candidate, record))

        if not anchor_candidates:
            return None

        anchor_candidates = sorted(anchor_candidates, key=lambda item: item[0], reverse=True)
        _, concept, anchor_obj, candidate, anchor_record = anchor_candidates[0]
        anchor_obj._instruction_reference_role = "anchor"
        anchor_obj._instruction_anchor_concept_id = concept.id
        anchor_obj._instruction_anchor_concept_name = concept.name
        anchor_obj._instruction_anchor_candidate_uid = candidate.uid
        mapper.anchor_search_ledger.mark(
            raw_instruction=raw_instruction,
            concept_id=concept.id,
            anchor_uid=candidate.uid,
            status="navigating_to_anchor",
            step=step,
            reason=anchor_record.reason,
        )
        return ObjectSearchResult(
            found=True,
            obj=anchor_obj,
            answer=(
                f"No terminal target '{mapper.target}' found; navigate to anchor reference "
                f"'{anchor_obj.tag}' for local search. Anchor is not a goal."
            ),
            concept_debug=concept_debug,
            anchor_record=anchor_record.as_dict(),
        )
