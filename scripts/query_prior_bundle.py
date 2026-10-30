#!/usr/bin/env python3
"""Reload a portable bundle and query existing label/alias priors with evidence."""
import argparse
import hashlib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from instruction_adapter.contracts import InstructionPlan, TargetQuery
from prior_map.alignment import PriorMapAlignment
from prior_map.bundle import load_bundle
from prior_map.memory import PriorMapMemory
from prior_map.observations import write_json
from prior_map.query import PriorMapQueryService


def main():
    """Query concepts or explicitly compile language; neither grants motion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle")
    parser.add_argument("concept", help="Concept term, not an unrestricted instruction parser")
    parser.add_argument("--output", required=True)
    parser.add_argument("--compile-instruction", action="store_true",
                        help="Use the existing strict language compiler before map query")
    parser.add_argument("--vlm", default="ark")
    args = parser.parse_args()
    bundle = load_bundle(args.bundle)
    prior = bundle["prior_map"]
    memory = PriorMapMemory(base_map=prior, alignment=PriorMapAlignment.identity(
        prior_frame_id=prior.frame_id, runtime_frame_id=prior.frame_id))
    task = None
    if args.compile_instruction:
        from instruction.compiler import compile_instruction
        from instruction_adapter.compiler import RUNTIME_CONSTRAINT_TYPES
        from instruction_adapter.task_compat import runtime_plan
        compiled = compile_instruction(args.concept, RUNTIME_CONSTRAINT_TYPES, vlm=args.vlm)
        if compiled.task is None:
            raise ValueError(f"{compiled.status}: {compiled.diagnostics}")
        task = compiled.task.as_dict()
        plan = runtime_plan(compiled.task)
    else:
        plan = InstructionPlan(raw_instruction=args.concept,
            targets=[TargetQuery(id="query", name=args.concept, detector_terms=[args.concept])], valid=True)
    result = PriorMapQueryService().query(plan, {}, memory)
    matches = []
    for candidate in result.object_rankings:
        if candidate.metadata["score_components"]["concept_relevance"] <= 0:
            continue
        evidence = [obs for obs in bundle["observations"] if candidate.object_uid in obs["object_uids"]]
        matches.append({"ranking": candidate.to_dict(), "evidence": evidence})
    write_json(args.output, {"map_id": bundle["manifest"]["map_id"],
        "map_revision": bundle["manifest"]["map_revision"],
        "prior_map_sha256": hashlib.sha256((Path(args.bundle) / 'prior_map.json').read_bytes()).hexdigest(),
        "concept": args.concept,
        "diagnostics": result.diagnostics, "matches": matches,
        "task_spec": task, "runtime_plan": plan.as_dict(),
        "scope": ("compiled_instruction_prior_retrieval_not_task_completion" if task is not None
                  else "offline_label_alias_retrieval_with_evidence")})
    print(f"{len(matches)} matching object hypotheses; receipt: {args.output}")


if __name__ == "__main__":
    main()
