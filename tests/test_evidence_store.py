"""Evidence scope follows task revision, predicate and object identity."""
from verification.contracts import VerificationResult
from verification.store import EvidenceStore


def test_old_revision_is_retained_without_rejecting_current_candidate():
    store = EvidenceStore(task_revision=2)
    result = VerificationResult(False, "reject_candidate", task_revision=1, target_uid="cup-1")
    store.put("find a cup", "cup-1", result)
    assert len(store.history) == 1
    assert store.history[0].verdict == "false"
    assert store.get("find a cup", "cup-1") is None
    assert not store.is_hard_rejected("find a cup", "cup-1")


def test_relation_rejection_stays_with_the_verified_pair():
    store = EvidenceStore(task_revision=1)
    result = VerificationResult(False, "reject_relation", task_revision=1,
                                target_uid="cup-1", predicate_id="on", anchor_uid="table-1",
                                evidence_ids=["frame-1"], dependencies={"map_frame": "v1"})
    record = store.put("cup on table", "cup-1", result)
    assert store.get("cup on table", "cup-1", predicate_id="on", anchor_uid="table-1") is record
    assert store.get("cup on table", "cup-1", predicate_id="on", anchor_uid="table-2") is None
    assert not store.is_hard_rejected("cup on table", "cup-1")
    result.dependencies["map_frame"] = "v2"
    assert record.dependencies == {"map_frame": "v1"}
    assert record.evidence_paths == ["frame-1"]


def test_unknown_has_distinct_semantics_from_counterevidence():
    result = VerificationResult(False, "uncertain", task_revision=1, target_uid="cup-1")
    assert result.verdict == "unknown"
    assert not result.accepted
