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


def test_candidate_observation_defaults_are_independent_and_legacy_records_stay_unbound():
    from verification.contracts import VerificationRecord

    first = VerificationResult(False, "uncertain")
    second = VerificationResult(False, "uncertain")
    first.candidate_observations.append({"role": "current"})
    assert second.candidate_observations == []
    legacy = VerificationRecord("instruction", "cup", "accepted", evidence_paths=["old.png"])
    other = VerificationRecord("instruction", "other", "accepted")
    assert legacy.candidate_observations == []
    legacy.candidate_observations.append({"role": "earlier"})
    assert other.candidate_observations == []
    stored = EvidenceStore().put("cup", "cup", second, evidence_paths=["old.png"])
    assert stored.evidence_paths == ["old.png"] and stored.candidate_observations == []


def test_store_deepcopies_candidate_observations_without_inventing_bindings(tmp_path):
    from copy import deepcopy
    from verification.contracts import VerificationRecord
    from verification.images import image_input

    path = tmp_path / "current.png"
    path.write_bytes(b"offline candidate frame")
    _, archive, digest = image_input(str(path), tmp_path / "blobs")
    own = [{"archive_path": archive, "candidate_uid": "cup-1", "task_revision": 1,
            "observation_id": "frame:7", "bbox": [1, 2, 3, 4], "role": "current"},
           {"archive_path": archive, "candidate_uid": "cup-1", "task_revision": 1,
            "observation_id": "frame:6", "bbox": [5, 6, 7, 8], "role": "earlier"}]
    original = deepcopy(own)
    result = VerificationResult(True, "accept", semantic_satisfied=True, task_revision=1,
                                evidence_ids=[archive, "other-role.png"],
                                dependencies={"image:" + archive: digest}, candidate_observations=own)
    store = EvidenceStore(1)
    record = store.put("cup", "cup-1", result)
    assert record.candidate_observations == original
    assert record.evidence_paths == [archive, "other-role.png"]
    assert record.dependencies == result.dependencies
    own[0]["bbox"][0] = 999
    own[1]["candidate_uid"] = "changed"
    own.append({"role": "history"})
    assert record.candidate_observations == original
    exported = record.as_dict()
    assert VerificationRecord(**exported).candidate_observations == original
    exported["candidate_observations"][0]["bbox"][0] = 123
    assert record.candidate_observations == original
    assert store.as_dict()[next(iter(store.as_dict()))]["candidate_observations"] == original
