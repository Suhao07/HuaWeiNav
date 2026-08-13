from types import SimpleNamespace

from instruction_adapter.contracts import ExecutionPolicy, InstructionPlan, SearchPriors, TargetQuery
from prior_map.alignment import PriorMapAlignment
from prior_map.contracts import PriorMapData, PriorObject, PriorRoom
from prior_map.memory import PriorMapMemory
from prior_map.query import PriorMapQueryService


def _memory() -> PriorMapMemory:
    base_map = PriorMapData(
        scene_id="query_scene",
        rooms=(
            PriorRoom(uid="room_kitchen", label="kitchen", confidence=0.6),
            PriorRoom(uid="room_living", label="living room", confidence=0.6),
        ),
        objects=(
            PriorObject(
                uid="prior_mug",
                label="mug",
                parent_room_uid="room_kitchen",
                exact=False,
                confidence=0.4,
                aliases=("cup",),
            ),
            PriorObject(
                uid="prior_table",
                label="table",
                parent_room_uid="room_kitchen",
                exact=False,
                confidence=0.5,
            ),
            PriorObject(
                uid="prior_book",
                label="book",
                parent_room_uid="room_living",
                exact=True,
                confidence=0.7,
            ),
        ),
    )
    return PriorMapMemory(base_map=base_map, alignment=PriorMapAlignment.identity())


def _geometry_memory() -> PriorMapMemory:
    base_map = PriorMapData(
        scene_id="geometry_scene",
        frame_id="habitat_world",
        rooms=(
            PriorRoom(uid="room_kitchen", label="kitchen", centroid_xy=(1.0, 1.0), confidence=0.6),
            PriorRoom(uid="room_living", label="living room", centroid_xy=(10.0, 10.0), confidence=0.6),
        ),
        objects=(
            PriorObject(
                uid="prior_mug",
                label="mug",
                position_xyz=(10.0, 0.0, 10.0),
                parent_room_uid="room_living",
                exact=True,
                confidence=0.9,
                aliases=("cup",),
            ),
        ),
    )
    return PriorMapMemory(
        base_map=base_map,
        alignment=PriorMapAlignment.identity(prior_frame_id="habitat_world", runtime_frame_id="habitat_world"),
    )


def _plan() -> InstructionPlan:
    return InstructionPlan(
        raw_instruction="find the mug on the table in the kitchen",
        dataset_target="mug",
        targets=[
            TargetQuery(
                id="target_mug",
                name="mug",
                detector_terms=["mug"],
                aliases=["cup"],
                terminal=True,
            )
        ],
        search_priors=SearchPriors(
            room_hints=["kitchen"],
            support_objects=["table"],
            affordances=["countertop"],
        ),
        execution=ExecutionPolicy(mode="any_target_success"),
        valid=True,
    )


def _plan_without_room_hint() -> InstructionPlan:
    plan = _plan()
    plan.search_priors = SearchPriors(room_hints=[], support_objects=[], affordances=[])
    return plan


def test_query_scores_rooms_objects_support_and_frontiers() -> None:
    memory = _memory()
    runtime_context = SimpleNamespace(
        objects=(SimpleNamespace(uid="runtime_mug_1", label="cup", confidence=0.9, room_id="room_kitchen"),),
        rooms=(SimpleNamespace(uid="room_kitchen", label="kitchen", confidence=0.8),),
        frontiers=(SimpleNamespace(uid="frontier_1", room_uid="room_kitchen"),),
    )
    memory.update_from_snapshot(SimpleNamespace(timestamp=1.0, source="test", robot_pose=(0, 0, 0), objects=runtime_context.objects, rooms=runtime_context.rooms))

    result = PriorMapQueryService().query(_plan(), runtime_context, memory)

    assert result.room_rankings[0].room_uid == "room_kitchen"
    assert result.room_rankings[0].metadata["score_components"]["room_relevance"] == 1.0
    assert result.object_rankings[0].object_uid == "prior_mug"
    assert result.object_rankings[0].matched_runtime_uid == "runtime_mug_1"
    assert result.object_rankings[0].metadata["score_components"]["live_match"] > 0.0
    assert result.support_regions[0].uid == "prior_table"
    assert result.frontier_biases == ()
    assert result.diagnostics["geometry_enabled"] is False
    assert result.diagnostics["geometry_disabled_reason"] == "no_positioned_target_object_prior"
    assert result.diagnostics["authority"] == "ranking_only"


def test_geometry_prior_generates_frontier_biases_and_lifts_parent_room() -> None:
    memory = _geometry_memory()
    runtime_context = SimpleNamespace(
        objects=(),
        rooms=(),
        frontiers=(
            SimpleNamespace(uid="frontier_far", habitat_world_position=(0.0, 0.0, 0.0)),
            SimpleNamespace(uid="frontier_close", habitat_world_position=(9.8, 0.0, 9.7)),
        ),
    )

    result = PriorMapQueryService().query(_plan_without_room_hint(), runtime_context, memory)

    assert result.room_rankings[0].room_uid == "room_living"
    assert result.room_rankings[0].metadata["score_components"]["target_object_room_relevance"] > 0.0
    assert result.frontier_biases[0].frontier_uid == "frontier_close"
    assert result.frontier_biases[0].metadata["source"] == "geometry_prior"
    components = result.frontier_biases[0].metadata["score_components"]
    for key in (
        "target_object_relevance",
        "target_room_relevance",
        "distance_score",
        "alignment_confidence",
        "visited_exhausted_penalty",
    ):
        assert key in components
    assert result.diagnostics["geometry_enabled"] is True
    assert result.diagnostics["geometry_disabled_reason"] is None


def test_query_penalizes_visited_exhausted_and_rejected_priors() -> None:
    memory = _memory()
    memory.mark_room_visited("room_kitchen", step=1)
    memory.room_states["room_living"].metadata["exhausted"] = True
    memory.mark_prior_rejected("prior_mug", "wrong instance", step=2)

    result = PriorMapQueryService().query(_plan(), SimpleNamespace(objects=(), rooms=(), frontiers=()), memory)
    room_by_uid = {room.room_uid: room for room in result.room_rankings}
    obj_by_uid = {obj.object_uid: obj for obj in result.object_rankings}

    assert room_by_uid["room_kitchen"].metadata["score_components"]["visited_adjustment"] < 0.0
    assert room_by_uid["room_living"].metadata["score_components"]["exhausted_penalty"] < 0.0
    assert obj_by_uid["prior_mug"].metadata["score_components"]["rejection_penalty"] < 0.0


def test_query_result_is_ranking_only_and_has_no_motion_goal() -> None:
    result = PriorMapQueryService().query(
        _plan(),
        SimpleNamespace(objects=(), rooms=(), frontiers=(SimpleNamespace(uid="frontier_1"),)),
        _geometry_memory(),
    )

    assert not hasattr(result, "motion_goal")
    assert not hasattr(result, "navigation_intent")
    assert not hasattr(result, "goal_pose")
    assert result.diagnostics["geometry_disabled_reason"] in {
        "no_runtime_frontier_world_position",
        "frontier_bias_empty",
    }
