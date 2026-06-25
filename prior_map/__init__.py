"""Prior-map contracts and adapters for STRIVE."""

from .alignment import PriorMapAlignment, PriorMapAlignmentError
from .contracts import (
    FrontierPrior,
    ObjectPrior,
    PriorMapData,
    PriorObject,
    PriorObservationRecord,
    PriorRoom,
    PriorTopologyEdge,
    RoomPrior,
    SearchPriorResult,
    SupportRegionPrior,
)
from .loaders import PriorMapLoader, PriorMapLoaderError
from .memory import PriorMapMemory, PriorObjectRuntimeState, PriorRoomRuntimeState
from .policy_adapter import (
    PriorAnnotatedCandidate,
    PriorMapPolicyAdapter,
    annotate_target_candidates,
    rank_frontiers,
    rank_rooms,
)
from .prompt_context import (
    PriorMapPromptContextBuilder,
    PromptContextBundle,
    summarize_prior_map,
    summarize_search_prior,
    to_compact_xml,
)
from .query import PriorMapQueryContext, PriorMapQueryService, PriorMapQueryWeights
from .real_robot import (
    PriorMapRealRobotConfig,
    PriorMapRealRobotRuntime,
    build_prior_map_real_robot_runtime,
    detect_live_prior_conflicts,
)
from .simulation import (
    PriorMapSimulationConfig,
    PriorMapSimulationRuntime,
    build_prior_map_simulation_runtime,
    configure_mapper_prior_map,
    refresh_mapper_prior_map_query,
)
from .visualizer import PriorMapSomVisualizer, SomMarker, SomView, render_global_view, render_room_view

__all__ = [
    "FrontierPrior",
    "ObjectPrior",
    "PriorMapAlignment",
    "PriorMapAlignmentError",
    "PriorMapLoader",
    "PriorMapLoaderError",
    "PriorMapMemory",
    "PriorAnnotatedCandidate",
    "PriorMapPolicyAdapter",
    "PriorMapPromptContextBuilder",
    "PriorMapQueryContext",
    "PriorMapQueryService",
    "PriorMapQueryWeights",
    "PriorMapRealRobotConfig",
    "PriorMapRealRobotRuntime",
    "PriorMapSimulationConfig",
    "PriorMapSimulationRuntime",
    "PriorMapSomVisualizer",
    "PriorMapData",
    "PriorObject",
    "PriorObjectRuntimeState",
    "PriorObservationRecord",
    "PriorRoom",
    "PriorRoomRuntimeState",
    "PriorTopologyEdge",
    "PromptContextBundle",
    "RoomPrior",
    "SearchPriorResult",
    "SomMarker",
    "SomView",
    "SupportRegionPrior",
    "annotate_target_candidates",
    "build_prior_map_real_robot_runtime",
    "build_prior_map_simulation_runtime",
    "configure_mapper_prior_map",
    "detect_live_prior_conflicts",
    "rank_frontiers",
    "rank_rooms",
    "refresh_mapper_prior_map_query",
    "render_global_view",
    "render_room_view",
    "summarize_prior_map",
    "summarize_search_prior",
    "to_compact_xml",
]
