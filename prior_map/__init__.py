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
from .evaluation import (
    prior_map_failure_modes,
    prior_map_metrics_fields,
    prior_map_metrics_summary,
    write_prior_map_static_artifacts,
    write_prior_map_step_artifacts,
)
from .habitat_objectnav import (
    HabitatObjectNavPriorMapBuilder,
    HabitatPriorMapBuildResult,
    build_habitat_objectnav_prior_map,
    write_prior_map_with_alignment,
)
from .hm3d_semantic import (
    HM3DSemanticInstance,
    HM3DSemanticPriorMapBuildResult,
    HM3DSemanticTxtPriorMapBuilder,
    build_hm3d_semantic_prior_map,
    write_hm3d_semantic_prior_map_with_alignment,
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
from .visualizer import (
    PriorMapSomVisualizer,
    SomMarker,
    SomView,
    render_global_view,
    render_room_view,
    write_som_artifacts,
)

__all__ = [
    "FrontierPrior",
    "HabitatObjectNavPriorMapBuilder",
    "HabitatPriorMapBuildResult",
    "HM3DSemanticInstance",
    "HM3DSemanticPriorMapBuildResult",
    "HM3DSemanticTxtPriorMapBuilder",
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
    "build_habitat_objectnav_prior_map",
    "build_hm3d_semantic_prior_map",
    "configure_mapper_prior_map",
    "detect_live_prior_conflicts",
    "prior_map_failure_modes",
    "prior_map_metrics_fields",
    "prior_map_metrics_summary",
    "rank_frontiers",
    "rank_rooms",
    "refresh_mapper_prior_map_query",
    "render_global_view",
    "render_room_view",
    "summarize_prior_map",
    "summarize_search_prior",
    "to_compact_xml",
    "write_prior_map_static_artifacts",
    "write_prior_map_step_artifacts",
    "write_prior_map_with_alignment",
    "write_hm3d_semantic_prior_map_with_alignment",
    "write_som_artifacts",
]
