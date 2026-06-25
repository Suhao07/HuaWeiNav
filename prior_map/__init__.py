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
    "rank_frontiers",
    "rank_rooms",
    "render_global_view",
    "render_room_view",
    "summarize_prior_map",
    "summarize_search_prior",
    "to_compact_xml",
]
