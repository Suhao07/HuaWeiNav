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
from .query import PriorMapQueryContext, PriorMapQueryService, PriorMapQueryWeights

__all__ = [
    "FrontierPrior",
    "ObjectPrior",
    "PriorMapAlignment",
    "PriorMapAlignmentError",
    "PriorMapLoader",
    "PriorMapLoaderError",
    "PriorMapMemory",
    "PriorMapQueryContext",
    "PriorMapQueryService",
    "PriorMapQueryWeights",
    "PriorMapData",
    "PriorObject",
    "PriorObjectRuntimeState",
    "PriorObservationRecord",
    "PriorRoom",
    "PriorRoomRuntimeState",
    "PriorTopologyEdge",
    "RoomPrior",
    "SearchPriorResult",
    "SupportRegionPrior",
]
