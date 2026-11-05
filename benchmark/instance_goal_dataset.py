"""Per-episode goal identity for language-conditioned Habitat benchmarks."""
import json
import os
from habitat.core.registry import registry
from habitat.core.simulator import AgentState
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat.tasks.nav.object_nav_task import ObjectGoal, ObjectGoalNavEpisode, ObjectViewLocation


@registry.register_dataset(name='InstanceGoalNav-v1')
class InstanceGoalDataset(PointNavDatasetV1):
    """Retain per-task targets; category-wide deduplication would lose instance identity."""

    def from_json(self, json_str, scenes_dir=None):
        """Decode standard ObjectNav records without merging same-category goals."""
        data = json.loads(json_str)
        self.category_to_task_category_id = data['category_to_task_category_id']
        self.category_to_scene_annotation_category_id = data['category_to_scene_annotation_category_id']
        for row in data['episodes']:
            ep = ObjectGoalNavEpisode(**row)
            if scenes_dir is not None:
                ep.scene_id = os.path.join(scenes_dir, ep.scene_id)
            ep.goals = [ObjectGoal(**goal) for goal in ep.goals]
            for goal in ep.goals:
                goal.view_points = [ObjectViewLocation(agent_state=AgentState(**view['agent_state']),
                                                       iou=view['iou']) for view in goal.view_points]
            self.episodes.append(ep)
