"""Habitat bindings for LangMap; navigation policies consume observations unchanged."""
import numpy as np
import habitat_sim
from habitat.core.registry import registry
from habitat.core.embodied_task import Measure
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from .langmap import terminal_metrics


@registry.register_simulator(name='LangMapSim-v0')
class LangMapSimulator(HabitatSim):
    """Rebuild each loaded scene using the official Stretch navigation mesh preset."""

    def __init__(self, config):
        super().__init__(config)
        self._ensure_mesh()

    def reconfigure(self, habitat_config):
        """Rebuild on scene changes before measurements are reset."""
        super().reconfigure(habitat_config)
        self._ensure_mesh()

    def _ensure_mesh(self):
        if getattr(self, '_langmap_scene', None) == self._current_scene:
            return
        settings = habitat_sim.NavMeshSettings()
        settings.set_defaults()
        for key, value in dict(agent_radius=0.17, agent_height=1.41, agent_max_climb=0.10,
                               cell_height=0.05, agent_max_slope=45.0).items():
            setattr(settings, key, value)
        if not self.recompute_navmesh(self.pathfinder, settings, include_static_objects=True):
            raise RuntimeError('LangMap Stretch navmesh rebuild failed')
        self._langmap_scene = self._current_scene


@registry.register_measure
class LangMapEvaluation(Measure):
    """Track every displacement and evaluate terminal proximity independently of STOP."""

    cls_uuid = 'langmap_evaluation'

    def __init__(self, sim, config, **kwargs):
        self._sim = sim
        super().__init__()

    def _get_uuid(self, *args, **kwargs):
        return self.cls_uuid

    def reset_metric(self, episode, *args, **kwargs):
        """Freeze evaluator viewpoints and start distance for this task."""
        self._goals = [view.agent_state.position for goal in episode.goals for view in goal.view_points]
        self._last = np.array(self._sim.get_agent_state().position)
        self._length = 0.0
        self._initial = self._distance(self._last)
        self.update_metric(episode)

    def _distance(self, position):
        return self._sim.geodesic_distance(position, self._goals)

    def update_metric(self, episode, *args, **kwargs):
        """Accumulate L=sum(||x_t-x_(t-1)||_2), including turns and final actions."""
        position = np.array(self._sim.get_agent_state().position)
        self._length += float(np.linalg.norm(position - self._last))
        self._last = position
        self._metric = terminal_metrics(self._initial, self._distance(position), self._length)
