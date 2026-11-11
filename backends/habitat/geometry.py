"""Native Habitat queries in the mapper's world-coordinate ordering."""
import numpy as np


class HabitatMapperQueries:
    domain = "habitat_navmesh"

    def __init__(self, mapper):
        self.mapper = mapper

    def navigable_mask(self, positions):
        return np.array([self.mapper.env.sim.is_navigable([p[0], p[2], p[1]])
                         for p in positions], dtype=bool)

    def path_exists(self, start, end):
        import habitat_sim

        path = habitat_sim.ShortestPath()
        path.requested_start = [start[0], start[2], start[1]]
        path.requested_end = [end[0], end[2], end[1]]
        return bool(self.mapper.env.sim.pathfinder.find_path(path))

    def node_distances(self, start, ends):
        height = self.mapper.initial_position[2] - 0.88
        current = [start[0], height, start[1]]
        return [float(self.mapper.env.sim.geodesic_distance(current, [end[0], height, end[1]]))
                for end in ends]

    def distances_from_agent(self, ends):
        sim = self.mapper.env.sim
        current = sim.get_agent_state().position
        return [float(sim.geodesic_distance(current, [end[0], current[1], end[1]])) for end in ends]

    def distance_from_agent(self, end):
        return self.distances_from_agent([end])[0]
