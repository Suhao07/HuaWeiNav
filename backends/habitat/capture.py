"""Task-independent coverage scans using Habitat geometry, not semantic GT."""
from pathlib import Path
from dataclasses import dataclass, asdict
import math
import time
import hashlib
import subprocess
import numpy as np
from prior_map.observations import COORDINATES, SCHEMA_VERSION, write_frame, write_json

# Habitat world (x right, y up, z back) -> right-handed map (x, -z, y).
WORLD_TO_MAP = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
OPTICAL_TO_GL = np.diag([1., -1, -1])


@dataclass(frozen=True)
class CaptureConfig:
    """Bounded scan settings. Candidate coverage is measured on a separate sample."""
    seed: int = 17
    width: int = 320
    height: int = 240
    hfov: float = 90.
    sensor_height: float = 1.25
    samples: int = 256
    max_positions: int = 8
    spacing: float = 1.0
    headings: int = 8
    min_island_area: float = 1.0
    clearance: float = 0.25
    island: int = -1

    def __post_init__(self):
        if min(self.width, self.height, self.samples, self.max_positions, self.headings) <= 0:
            raise ValueError("Counts and image dimensions must be positive")
        if not 0 < self.hfov < 180 or min(self.spacing, self.sensor_height) <= 0:
            raise ValueError("Invalid field of view, spacing or sensor height")


def select_positions(points, spacing, limit, distance=None, coverage_points=None):
    """Greedily cover navigation probes using admissible camera positions.

    Select argmax_v |{q in Q_uncovered: d_nav(q,v) <= spacing}| until no gain
    or the position budget is exhausted. Probes include navigable locations
    that fail camera clearance; the independent audit set stays unused here.
    Euclidean distance is a lower bound used to skip impossible path matches.
    """
    if len(points) == 0:
        return []
    probes = points if coverage_points is None else np.asarray(coverage_points)
    cover = np.linalg.norm(np.asarray(points)[:, None] - probes[None], axis=2) <= spacing
    if distance is not None:
        for i, j in zip(*np.where(cover)):
            cover[i, j] = distance(points[i], probes[j]) <= spacing
    uncovered = np.ones(len(probes), dtype=bool)
    chosen = []
    while len(chosen) < limit:
        gains = cover[:, uncovered].sum(axis=1)
        index = int(gains.argmax())
        if gains[index] == 0:
            break
        chosen.append(index)
        uncovered &= ~cover[index]
    return [points[i] for i in chosen]


def order_positions(positions, distance):
    """Order one connected capture segment by nearest remaining path distance.

    This is a rendering order, not a record of executed robot motion.
    """
    remaining = list(positions[1:])
    ordered = list(positions[:1])
    while remaining:
        index = min(range(len(remaining)), key=lambda i: distance(ordered[-1], remaining[i]))
        ordered.append(remaining.pop(index))
    return ordered


def collect(scene_dir, output, config):
    """Render finite coverage_scan observations and report per-island coverage.

    Uses NavMesh only to select positions and measure geodesic coverage.
    No task dataset, semantic sensor, object goals, detector or LLM is loaded.
    """
    import habitat_sim
    import quaternion
    scene_dir, output = Path(scene_dir), Path(output)
    mesh, = scene_dir.glob("*.basis.glb")
    navmesh, = scene_dir.glob("*.basis.navmesh")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": SCHEMA_VERSION, "coordinates": COORDINATES,
                "scene_id": mesh.name.removesuffix(".basis.glb"), "scene_path": str(mesh),
                "navmesh_path": str(navmesh), "habitat_version": habitat_sim.__version__,
                "mode": "coverage_scan", "pose_source": "simulator_gt",
                "geometry_assisted": True, "semantic_gt_used": False,
                "config": asdict(config), "status": "running"}
    write_json(output / "manifest.json", manifest)
    manifest["scene_sha256"] = hashlib.sha256(mesh.read_bytes()).hexdigest()
    manifest["navmesh_sha256"] = hashlib.sha256(navmesh.read_bytes()).hexdigest()
    manifest["collector_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    repo = Path(__file__).resolve().parents[2]
    manifest["git_head"] = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    manifest["git_dirty"] = bool(subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip())
    started = time.monotonic()
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(mesh)
    sim_cfg.enable_physics = False
    sensors = []
    for name, kind in [("rgb", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH)]:
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid, spec.sensor_type = name, kind
        spec.resolution = [config.height, config.width]
        spec.position = [0., config.sensor_height, 0.]
        spec.hfov = config.hfov
        sensors.append(spec)
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensors
    sim = None
    try:
        sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
        pf = sim.pathfinder
        if not pf.load_nav_mesh(str(navmesh)):
            raise ValueError(f"Cannot load NavMesh: {navmesh}")
        if config.island >= pf.num_islands:
            raise ValueError(f"Island {config.island} absent; count={pf.num_islands}")
        def path_distance(start, end):
            path = habitat_sim.ShortestPath()
            path.requested_start, path.requested_end = start, end
            return float(path.geodesic_distance) if pf.find_path(path) else float("inf")

        pf.seed(config.seed)
        regions, plan, coverage = [], [], []
        for island in range(pf.num_islands):
            area = float(pf.island_area(island))
            included = area >= config.min_island_area and config.island in (-1, island)
            region = {"island_id": island, "area_m2": area, "included": included,
                      "reason": "included" if included else "area_or_explicit_selection"}
            regions.append(region)
            if not included:
                continue
            samples = np.array([pf.get_random_navigable_point(island_index=island)
                                for _ in range(config.samples)])
            if not np.isfinite(samples).all():
                raise ValueError(f"Non-finite NavMesh samples in island {island}")
            region["height_range_habitat_y"] = [float(samples[:, 1].min()), float(samples[:, 1].max())]
            candidates = np.array([p for p in samples if pf.distance_to_closest_obstacle(p) >= config.clearance])
            positions = select_positions(candidates, config.spacing, config.max_positions, path_distance,
                                         coverage_points=samples)
            positions = order_positions(positions, path_distance)
            if not positions:
                region["reason"] = "no_candidate_with_clearance"
                coverage.append({"island_id": island, "position_count": 0, "fraction": 0.})
                continue
            # 独立检查点作为分母；测量 NavMesh 路径距离，不把欧氏接近当作跨墙覆盖。
            checks = [pf.get_random_navigable_point(island_index=island) for _ in range(config.samples)]
            nearest = [min(path_distance(check, position) for position in positions) for check in checks]
            uncovered = [i for i, value in enumerate(nearest) if value > config.spacing]
            covered = len(checks) - len(uncovered)
            coverage.append({"island_id": island, "check_count": len(checks),
                             "covered_count": covered, "fraction": covered / len(checks),
                             "radius_m": config.spacing, "position_count": len(positions),
                             "candidate_count": len(candidates),
                             "check_points_habitat": [np.asarray(p).tolist() for p in checks],
                             "nearest_path_distance_m": [d if math.isfinite(d) else None for d in nearest],
                             "uncovered_check_indices": uncovered,
                             "unreachable_check_count": sum(not math.isfinite(d) for d in nearest)})
            plan.extend((island, p) for p in positions)
        write_json(output / "regions.json", regions)
        write_json(output / "coverage.json", {"measurement": "sampled_geodesic_position_coverage",
                                               "regions": coverage, "selection": "geodesic_probe_coverage_then_nearest_order",
                                               "surface_coverage": "not_measured"})
        if not plan:
            raise ValueError("No valid scan positions in selected regions")
        f = config.width / (2 * math.tan(math.radians(config.hfov) / 2))
        k = np.array([[f, 0, (config.width - 1) / 2], [0, f, (config.height - 1) / 2], [0, 0, 1.]])
        agent = sim.get_agent(0)
        frames, poses, groups = 0, [], []
        for position_id, (island, position) in enumerate(plan):
            group = []
            for heading in range(config.headings):
                angle = 2 * math.pi * heading / config.headings
                state = habitat_sim.AgentState()
                state.position = position
                state.rotation = quaternion.from_rotation_vector([0., angle, 0.])
                agent.set_state(state)
                obs = sim.get_sensor_observations()
                camera = agent.get_state().sensor_states["rgb"]
                pose = np.eye(4)
                pose[:3, :3] = WORLD_TO_MAP @ quaternion.as_rotation_matrix(camera.rotation) @ OPTICAL_TO_GL
                pose[:3, 3] = WORLD_TO_MAP @ camera.position
                valid = float((obs["depth"] > 0).mean())
                write_frame(output, frames, obs["rgb"], obs["depth"], k, pose,
                            position_id=position_id, island_id=island, heading_index=heading,
                            valid_depth_fraction=valid,
                            depth_percentiles_m=np.quantile(obs["depth"][obs["depth"] > 0], [.05, .5, .95]).tolist()
                                if valid else None,
                            near_depth_fraction=float(((obs["depth"] > 0) & (obs["depth"] < .25)).mean()),
                            sim_step=None)
                poses.append({"frame_id": frames, "agent_position_habitat": np.asarray(position).tolist(),
                              "camera_to_map": pose.tolist()})
                group.append(frames)
                frames += 1
            groups.append({"group_id": position_id, "kind": "panorama", "frame_ids": group, "complete": True})
        for name, records in [("poses", poses), ("observation_groups", groups)]:
            import json
            (output / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
        manifest.update(status="complete", frame_count=frames, position_count=len(plan),
                        elapsed_seconds=time.monotonic() - started,
                        termination="bounded_scan_finished", coverage_requirement_met=None)
        write_json(output / "manifest.json", manifest)
        return manifest
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "manifest.json", manifest)
        raise
    finally:
        if sim is not None:
            sim.close()
