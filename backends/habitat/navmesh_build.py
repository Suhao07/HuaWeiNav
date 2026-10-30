"""Build scene-bound NavMeshes with explicit robot dimensions and provenance."""
import hashlib
import json
from pathlib import Path
from prior_map.observations import write_json


def build_navmesh(scene, output, *, radius, height, max_climb, max_slope):
    """Generate a mesh and receipt using explicit dimensions in metres/degrees.

    Radius erodes walkable space; height constrains overhead clearance. Recast
    voxelization affects these constraints, so all effective settings are saved.
    A verified build establishes configuration provenance, not physical clearance
    acceptance or equivalence to an actual robot model.
    """
    import habitat_sim
    scene, output = Path(scene), Path(output)
    if radius <= 0 or height <= 0 or max_climb < 0 or not 0 <= max_slope < 90:
        raise ValueError('Invalid robot dimensions or slope')
    output.mkdir(parents=True, exist_ok=False)
    settings = habitat_sim.NavMeshSettings()
    settings.set_defaults()
    settings.agent_radius, settings.agent_height = radius, height
    settings.agent_max_climb, settings.agent_max_slope = max_climb, max_slope
    config = habitat_sim.SimulatorConfiguration()
    config.scene_id = str(scene)
    config.enable_physics = False
    config.create_renderer = False
    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = []
    with habitat_sim.Simulator(habitat_sim.Configuration(config, [agent])) as sim:
        if not sim.recompute_navmesh(sim.pathfinder, settings, include_static_objects=False):
            raise RuntimeError('Habitat NavMesh construction failed')
        mesh = output / 'scene.navmesh'
        sim.pathfinder.save_nav_mesh(str(mesh))
        receipt = {'status': 'complete', 'scene_sha256': hashlib.sha256(scene.read_bytes()).hexdigest(),
            'navmesh_sha256': hashlib.sha256(mesh.read_bytes()).hexdigest(),
            'habitat_version': habitat_sim.__version__, 'include_static_objects': False,
            'settings': {name: getattr(settings, name) for name in dir(settings)
                         if not name.startswith('_') and not callable(getattr(settings, name))},
            'num_islands': sim.pathfinder.num_islands, 'navigable_area_m2': sim.pathfinder.navigable_area,
            'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope': 'verified_build_configuration_not_physical_robot_acceptance'}
    write_json(output / 'build_receipt.json', receipt)
    return receipt


def verify_build_receipt(receipt_path, navmesh_sha256, scene_sha256):
    """Bind explicit build settings to the current mesh and captured scene."""
    receipt = json.loads(Path(receipt_path).read_text())
    if receipt['status'] != 'complete':
        raise ValueError('NavMesh build is incomplete')
    if receipt['navmesh_sha256'] != navmesh_sha256:
        raise ValueError('NavMesh build receipt hash mismatch')
    if receipt['scene_sha256'] != scene_sha256:
        raise ValueError('NavMesh build scene differs from captured scene')
    return receipt['settings']
