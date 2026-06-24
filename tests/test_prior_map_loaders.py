import json
from pathlib import Path

import pytest

from prior_map.contracts import PriorMapData, PriorObject, PriorRoom, PriorTopologyEdge
from prior_map.loaders import PriorMapLoader, PriorMapLoaderError


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loader_stays_platform_neutral() -> None:
    source = Path("prior_map/loaders.py").read_text(encoding="utf-8")

    forbidden = (
        "import rclpy",
        "import rospy",
        "import habitat",
        "import cv2",
        "import numpy",
        "from rclpy",
        "from rospy",
        "from habitat",
        "from cv2",
        "from numpy",
        "openai",
        "anthropic",
    )
    for text in forbidden:
        assert text not in source


def test_load_canonical_json_roundtrip(tmp_path) -> None:
    original = PriorMapData(
        scene_id="scene_canonical",
        rooms=(PriorRoom(uid="room_1", label="office", centroid_xy=(1.0, 2.0)),),
        objects=(
            PriorObject(
                uid="obj_1",
                label="desk",
                position_xyz=(1.5, 2.0, 0.0),
                parent_room_uid="room_1",
                exact=True,
            ),
        ),
        topology_edges=(
            PriorTopologyEdge(
                uid="edge_room_obj",
                source_uid="room_1",
                target_uid="obj_1",
                edge_type="room-object",
                relation="contains",
            ),
        ),
        source_format="json",
        frame_id="prior_map",
    )
    path = _write_json(tmp_path / "canonical.json", original.to_dict())

    loaded = PriorMapLoader().load(path)

    assert loaded == original


def test_load_floorplan_json_levels_regions_objects_and_metadata(tmp_path) -> None:
    path = _write_json(
        tmp_path / "floorplan.json",
        {
            "scene_id": "floor_scene",
            "frame_id": "floorplan",
            "levels": {
                "0": {
                    "regions": {
                        "kitchen_a": {
                            "type": "kitchen",
                            "boundaries": [[0, 0], [2, 0], [2, 2], [0, 2]],
                            "connections": ["living_a"],
                            "confidence": 0.8,
                            "source_page": 3,
                        },
                        "living_a": {
                            "type": "living room",
                            "center": [3.0, 1.0],
                            "connections": ["kitchen_a"],
                        },
                    }
                }
            },
            "objects": [
                {
                    "id": "fridge_1",
                    "type": "fridge",
                    "position": [1.0, 1.0, 0.0],
                    "parent_room": "kitchen_a",
                    "aliases": ["refrigerator"],
                }
            ],
            "metadata": {"source": "unit_test"},
        },
    )

    loaded = PriorMapLoader().load(path, source_format="floorplan_json")

    assert loaded.scene_id == "floor_scene"
    assert loaded.source_format == "floorplan_json"
    assert loaded.frame_id == "floorplan"
    assert len(loaded.rooms) == 2
    assert loaded.room_by_uid("kitchen_a").boundary_xy == ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0))
    assert loaded.room_by_uid("kitchen_a").metadata["source_page"] == 3
    assert loaded.object_by_uid("fridge_1").exact is True
    assert loaded.object_by_uid("fridge_1").aliases == ("refrigerator",)
    assert any(edge.edge_type == "room-room" for edge in loaded.topology_edges)
    assert any(edge.source_uid == "kitchen_a" and edge.target_uid == "fridge_1" for edge in loaded.topology_edges)
    assert loaded.metadata["source"] == "unit_test"


def test_load_floorplan_vln_json_uses_labels_and_3d_boundaries(tmp_path) -> None:
    path = _write_json(
        tmp_path / "mp3d_floorplan.json",
        {
            "levels": {
                "1": {
                    "regions": {
                        "7": {
                            "label": "bedroom",
                            "boundaries": [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
                            "center": [0.5, 0.5, 0.0],
                            "connectivity": [8],
                        }
                    }
                }
            }
        },
    )

    loaded = PriorMapLoader().load(path, source_format="floorplan_vln_json")

    assert loaded.source_format == "floorplan_vln_json"
    room = loaded.room_by_uid("7")
    assert room.label == "bedroom"
    assert room.level == 1
    assert room.boundary_xy == ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))
    assert room.centroid_xy == (0.5, 0.5)
    assert room.neighbors == ("8",)


def test_load_hm3d_generated_json_cleans_scene_id(tmp_path) -> None:
    path = _write_json(
        tmp_path / "00877-4ok3usBNeis_semantic.json",
        {
            "scene_id": "00877-4ok3usBNeis_semantic",
            "levels": {
                "0": {
                    "regions": {
                        "-199682": {
                            "type": "kitchen",
                            "boundaries": [[0, 0], [2, 0], [2, 2]],
                            "connections": ["-199683"],
                        }
                    }
                }
            },
        },
    )

    loaded = PriorMapLoader().load(path, source_format="hm3d_json")

    assert loaded.scene_id == "00877-4ok3usBNeis"
    assert loaded.source_format == "hm3d_json"
    assert loaded.room_by_uid("-199682").label == "kitchen"


def test_load_osm_xml_rooms_objects_and_topology(tmp_path) -> None:
    path = tmp_path / "lab.osm"
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6">
  <node id="1" lat="0.0" lon="0.0" />
  <node id="2" lat="0.0" lon="2.0" />
  <node id="3" lat="2.0" lon="2.0" />
  <node id="4" lat="2.0" lon="0.0" />
  <node id="100" lat="1.0" lon="1.0">
    <tag k="semantic_osmAG:object_name" v="microwave"/>
    <tag k="osmAG:parent" v="10"/>
  </node>
  <way id="10">
    <nd ref="1"/>
    <nd ref="2"/>
    <nd ref="3"/>
    <nd ref="4"/>
    <tag k="name" v="kitchen"/>
    <tag k="level" v="0"/>
    <tag k="semantic_osmAG:connections" v="11"/>
  </way>
</osm>
""",
        encoding="utf-8",
    )

    loaded = PriorMapLoader().load(path)

    assert loaded.source_format == "osm_xml"
    assert loaded.room_by_uid("10").label == "kitchen"
    assert loaded.room_by_uid("10").boundary_xy == ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0))
    assert loaded.object_by_uid("100").position_xyz == (1.0, 1.0, 0.0)
    assert loaded.object_by_uid("100").parent_room_uid == "10"
    assert any(edge.source_uid == "10" and edge.target_uid == "100" for edge in loaded.topology_edges)


def test_load_vlm_reconstruction_json_nested_payload(tmp_path) -> None:
    path = _write_json(
        tmp_path / "vlm_reconstruction.json",
        {
            "model": "offline_fixture",
            "prior_map_data": {
                "scene_id": "vlm_scene",
                "rooms": [
                    {"id": "room_a", "label": "study", "centroid": {"x": 1.0, "y": 2.0}},
                ],
                "objects": [
                    {
                        "id": "book_hint",
                        "label": "book",
                        "parent_room_uid": "room_a",
                        "exact": False,
                        "confidence": 0.3,
                    }
                ],
                "source_format": "vlm_reconstruction_json",
            },
        },
    )

    loaded = PriorMapLoader().load(path, source_format="vlm_reconstruction")

    assert loaded.scene_id == "vlm_scene"
    assert loaded.source_format == "vlm_reconstruction_json"
    assert loaded.room_by_uid("room_a").centroid_xy == (1.0, 2.0)
    assert loaded.object_by_uid("book_hint").exact is False


def test_loader_rejects_unknown_format(tmp_path) -> None:
    path = tmp_path / "map.unknown"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(PriorMapLoaderError):
        PriorMapLoader().load(path)
