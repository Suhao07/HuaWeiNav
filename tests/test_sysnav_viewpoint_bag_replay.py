import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_replay_module():
    path = Path(__file__).parents[1] / "scripts" / "replay_sysnav_viewpoint_bag.py"
    spec = importlib.util.spec_from_file_location("replay_sysnav_viewpoint_bag", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _header(sec: int, nanosec: int = 0, frame_id: str = "map"):
    return SimpleNamespace(
        stamp=SimpleNamespace(sec=sec, nanosec=nanosec),
        frame_id=frame_id,
    )


def _odom(sec: int, nanosec: int, x: float, y: float):
    return SimpleNamespace(
        header=_header(sec, nanosec),
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=x, y=y, z=0.0),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            )
        ),
    )


def _viewpoint(viewpoint_id: int, sec: int, nanosec: int):
    return SimpleNamespace(
        header=_header(sec, nanosec),
        viewpoint_id=viewpoint_id,
    )


def _objects(viewpoint_id: int, *object_ids: int):
    return SimpleNamespace(
        nodes=[SimpleNamespace(viewpoint_id=viewpoint_id, object_id=list(object_ids))]
    )


class _FakeReader:
    events = []

    def __init__(self):
        self.index = 0
        self.opened = None

    def open(self, storage_options, converter_options):
        self.opened = (storage_options, converter_options)

    def get_all_topics_and_types(self):
        return [SimpleNamespace(name=topic, type=f"type:{topic}") for topic in self._topics]

    def has_next(self):
        return self.index < len(self.events)

    def read_next(self):
        event = self.events[self.index]
        self.index += 1
        return event


def _patch_fake_rosbag(monkeypatch, module, events, topics):
    class Reader(_FakeReader):
        pass

    Reader.events = list(events)
    Reader._topics = tuple(topics)
    monkeypatch.setattr(
        module,
        "_load_rosbag_runtime",
        lambda: (
            Reader,
            lambda **kwargs: kwargs,
            lambda **kwargs: kwargs,
            lambda serialized, message_type: serialized,
            lambda message_type: message_type,
        ),
    )


def test_bag_replay_joins_topics_and_preserves_exact_timestamp(monkeypatch, tmp_path) -> None:
    module = _load_replay_module()
    viewpoint_topic = "/viewpoint_rep_header"
    object_topic = "/object_nodes_list"
    odom_topic = "/aft_mapped_to_init"
    events = [
        (odom_topic, _odom(10, 123_456_789, 1.0, 2.0), 100),
        (viewpoint_topic, _viewpoint(7, 10, 123_456_789), 200),
        ("/camera/image", object(), 300),
        (object_topic, _objects(7, 42), 400),
    ]
    _patch_fake_rosbag(monkeypatch, module, events, [viewpoint_topic, object_topic, odom_topic])
    args = SimpleNamespace(
        bag_path=tmp_path,
        storage_id="sqlite3",
        viewpoint_topic=viewpoint_topic,
        object_topic=object_topic,
        odom_topic=odom_topic,
        max_time_offset_s=0.001,
        odom_history_size=10,
    )
    output = io.StringIO()

    count = module.replay_bag(args, output)
    records = [json.loads(line) for line in output.getvalue().splitlines()]

    assert count == 2
    assert records[0]["timestamp_ns"] == 10_123_456_789
    assert records[0]["pose"]["position"] == [1.0, 2.0, 0.0]
    assert records[0]["observed_object_ids"] == []
    assert records[1]["observed_object_ids"] == [42]
    assert records[1]["bag_timestamp_ns"] == 400


def test_bag_replay_rejects_missing_sysnav_topic(monkeypatch, tmp_path) -> None:
    module = _load_replay_module()
    _patch_fake_rosbag(monkeypatch, module, [], ["/object_nodes_list", "/aft_mapped_to_init"])
    args = SimpleNamespace(
        bag_path=tmp_path,
        storage_id="sqlite3",
        viewpoint_topic="/viewpoint_rep_header",
        object_topic="/object_nodes_list",
        odom_topic="/aft_mapped_to_init",
        max_time_offset_s=0.25,
        odom_history_size=10,
    )

    with pytest.raises(RuntimeError, match="viewpoint_rep_header"):
        module.replay_bag(args, io.StringIO())
