import gzip
import json
from argparse import Namespace

from benchmark import get_provider


def _write_dataset(path, episodes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({"episodes": episodes, "category_to_task_category_id": {"tv_monitor": 0}}, f)


def test_hm3d_ovon_provider_materializes_single_episode(tmp_path):
    root = tmp_path / "hm3d_ovon" / "v1"
    content = root / "val_seen_instruction_2k" / "content" / "sceneA.json.gz"
    _write_dataset(
        content,
        [
            {"episode_id": "0", "object_category": "chair"},
            {"episode_id": "1", "object_category": "tv_monitor"},
        ],
    )
    args = Namespace(
        benchmark="hm3d_ovon",
        benchmark_split="val_seen_instruction_2k",
        dataset_root=str(root),
        dataset_path="",
        scene_id="sceneA",
        scene_id_contains=None,
        object_category="tv monitor",
        episode_rank=0,
        filtered_dataset_dir=str(tmp_path / "filtered"),
        success_distance=None,
        split="",
        eval_episodes=100,
        dataset_episodes=None,
        start_episode=0,
    )

    spec = get_provider(args).prepare(args)

    assert spec.benchmark == "hm3d_ovon"
    assert spec.split == "val_seen_instruction_2k"
    assert spec.success_distance == 1.5
    assert spec.is_filtered is True
    assert spec.source_file == str(content)
    assert args.eval_episodes == 1

    with gzip.open(spec.dataset_path, "rt", encoding="utf-8") as f:
        filtered = json.load(f)
    assert [ep["episode_id"] for ep in filtered["episodes"]] == ["1"]
