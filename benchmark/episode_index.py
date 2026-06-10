"""Dataset indexing helpers for Habitat ObjectNav JSON files."""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union


PathLike = Union[str, Path]


def load_json_gz(path: PathLike) -> Dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def write_json_gz(payload: Dict, path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


def canonical_category(value: object) -> str:
    """Normalize category labels without introducing semantic aliases.

    Benchmark filtering should not guess that two different object categories
    are equivalent.  We only remove presentation differences such as spaces,
    hyphens and repeated separators.
    """

    text = str(value or "").strip().lower()
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def episode_category(episode: Dict) -> str:
    return canonical_category(
        episode.get("object_category", episode.get("object_category_name", ""))
    )


def filter_episodes_by_category(episodes: Iterable[Dict], object_category: str) -> List[Dict]:
    target = canonical_category(object_category)
    return [episode for episode in episodes if episode_category(episode) == target]


def list_episode_categories(episodes: Iterable[Dict]) -> List[str]:
    return sorted({episode_category(episode) for episode in episodes if episode_category(episode)})


def find_matching_episode_file(
    candidate_files: Iterable[Path],
    object_category: str,
) -> Tuple[Path, Dict, List[Dict]]:
    searched = []
    for path in candidate_files:
        searched.append(str(path))
        if not path.exists():
            continue
        dataset = load_json_gz(path)
        episodes = filter_episodes_by_category(dataset.get("episodes", []), object_category)
        if episodes:
            return path, dataset, episodes

    raise FileNotFoundError(
        "No episode matched object_category={!r}. Searched: {}".format(
            object_category,
            searched,
        )
    )
