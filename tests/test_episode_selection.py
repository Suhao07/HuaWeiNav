"""Reproduce Habitat's preselected first-reset semantics without simulator IO."""
from types import SimpleNamespace
import pytest
from benchmark.episode_selection import select_episode, episode_identity


class Environment:
    def __init__(self):
        self.episode_iterator = SimpleNamespace(episodes=[SimpleNamespace(
            episode_id=str(i), scene_id="scene", start_position=[i, 0, 0],
            start_rotation=[0, 0, 0, 1], object_category="cup",
        ) for i in range(5)])
        self.current_episode = self.episode_iterator.episodes[0]

    def reset(self):
        return self.current_episode


def test_independent_first_resets_select_five_distinct_episodes():
    identities = []
    for i in range(5):
        env = Environment()
        expected = select_episode(env, i)
        assert episode_identity(env.reset()) == expected
        identities.append(expected["episode_id"])
    assert identities == ["0", "1", "2", "3", "4"]


def test_selection_is_stable_and_out_of_range_does_not_wrap():
    env = Environment()
    select_episode(env, 4)
    assert select_episode(env, 1)["episode_id"] == "1"
    for index in (-1, 5):
        with pytest.raises(IndexError):
            select_episode(env, index)
