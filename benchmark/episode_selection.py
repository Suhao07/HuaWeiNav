"""Select benchmark tasks through Habitat's public episode API."""


def episode_identity(episode):
    """Return evaluator provenance, kept outside the Agent observation context."""
    return {
        "episode_id": str(episode.episode_id), "scene_id": episode.scene_id,
        "start_position": list(episode.start_position),
        "start_rotation": list(episode.start_rotation),
        "object_category": episode.object_category,
    }


def select_episode(env, index, *, expected_identity=None):
    """Bind the exact indexed episode to the next reset, including the first one.

    Habitat construction preselects an episode and retains it on first reset.
    Assigning current_episode uses the public setter to replace that selection;
    iterator internals and cycling behavior remain independent of evaluation.
    """
    episodes = env.episode_iterator.episodes
    if not 0 <= index < len(episodes):
        raise IndexError(f"Episode index {index} is outside [0, {len(episodes)})")
    if expected_identity is None:
        episode = episodes[index]
    else:
        # Habitat sampling can reorder even with shuffle=False. Frozen paired
        # evaluations bind the full task identity, including start pose/category.
        matches = [item for item in episodes if episode_identity(item) == expected_identity]
        if len(matches) != 1:
            raise ValueError(f"Expected one frozen episode identity, found {len(matches)}")
        episode = matches[0]
    env.current_episode = episode
    return episode_identity(episode)
