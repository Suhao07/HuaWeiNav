"""Select benchmark tasks through Habitat's public episode API."""


def episode_identity(episode):
    """Return evaluator provenance, kept outside the Agent observation context."""
    return {
        "episode_id": str(episode.episode_id), "scene_id": episode.scene_id,
        "start_position": list(episode.start_position),
        "start_rotation": list(episode.start_rotation),
        "object_category": episode.object_category,
    }


def select_episode(env, index):
    """Bind the exact indexed episode to the next reset, including the first one.

    Habitat construction preselects an episode and retains it on first reset.
    Assigning current_episode uses the public setter to replace that selection;
    iterator internals and cycling behavior remain independent of evaluation.
    """
    episodes = env.episode_iterator.episodes
    if not 0 <= index < len(episodes):
        raise IndexError(f"Episode index {index} is outside [0, {len(episodes)})")
    episode = episodes[index]
    env.current_episode = episode
    return episode_identity(episode)
