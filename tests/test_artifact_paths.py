from artifact_utils.path_builder import detection_step_dir, episode_dir, episode_subdir


def test_episode_path_builders_create_stable_layout(tmp_path):
    """Artifact paths should follow the repository log directory convention."""

    root = str(tmp_path)

    assert episode_dir(root, 3) == str(tmp_path / "episode-3")
    assert episode_subdir(root, 3, "obs") == str(tmp_path / "episode-3" / "obs")
    assert detection_step_dir(root, 3, 17) == str(
        tmp_path / "episode-3" / "detection" / "step_17"
    )
    assert (tmp_path / "episode-3" / "detection" / "step_17").is_dir()
