from real_robot.detector_vocabulary import DetectorVocabularyAdapter, merge_label_provenance


def test_sysnav_objects_yaml_loads_label_and_prompt_space(tmp_path) -> None:
    config = tmp_path / "objects.yaml"
    config.write_text(
        """
prompts:
  tv_monitor:
    prompts:
      - tv_monitor
    is_instance: true
  garbage_bin:
    prompts:
      - trash can
    is_instance: true
  wall:
    prompts:
      - wall
    is_instance: false
""",
        encoding="utf-8",
    )

    vocabulary = DetectorVocabularyAdapter.from_sysnav_objects_yaml(str(config))

    assert vocabulary.label_space == ("tv_monitor", "garbage_bin", "wall")
    assert "trash can" in vocabulary.prompt_space
    assert vocabulary.find_entry("tv monitor").canonical_label == "tv_monitor"
    assert vocabulary.find_entry("trash can").canonical_label == "garbage_bin"


def test_detector_provenance_keeps_raw_label_and_records_config_match(tmp_path) -> None:
    config = tmp_path / "objects.yaml"
    config.write_text(
        """
prompts:
  garbage_bin:
    prompts:
      - trash can
    is_instance: true
""",
        encoding="utf-8",
    )
    vocabulary = DetectorVocabularyAdapter.from_sysnav_objects_yaml(str(config), detector_name="sysnav_yolo_world")

    provenance = vocabulary.provenance_for("trash can")

    assert provenance["raw_detector_label"] == "trash can"
    assert provenance["known_in_detector_vocabulary"] is True
    assert provenance["canonical_label"] == "garbage_bin"
    assert provenance["matched_by"] == "prompt"
    assert provenance["detector_name"] == "sysnav_yolo_world"


def test_unknown_label_provenance_is_explicit(tmp_path) -> None:
    config = tmp_path / "objects.yaml"
    config.write_text("prompts: {}\n", encoding="utf-8")
    vocabulary = DetectorVocabularyAdapter.from_sysnav_objects_yaml(str(config))

    provenance = vocabulary.provenance_for("bookcase")

    assert provenance["raw_detector_label"] == "bookcase"
    assert provenance["known_in_detector_vocabulary"] is False
    assert provenance["canonical_label"] is None
    assert provenance["prompt_labels"] == []


def test_merge_label_provenance_preserves_existing_metadata() -> None:
    merged = merge_label_provenance(
        {"source": "sysnav"},
        {"raw_detector_label": "cup", "known_in_detector_vocabulary": True},
    )

    assert merged["source"] == "sysnav"
    assert merged["label_provenance"]["raw_detector_label"] == "cup"
