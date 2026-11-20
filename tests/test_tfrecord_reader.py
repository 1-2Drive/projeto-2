"""Tests for TFRecord discovery, metadata, and scenario streaming."""

import pytest
from pathlib import Path
from src.data.tfrecord_reader import (
    discover_tfrecords,
    get_dataset_metadata,
    decode_scenario,
    iter_scenarios,
)


def test_discover_tfrecords():
    files = discover_tfrecords()
    assert len(files) >= 1
    for f in files:
        assert f.exists()
        assert "tfrecord" in f.name


def test_discover_tfrecords_invalid_dir():
    with pytest.raises(FileNotFoundError):
        discover_tfrecords(data_dir="/nonexistent/directory/path/for/testing")


def test_get_dataset_metadata():
    metadata = get_dataset_metadata()
    assert metadata["file_count"] >= 1
    assert metadata["total_bytes"] > 0
    assert metadata["total_mb"] > 0
    assert len(metadata["files"]) == metadata["file_count"]


def test_iter_scenarios_limit():
    scenarios = list(iter_scenarios(max_scenarios=2))
    assert len(scenarios) == 2
    for sc in scenarios:
        assert hasattr(sc, "scenario_id")
        assert len(sc.scenario_id) > 0
        assert len(sc.timestamps_seconds) > 0
        assert len(sc.tracks) > 0
