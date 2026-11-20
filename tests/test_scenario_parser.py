"""Tests for Scenario parsing, ego kinematics, and invalid-state handling."""

import math
import numpy as np
import pytest
from waymo_open_dataset.protos import scenario_pb2
from src.data.scenario_parser import (
    calculate_speed,
    parse_scenario_summary,
    parse_objects_at_timestamp,
    parse_ego_at_timestamp,
    ScenarioParser,
)


def test_calculate_speed():
    # 3-4-5 right triangle
    assert math.isclose(calculate_speed(3.0, 4.0), 5.0)
    assert calculate_speed(0.0, 0.0) == 0.0
    # Negative velocities
    assert math.isclose(calculate_speed(-3.0, -4.0), 5.0)
    # NaN handling
    assert calculate_speed(float("nan"), 4.0) == 0.0


def test_parse_scenario_summary(sample_scenarios):
    sc = sample_scenarios[0]
    summary = parse_scenario_summary(sc)

    assert "scenario_id" in summary
    assert summary["scenario_id"] == sc.scenario_id
    assert summary["num_tracks"] == len(sc.tracks)
    assert summary["num_vehicles"] + summary["num_pedestrians"] + summary["num_cyclists"] + summary["num_other"] == len(sc.tracks)
    assert summary["ego_valid"] is True
    assert summary["num_lanes"] >= 0


def test_parse_objects_at_timestamp(sample_scenarios):
    sc = sample_scenarios[0]
    objects = parse_objects_at_timestamp(sc, time_index=0)

    assert len(objects) == len(sc.tracks)
    expected_cols = {
        "scenario_id", "object_id", "object_type", "is_ego", "x", "y", "z",
        "velocity_x", "velocity_y", "speed", "heading", "length", "width", "height", "valid"
    }
    assert expected_cols.issubset(set(objects[0].keys()))

    ego_count = sum(1 for obj in objects if obj["is_ego"])
    assert ego_count == 1


def test_parse_ego_at_timestamp(sample_scenarios):
    sc = sample_scenarios[0]
    ego = parse_ego_at_timestamp(sc, time_index=0, nearby_radius=50.0)

    assert ego["scenario_id"] == sc.scenario_id
    assert not np.isnan(ego["ego_x"])
    assert not np.isnan(ego["ego_y"])
    assert not np.isnan(ego["ego_speed"])
    assert ego["nearby_vehicle_count"] >= 0
    assert ego["nearby_pedestrian_count"] >= 0
    assert ego["nearby_cyclist_count"] >= 0


def test_invalid_state_handling():
    """Verify behavior when scenario has invalid or out-of-bounds states."""
    sc = scenario_pb2.Scenario()
    sc.scenario_id = "test_invalid_scenario"
    sc.sdc_track_index = 0

    # Add a track with invalid state
    track = sc.tracks.add()
    track.id = 1
    track.object_type = scenario_pb2.Track.ObjectType.TYPE_VEHICLE
    st = track.states.add()
    st.valid = False

    summary = parse_scenario_summary(sc)
    assert summary["ego_valid"] is False

    objects = parse_objects_at_timestamp(sc, time_index=0)
    assert len(objects) == 1
    assert objects[0]["valid"] is False
    assert np.isnan(objects[0]["x"])
    assert np.isnan(objects[0]["speed"])

    ego = parse_ego_at_timestamp(sc, time_index=0)
    assert np.isnan(ego["ego_x"])
    assert np.isnan(ego["ego_speed"])
