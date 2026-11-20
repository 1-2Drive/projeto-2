"""Tests for data quality audits, schema consistency, and geometric transformations."""

import math
import numpy as np
import pandas as pd
import pytest
from src.features.feature_engineering import (
    compute_ttc,
    get_candidate_feature_names,
    get_planned_target_classes,
    transform_to_ego_frame,
)
from src.preprocessing.preprocessing import DataQualityChecker, run_full_quality_audit


def test_data_quality_clean(sample_scenarios):
    from src.data.scenario_parser import ScenarioParser
    parser = ScenarioParser()
    summary_df, objects_df, ego_df = parser.parse_scenarios(sample_scenarios)

    audit = run_full_quality_audit(summary_df, objects_df, ego_df)
    assert audit["overall_passed"] is True
    assert audit["summary_quality"]["duplicate_scenario_ids"] == 0
    assert audit["objects_quality"]["duplicate_object_ids_in_scenario"] == 0
    assert audit["objects_quality"]["negative_speeds"] == 0
    assert audit["objects_quality"]["infinite_speeds"] == 0


def test_data_quality_detects_flaws():
    """Verify quality checker catches corrupt or duplicate rows."""
    checker = DataQualityChecker()

    # Corrupt summary with duplicate scenario ID
    bad_summary = pd.DataFrame([
        {"scenario_id": "dup_1", "ego_valid": True},
        {"scenario_id": "dup_1", "ego_valid": True},
    ])
    res_summary = checker.check_summary_quality(bad_summary)
    assert res_summary["passed"] is False
    assert res_summary["duplicate_scenario_ids"] == 1

    # Corrupt objects with negative speed and duplicate object ID
    bad_objects = pd.DataFrame([
        {"scenario_id": "sc1", "object_id": 1, "valid": True, "speed": -5.0, "x": 0.0, "y": 0.0, "z": 0.0},
        {"scenario_id": "sc1", "object_id": 1, "valid": True, "speed": 10.0, "x": 1.0, "y": 1.0, "z": 0.0},
    ])
    res_objects = checker.check_objects_quality(bad_objects)
    assert res_objects["passed"] is False
    assert res_objects["duplicate_object_ids_in_scenario"] == 1
    assert res_objects["negative_speeds"] == 1


def test_ego_frame_transformation():
    # Ego at (0, 0) facing North (pi/2)
    # Target at (0, 10) in global coords is directly in front of ego (x_ego = 10, y_ego = 0)
    x_ego, y_ego = transform_to_ego_frame(0.0, 10.0, 0.0, 0.0, math.pi / 2)
    assert math.isclose(x_ego, 10.0, abs_tol=1e-5)
    assert math.isclose(y_ego, 0.0, abs_tol=1e-5)


def test_compute_ttc():
    # Distance = 20m, closing at 10m/s -> TTC = 2.0s
    assert math.isclose(compute_ttc(20.0, 10.0), 2.0)
    # Opening distance (negative relative speed) -> TTC = inf
    assert math.isinf(compute_ttc(20.0, -5.0))
    # Distance <= 0 -> TTC = 0
    assert compute_ttc(0.0, 5.0) == 0.0


def test_candidate_features_and_targets():
    features = get_candidate_feature_names()
    assert len(features) >= 15
    assert "ego_speed" in features
    assert "front_ttc" in features

    targets = get_planned_target_classes()
    assert "lateral" in targets
    assert "longitudinal" in targets
    assert "KEEP_LANE" in targets["lateral"]
    assert "ACCELERATE" in targets["longitudinal"]
