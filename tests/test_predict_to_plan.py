"""
Unit tests for the Predict-to-Plan subsystem:
- SequenceExtractor
- Trajectory Predictors (CV, CA, DecisionTree, RandomForest)
- AStarTrajectoryPlanner
"""

import numpy as np
import pandas as pd
import pytest

from src.models.trajectory_predictor import (
    ConstantAccelerationPredictor,
    ConstantVelocityPredictor,
    MLTrajectoryPredictor,
    evaluate_trajectory_predictor,
)
from src.planning.astar_planner import AStarTrajectoryPlanner
from src.preprocessing.sequence_extractor import SequenceExtractor, TrajectorySample


@pytest.fixture
def mock_trajectory_df():
    """Generates synthetic interactive trajectory samples for test validation."""
    np.random.seed(42)
    n = 100
    data = {
        "scenario_id": [f"scen_{i//10}" for i in range(n)],
        "track_id": list(range(n)),
        "object_type": [1] * n,
        "current_time_step": [20] * n,
        "ego_x": np.random.uniform(0, 10, n),
        "ego_y": np.random.uniform(0, 10, n),
        "ego_speed": np.random.uniform(5, 15, n),
        "ego_heading": np.random.uniform(-np.pi, np.pi, n),
        "agent_x": np.random.uniform(15, 30, n),
        "agent_y": np.random.uniform(15, 30, n),
        "agent_vx": np.random.uniform(2, 10, n),
        "agent_vy": np.random.uniform(0, 5, n),
        "agent_speed": np.random.uniform(2, 12, n),
        "agent_heading": np.random.uniform(-np.pi, np.pi, n),
        "rel_x": np.random.uniform(5, 25, n),
        "rel_y": np.random.uniform(-5, 5, n),
        "rel_vx": np.random.uniform(-2, 2, n),
        "rel_vy": np.random.uniform(-1, 1, n),
        "rel_heading": np.random.uniform(-0.5, 0.5, n),
        "distance_to_ego": np.random.uniform(5, 30, n),
        "past_dx_1s": np.random.uniform(2, 8, n),
        "past_dy_1s": np.random.uniform(0, 3, n),
        "past_speed_mean": np.random.uniform(2, 10, n),
        "past_heading_change": np.random.uniform(-0.1, 0.1, n),
    }

    # Ground truth future displacements
    data["future_dx_1s"] = data["agent_vx"] * 1.0 + np.random.normal(0, 0.1, n)
    data["future_dy_1s"] = data["agent_vy"] * 1.0 + np.random.normal(0, 0.1, n)
    data["future_dx_3s"] = data["agent_vx"] * 3.0 + np.random.normal(0, 0.3, n)
    data["future_dy_3s"] = data["agent_vy"] * 3.0 + np.random.normal(0, 0.3, n)
    data["future_dx_5s"] = data["agent_vx"] * 5.0 + np.random.normal(0, 0.5, n)
    data["future_dy_5s"] = data["agent_vy"] * 5.0 + np.random.normal(0, 0.5, n)

    return pd.DataFrame(data)


def test_constant_velocity_predictor(mock_trajectory_df):
    predictor = ConstantVelocityPredictor()
    metrics, preds = evaluate_trajectory_predictor(predictor, mock_trajectory_df)

    assert "1s" in preds and "3s" in preds and "5s" in preds
    assert preds["1s"].shape == (len(mock_trajectory_df), 2)
    assert metrics.ade_1s >= 0.0
    assert metrics.fde_3s >= 0.0


def test_constant_acceleration_predictor(mock_trajectory_df):
    predictor = ConstantAccelerationPredictor()
    metrics, preds = evaluate_trajectory_predictor(predictor, mock_trajectory_df)

    assert preds["3s"].shape == (len(mock_trajectory_df), 2)
    assert metrics.ade_3s > 0.0


def test_ml_trajectory_predictor_rf(mock_trajectory_df):
    train_df = mock_trajectory_df.iloc[:70]
    test_df = mock_trajectory_df.iloc[70:]

    predictor = MLTrajectoryPredictor(model_type="random_forest", n_estimators=20, max_depth=5)
    predictor.fit(train_df)
    metrics, preds = evaluate_trajectory_predictor(predictor, test_df)

    assert preds["3s"].shape == (len(test_df), 2)
    assert metrics.ade_3s < 10.0  # Reasonable error on synthetic data
    importances = predictor.get_feature_importances()
    assert importances is not None
    assert len(importances) == len(MLTrajectoryPredictor.FEATURE_COLS)


def test_ml_trajectory_predictor_dt(mock_trajectory_df):
    train_df = mock_trajectory_df.iloc[:70]
    test_df = mock_trajectory_df.iloc[70:]

    predictor = MLTrajectoryPredictor(model_type="decision_tree", max_depth=4)
    predictor.fit(train_df)
    metrics, _ = evaluate_trajectory_predictor(predictor, test_df)
    assert metrics.ade_1s > 0.0


def test_astar_planner_collision_avoidance():
    # Setup ego path: straight line from x=0 to x=50, y=0
    ego_path = np.column_stack([np.linspace(0, 50, 100), np.zeros(100)])
    planner = AStarTrajectoryPlanner(horizon_steps=20, dt=0.1, max_speed=15.0)

    # Obstacle crossing path at x=20, y moving from -5 to +5
    obs_gt = {101: [(20.0, -5.0 + 0.5 * step) for step in range(25)]}

    # Plan with predicted obstacle
    s_traj, v_traj, a_traj = planner.plan_trajectory(
        ego_init_v=10.0, ego_path_xy=ego_path, predicted_obstacles=obs_gt
    )

    assert len(s_traj) == 21
    assert len(v_traj) == 21
    assert len(a_traj) == 21

    # Evaluate plan
    res = planner.evaluate_plan_against_ground_truth(
        mode="test_eval",
        ego_path_xy=ego_path,
        s_traj=s_traj,
        v_traj=v_traj,
        a_traj=a_traj,
        gt_obstacles=obs_gt,
    )
    assert res.total_travel_distance > 0.0
    assert res.average_speed >= 0.0
