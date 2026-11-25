"""
Unit tests for the Predict-to-Plan subsystem:
- SequenceExtractor
- Trajectory Predictors (CV, CA, DecisionTree, RandomForest)
- AStarTrajectoryPlanner
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

from scripts.run_predict_to_plan import load_benchmark_splits
from src.features.pandas_feature_pipeline import PandasFeaturePipeline
from src.models.trajectory_predictor import (
    ConstantAccelerationPredictor,
    ConstantVelocityPredictor,
    ManeuverClassifier,
    MLTrajectoryPredictor,
    evaluate_maneuver_classifier,
    evaluate_trajectory_predictor,
    tune_trajectory_predictor,
)
from src.planning.astar_planner import (
    AStarTrajectoryPlanner,
    DijkstraTrajectoryPlanner,
    GreedyTrajectoryPlanner,
    check_obb_collision,
    compute_wilson_score_interval,
)
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
        "lead_gap": np.random.uniform(2, 100, n),
        "lead_speed": np.random.uniform(0, 12, n),
        "lead_closing_speed": np.random.uniform(-3, 3, n),
        "closing_speed": np.random.uniform(-2, 4, n),
        "ttc": np.random.uniform(5, 50, n),
        "azimuth_angle": np.random.uniform(-np.pi, np.pi, n),
        "longitudinal_gap": np.random.uniform(5, 25, n),
        "lateral_gap": np.random.uniform(-5, 5, n),
        "yaw_rate_1s": np.random.uniform(-0.1, 0.1, n),
        "agent_accel_est": np.random.uniform(-1, 1, n),
    }

    # Ground truth future displacements
    data["future_dx_1s"] = data["agent_vx"] * 1.0 + np.random.normal(0, 0.1, n)
    data["future_dy_1s"] = data["agent_vy"] * 1.0 + np.random.normal(0, 0.1, n)
    data["future_dx_3s"] = data["agent_vx"] * 3.0 + np.random.normal(0, 0.3, n)
    data["future_dy_3s"] = data["agent_vy"] * 3.0 + np.random.normal(0, 0.3, n)
    data["future_dx_5s"] = data["agent_vx"] * 5.0 + np.random.normal(0, 0.5, n)
    data["future_dy_5s"] = data["agent_vy"] * 5.0 + np.random.normal(0, 0.5, n)
    data["future_agent_dx_3s"] = data["future_dx_3s"]
    data["future_agent_dy_3s"] = data["future_dy_3s"]
    data["maneuver_class"] = np.random.choice(
        ["STRAIGHT", "TURN_LEFT", "TURN_RIGHT", "DECELERATING", "ACCELERATING"], size=n
    )

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
    assert len(importances) == len(predictor.feature_cols)


def test_ml_trajectory_predictor_first_version_still_available(mock_trajectory_df):
    """frame="global", residual=False reproduces the first version (map frame, raw displacement)."""
    predictor = MLTrajectoryPredictor(
        model_type="random_forest", frame="global", residual=False, n_estimators=20, max_depth=5
    )
    predictor.fit(mock_trajectory_df.iloc[:70])
    assert predictor.feature_cols == PandasFeaturePipeline.FEATURE_COLS
    assert predictor.predict_displacements(mock_trajectory_df.iloc[70:])["5s"].shape == (30, 2)


def _rotate_scene(df, angle):
    """Rotates every global-frame column of the samples by `angle` (the scene seen from another compass direction)."""
    out = df.copy()
    c, s = np.cos(angle), np.sin(angle)
    for x_col, y_col in [
        ("agent_vx", "agent_vy"), ("past_dx_1s", "past_dy_1s"),
        ("future_dx_1s", "future_dy_1s"), ("future_dx_3s", "future_dy_3s"), ("future_dx_5s", "future_dy_5s"),
    ]:
        x, y = df[x_col].to_numpy(), df[y_col].to_numpy()
        out[x_col], out[y_col] = c * x - s * y, s * x + c * y
    out["agent_heading"] = df["agent_heading"] + angle
    out["ego_heading"] = df["ego_heading"] + angle
    return out


def test_agent_frame_prediction_does_not_depend_on_compass_direction(mock_trajectory_df):
    """
    Turning the whole scene by 90 degrees must turn the predictions by 90 degrees and nothing else:
    a vehicle going north and one going east are the same case for the agent-frame model.
    """
    train_df = mock_trajectory_df.iloc[:70]
    test_df = mock_trajectory_df.iloc[70:]
    angle = np.pi / 2

    predictor = MLTrajectoryPredictor(model_type="random_forest", n_estimators=20, max_depth=5).fit(train_df)
    preds = predictor.predict_displacements(test_df)["3s"]
    preds_rotated_scene = predictor.predict_displacements(_rotate_scene(test_df, angle))["3s"]

    c, s = np.cos(angle), np.sin(angle)
    expected = np.column_stack([c * preds[:, 0] - s * preds[:, 1], s * preds[:, 0] + c * preds[:, 1]])
    assert np.allclose(preds_rotated_scene, expected, atol=1e-6)


def test_missing_5s_target_is_ignored_not_invented(mock_trajectory_df):
    """Samples without a real 5s future (NaN) are skipped for the 5s model and the 5s metrics."""
    df = mock_trajectory_df.copy()
    df.loc[df.index[::4], ["future_dx_5s", "future_dy_5s"]] = np.nan

    predictor = MLTrajectoryPredictor(model_type="random_forest", n_estimators=20, max_depth=5).fit(df.iloc[:70])
    metrics, preds = evaluate_trajectory_predictor(predictor, df.iloc[70:])

    assert not np.isnan(preds["5s"]).any()
    assert np.isfinite(metrics.fde_5s) and np.isfinite(metrics.ade_5s)


def test_ml_trajectory_predictor_dt(mock_trajectory_df):
    train_df = mock_trajectory_df.iloc[:70]
    test_df = mock_trajectory_df.iloc[70:]

    predictor = MLTrajectoryPredictor(model_type="decision_tree", max_depth=4)
    predictor.fit(train_df)
    metrics, _ = evaluate_trajectory_predictor(predictor, test_df)
    assert metrics.ade_1s > 0.0


def test_ml_trajectory_predictor_gb(mock_trajectory_df):
    train_df = mock_trajectory_df.iloc[:70]
    test_df = mock_trajectory_df.iloc[70:]

    predictor = MLTrajectoryPredictor(model_type="gradient_boosting", max_iter=20, max_depth=3)
    predictor.fit(train_df)
    metrics, preds = evaluate_trajectory_predictor(predictor, test_df)
    assert preds["3s"].shape == (len(test_df), 2)
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


def test_maneuver_classifier_rf(mock_trajectory_df):
    train_df = mock_trajectory_df.iloc[:70]
    test_df = mock_trajectory_df.iloc[70:]

    clf = ManeuverClassifier(model_type="random_forest", n_estimators=20, max_depth=5)
    clf.fit(train_df)

    res = evaluate_maneuver_classifier(clf, test_df)
    assert res["accuracy"] >= 0.0
    assert res["f1_macro"] >= 0.0
    assert "report" in res

    preds = clf.predict(test_df)
    assert len(preds) == len(test_df)
    assert set(preds).issubset({"STRAIGHT", "TURN_LEFT", "TURN_RIGHT", "DECELERATING", "ACCELERATING"})


def test_maneuver_classifier_gb(mock_trajectory_df):
    train_df = mock_trajectory_df.iloc[:70]
    test_df = mock_trajectory_df.iloc[70:]

    clf = ManeuverClassifier(model_type="gradient_boosting", max_iter=20, max_depth=3)
    clf.fit(train_df)

    res = evaluate_maneuver_classifier(clf, test_df)
    assert res["accuracy"] >= 0.0
    assert res["f1_macro"] >= 0.0


def test_pandas_feature_pipeline_split(mock_trajectory_df):
    X, y_reg, y_cls, groups, metadata = PandasFeaturePipeline.get_feature_target_split(mock_trajectory_df)

    assert X.shape[1] == len(PandasFeaturePipeline.FEATURE_COLS)
    assert y_reg.shape[1] == len(PandasFeaturePipeline.TARGET_REGRESSION_COLS)
    assert len(y_cls) == len(mock_trajectory_df)
    assert len(groups) == len(mock_trajectory_df)
    assert metadata.shape[1] == len(PandasFeaturePipeline.METADATA_COLS)

    # Check scenario-based split
    df_train, df_test = PandasFeaturePipeline.create_scenario_based_split(mock_trajectory_df, test_size=0.3)
    train_scenarios = set(df_train["scenario_id"].unique())
    test_scenarios = set(df_test["scenario_id"].unique())
    # Zero leakage between scenarios
    assert len(train_scenarios.intersection(test_scenarios)) == 0



def test_predictor_accepts_samples_without_lead_columns(mock_trajectory_df):
    """Tables extracted before the lead-vehicle features existed must still train and predict."""
    old_table = mock_trajectory_df.drop(columns=["lead_gap", "lead_speed", "lead_closing_speed"])
    with pytest.warns(UserWarning, match="Lead-vehicle columns are missing"):
        predictor = MLTrajectoryPredictor(model_type="decision_tree", max_depth=3).fit(old_table)
    with pytest.warns(UserWarning):
        displacements = predictor.predict_displacements(old_table)
    assert displacements["3s"].shape == (len(old_table), 2)
    assert np.isfinite(displacements["3s"]).all()


def test_check_obb_collision_sat():
    # 1. Parallel vehicles in adjacent lanes (3.5m lateral distance): should NOT collide
    assert not check_obb_collision(
        ax=0.0, ay=0.0, a_head=0.0, a_l=4.8, a_w=2.0,
        bx=0.0, by=3.5, b_head=0.0, b_l=4.8, b_w=2.0,
        safety_margin=0.1
    )

    # 2. Side-by-side vehicle with only 1.8m lateral gap: SHOULD collide
    assert check_obb_collision(
        ax=0.0, ay=0.0, a_head=0.0, a_l=4.8, a_w=2.0,
        bx=0.0, by=1.8, b_head=0.0, b_l=4.8, b_w=2.0,
        safety_margin=0.1
    )

    # 3. In-lane vehicle ahead by 3.0m (bumper overlap): SHOULD collide
    assert check_obb_collision(
        ax=0.0, ay=0.0, a_head=0.0, a_l=4.8, a_w=2.0,
        bx=3.0, by=0.0, b_head=0.0, b_l=4.8, b_w=2.0,
        safety_margin=0.1
    )

    # 4. In-lane vehicle ahead by 6.0m: should NOT collide
    assert not check_obb_collision(
        ax=0.0, ay=0.0, a_head=0.0, a_l=4.8, a_w=2.0,
        bx=6.0, by=0.0, b_head=0.0, b_l=4.8, b_w=2.0,
        safety_margin=0.1
    )


def test_compute_wilson_score_interval():
    # 0 out of 100
    rate, lower, upper = compute_wilson_score_interval(0, 100)
    assert rate == 0.0
    assert 0.0 <= lower <= upper <= 5.0

    # 5 out of 100
    rate, lower, upper = compute_wilson_score_interval(5, 100)
    assert rate == 5.0
    assert 1.0 <= lower < 5.0 < upper <= 12.0

    # 50 out of 100
    rate, lower, upper = compute_wilson_score_interval(50, 100)
    assert rate == 50.0
    assert 40.0 <= lower < 50.0 < upper <= 60.0


def test_regression_metrics_mae_rmse_r2(mock_trajectory_df):
    train_df = mock_trajectory_df.iloc[:70]
    test_df = mock_trajectory_df.iloc[70:]

    pred = MLTrajectoryPredictor(model_type="decision_tree", max_depth=4)
    pred.fit(train_df)
    metrics, preds = evaluate_trajectory_predictor(pred, test_df)

    assert metrics.mae >= 0.0
    assert metrics.rmse >= 0.0
    assert -10.0 <= metrics.r2 <= 1.0
    assert metrics.mae_3s >= 0.0
    assert metrics.rmse_3s >= 0.0


@pytest.mark.parametrize("model_type", ["gradient_boosting", "random_forest"])
def test_tune_trajectory_predictor(mock_trajectory_df, monkeypatch, model_type):
    train_df = mock_trajectory_df.iloc[:60].copy()
    train_df.loc[train_df.index[0], ["future_dx_3s", "future_dy_3s"]] = np.nan
    validation_df = mock_trajectory_df.iloc[60:80]
    test_df = mock_trajectory_df.iloc[80:]
    original_fit = RandomizedSearchCV.fit

    def check_search_groups(search, X, y, *, groups):
        assert isinstance(search.cv, GroupKFold)
        assert groups.tolist() == train_df["scenario_id"].iloc[1:].tolist()
        return original_fit(search, X, y, groups=groups)

    monkeypatch.setattr(RandomizedSearchCV, "fit", check_search_groups)

    tuned_model, best_params = tune_trajectory_predictor(
        train_df,
        model_type=model_type,
        n_iter=1,
        cv=2,
        random_state=17,
        df_refit=pd.concat([train_df, validation_df]),
    )
    assert best_params
    for model in tuned_model.models.values():
        estimator = model.estimator if model_type == "gradient_boosting" else model
        actual_params = estimator.get_params()
        assert {key: actual_params[key] for key in best_params} == best_params
        assert actual_params["random_state"] == 17

    metrics, _ = evaluate_trajectory_predictor(tuned_model, test_df)
    assert metrics.mae >= 0.0
    assert metrics.ade_3s >= 0.0


def test_load_benchmark_splits_preserves_held_out_scenarios(tmp_path):
    for name, ids in [("train", ["001", "001"]), ("val", ["002"]), ("test", ["003"])]:
        pd.DataFrame({"scenario_id": ids}).to_csv(tmp_path / f"{name}.csv", index=False)

    train, val, test = load_benchmark_splits(tmp_path)
    assert train["scenario_id"].tolist() == ["001", "001"]
    assert val["scenario_id"].tolist() == ["002"]
    assert test["scenario_id"].tolist() == ["003"]

    val.to_csv(tmp_path / "test.csv", index=False)
    with pytest.raises(ValueError, match=r"val\.csv and test\.csv share"):
        load_benchmark_splits(tmp_path)


def test_spatiotemporal_2d_frenet_astar_evasion():
    path_len = 60.0
    s_pts = np.linspace(0, path_len, 60)
    ego_path = np.column_stack([s_pts, np.zeros_like(s_pts)])

    # Stationary obstacle right in the center lane at s=25m
    obs = {1: [(25.0, 0.0)] * 31}
    planner = AStarTrajectoryPlanner(horizon_steps=30, dt=0.1, max_speed=15.0)

    # 1D A* along centerline cannot evade laterally, must slow down/stop
    s_1d, v_1d, a_1d = planner.plan_trajectory(8.0, ego_path, obs)
    res_1d = planner.evaluate_plan_against_ground_truth(
        mode="1D", ego_path_xy=ego_path, s_traj=s_1d, v_traj=v_1d, a_traj=a_1d, gt_obstacles=obs
    )

    # 2D Frenet A* can change lane (evade laterally)
    s_2d, v_2d, a_2d, d_2d = planner.plan_trajectory_2d(
        ego_init_v=8.0, ego_path_xy=ego_path, predicted_obstacles=obs
    )
    res_2d = planner.evaluate_plan_against_ground_truth(
        mode="2D_Frenet", ego_path_xy=ego_path, s_traj=s_2d, v_traj=v_2d, a_traj=a_2d, gt_obstacles=obs, d_traj=d_2d
    )

    assert len(s_2d) == 31
    assert len(d_2d) == 31
    # 2D planner executed a lateral maneuver
    assert res_2d.max_lateral_displacement > 0.5
    # 2D planner maintained higher speed / progress than 1D
    assert res_2d.total_travel_distance >= res_1d.total_travel_distance


def test_planning_baselines_dijkstra_and_greedy():
    ego_path = np.column_stack([np.linspace(0, 50, 50), np.zeros(50)])
    obs = {1: [(20.0, 0.0)] * 31}

    dijkstra = DijkstraTrajectoryPlanner(horizon_steps=20, dt=0.1)
    greedy = GreedyTrajectoryPlanner(horizon_steps=20, dt=0.1)

    s_d, v_d, a_d = dijkstra.plan_trajectory(8.0, ego_path, obs)
    s_g, v_g, a_g = greedy.plan_trajectory(8.0, ego_path, obs)

    assert len(s_d) == 21
    assert len(s_g) == 21
    assert dijkstra.last_expansions > 0
    assert greedy.last_expansions > 0



