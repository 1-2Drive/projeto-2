"""
Trajectory Prediction Models for Waymo Dynamic Obstacles.
Implements:
1. Physics Baselines (Constant Velocity, Constant Acceleration)
2. Machine Learning Models (Decision Tree, Random Forest, KNN)
Evaluates standard autonomous driving metrics: ADE, FDE, and inference latency.
"""

from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


@dataclass
class PredictionMetrics:
    model_name: str
    ade_1s: float
    fde_1s: float
    ade_3s: float
    fde_3s: float
    ade_5s: float
    fde_5s: float
    mean_inference_time_ms: float


class BasePredictor:
    """Base interface for trajectory predictors."""

    def predict_displacements(self, df_features: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Returns dictionary mapping horizon to array of shape (N, 2) containing [dx, dy].
        Horizons: '1s', '3s', '5s'.
        """
        raise NotImplementedError


class ConstantVelocityPredictor(BasePredictor):
    """
    Standard physics baseline: extrapolates current velocity forward in time.
    """

    def __init__(self, name: str = "Constant Velocity (CV)"):
        self.name = name

    def predict_displacements(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        vx = df["agent_vx"].to_numpy()
        vy = df["agent_vy"].to_numpy()

        # delta_t = 1.0s, 3.0s, 5.0s
        pred_1s = np.column_stack([vx * 1.0, vy * 1.0])
        pred_3s = np.column_stack([vx * 3.0, vy * 3.0])
        pred_5s = np.column_stack([vx * 5.0, vy * 5.0])

        return {"1s": pred_1s, "3s": pred_3s, "5s": pred_5s}


class ConstantAccelerationPredictor(BasePredictor):
    """
    Physics baseline: estimates acceleration from past 1s displacement and extrapolates.
    """

    def __init__(self, name: str = "Constant Acceleration (CA)"):
        self.name = name

    def predict_displacements(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        vx = df["agent_vx"].to_numpy()
        vy = df["agent_vy"].to_numpy()

        # past displacement over 1s: past_dx_1s
        # v_mean = past_d / 1.0
        # a = 2 * (v_curr - v_mean) / dt
        past_dx = df["past_dx_1s"].to_numpy()
        past_dy = df["past_dy_1s"].to_numpy()

        ax = 2.0 * (vx - past_dx)
        ay = 2.0 * (vy - past_dy)

        # Dampen extreme accelerations
        ax = np.clip(ax, -6.0, 6.0)
        ay = np.clip(ay, -6.0, 6.0)

        dt1 = 1.0
        pred_1s = np.column_stack([vx * dt1 + 0.5 * ax * (dt1**2), vy * dt1 + 0.5 * ay * (dt1**2)])

        dt3 = 3.0
        pred_3s = np.column_stack([vx * dt3 + 0.5 * ax * (dt3**2), vy * dt3 + 0.5 * ay * (dt3**2)])

        dt5 = 5.0
        pred_5s = np.column_stack([vx * dt5 + 0.5 * ax * (dt5**2), vy * dt5 + 0.5 * ay * (dt5**2)])

        return {"1s": pred_1s, "3s": pred_3s, "5s": pred_5s}


class MLTrajectoryPredictor(BasePredictor):
    """
    Machine Learning regressor predicting multi-horizon future trajectory displacements.
    Supports Decision Tree, Random Forest, and KNN.
    """

    FEATURE_COLS = [
        "agent_speed",
        "agent_heading",
        "agent_vx",
        "agent_vy",
        "rel_x",
        "rel_y",
        "rel_vx",
        "rel_vy",
        "rel_heading",
        "distance_to_ego",
        "past_dx_1s",
        "past_dy_1s",
        "past_speed_mean",
        "past_heading_change",
        "ego_speed",
        "closing_speed",
        "ttc",
        "azimuth_angle",
        "longitudinal_gap",
        "lateral_gap",
        "yaw_rate_1s",
        "agent_accel_est",
    ]

    TARGET_COLS = [
        "future_dx_1s",
        "future_dy_1s",
        "future_dx_3s",
        "future_dy_3s",
        "future_dx_5s",
        "future_dy_5s",
    ]

    def __init__(self, model_type: str = "random_forest", **kwargs):
        self.model_type = model_type
        self.kwargs = kwargs
        if model_type == "decision_tree":
            self.model = DecisionTreeRegressor(max_depth=kwargs.get("max_depth", 8), random_state=42)
            self.name = f"Decision Tree (depth={kwargs.get('max_depth', 8)})"
        elif model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=kwargs.get("n_estimators", 50),
                max_depth=kwargs.get("max_depth", 10),
                random_state=42,
                n_jobs=-1,
            )
            self.name = f"Random Forest (trees={kwargs.get('n_estimators', 50)})"
        elif model_type == "knn":
            self.model = KNeighborsRegressor(n_neighbors=kwargs.get("n_neighbors", 7), n_jobs=-1)
            self.name = f"KNN (k={kwargs.get('n_neighbors', 7)})"
        elif model_type == "gradient_boosting":
            # MultiOutput using HistGradientBoosting
            from sklearn.multioutput import MultiOutputRegressor
            base_gb = HistGradientBoostingRegressor(
                max_iter=kwargs.get("max_iter", 100),
                max_depth=kwargs.get("max_depth", 6),
                learning_rate=kwargs.get("learning_rate", 0.1),
                random_state=42,
            )
            self.model = MultiOutputRegressor(base_gb, n_jobs=-1)
            self.name = f"Gradient Boosting (HistGB, iter={kwargs.get('max_iter', 100)})"
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def fit(self, df_train: pd.DataFrame):
        X = df_train[self.FEATURE_COLS].to_numpy()
        y = df_train[self.TARGET_COLS].to_numpy()
        self.model.fit(X, y)
        return self

    def predict_displacements(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        X = df[self.FEATURE_COLS].to_numpy()
        preds = self.model.predict(X)
        pred_1s = preds[:, 0:2]
        pred_3s = preds[:, 2:4]
        pred_5s = preds[:, 4:6]
        return {"1s": pred_1s, "3s": pred_3s, "5s": pred_5s}

    def get_feature_importances(self) -> Optional[pd.Series]:
        if hasattr(self.model, "feature_importances_"):
            return pd.Series(self.model.feature_importances_, index=self.FEATURE_COLS).sort_values(ascending=False)
        return None


def evaluate_trajectory_predictor(
    predictor: BasePredictor, df_eval: pd.DataFrame
) -> Tuple[PredictionMetrics, Dict[str, np.ndarray]]:
    """
    Evaluates Average Displacement Error (ADE) and Final Displacement Error (FDE)
    at 1s, 3s, and 5s horizons on test data.
    """
    t0 = time.perf_counter()
    preds = predictor.predict_displacements(df_eval)
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0 / max(1, len(df_eval))

    gt_1s = df_eval[["future_dx_1s", "future_dy_1s"]].to_numpy()
    gt_3s = df_eval[["future_dx_3s", "future_dy_3s"]].to_numpy()
    gt_5s = df_eval[["future_dx_5s", "future_dy_5s"]].to_numpy()

    err_1s = np.linalg.norm(preds["1s"] - gt_1s, axis=1)
    err_3s = np.linalg.norm(preds["3s"] - gt_3s, axis=1)
    err_5s = np.linalg.norm(preds["5s"] - gt_5s, axis=1)

    fde_1s = float(np.mean(err_1s))
    fde_3s = float(np.mean(err_3s))
    fde_5s = float(np.mean(err_5s))

    ade_1s = fde_1s
    ade_3s = float(np.mean((err_1s + err_3s) / 2.0))
    ade_5s = float(np.mean((err_1s + err_3s + err_5s) / 3.0))

    metrics = PredictionMetrics(
        model_name=getattr(predictor, "name", type(predictor).__name__),
        ade_1s=round(ade_1s, 3),
        fde_1s=round(fde_1s, 3),
        ade_3s=round(ade_3s, 3),
        fde_3s=round(fde_3s, 3),
        ade_5s=round(ade_5s, 3),
        fde_5s=round(fde_5s, 3),
        mean_inference_time_ms=round(latency_ms, 4),
    )
    return metrics, preds
