"""
Trajectory Prediction Models for Waymo Dynamic Obstacles.
Implements:
1. Physics Baselines (Constant Velocity, Constant Acceleration)
2. Machine Learning Models (Decision Tree, Random Forest, KNN)
Evaluates standard autonomous driving metrics: ADE, FDE, and inference latency.
"""

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


from src.features.pandas_feature_pipeline import PandasFeaturePipeline


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
    mae: float = 0.0
    rmse: float = 0.0
    r2: float = 0.0
    mae_3s: float = 0.0
    rmse_3s: float = 0.0
    r2_3s: float = 0.0


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
    Supports Decision Tree, Random Forest, KNN, and Gradient Boosting.
    """

    FEATURE_COLS = PandasFeaturePipeline.FEATURE_COLS
    TARGET_COLS = PandasFeaturePipeline.TARGET_REGRESSION_COLS

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

    gt_all = np.column_stack([gt_1s, gt_3s, gt_5s])
    preds_all = np.column_stack([preds["1s"], preds["3s"], preds["5s"]])

    # Global regression metrics across all displacement components
    mae_global = float(mean_absolute_error(gt_all, preds_all))
    rmse_global = float(np.sqrt(mean_squared_error(gt_all, preds_all)))
    r2_global = float(r2_score(gt_all, preds_all))

    # Horizon-specific metrics at 3.0s (planning horizon target)
    mae_3s = float(mean_absolute_error(gt_3s, preds["3s"]))
    rmse_3s = float(np.sqrt(mean_squared_error(gt_3s, preds["3s"])))
    r2_3s = float(r2_score(gt_3s, preds["3s"]))

    metrics = PredictionMetrics(
        model_name=getattr(predictor, "name", type(predictor).__name__),
        ade_1s=round(ade_1s, 3),
        fde_1s=round(fde_1s, 3),
        ade_3s=round(ade_3s, 3),
        fde_3s=round(fde_3s, 3),
        ade_5s=round(ade_5s, 3),
        fde_5s=round(fde_5s, 3),
        mean_inference_time_ms=round(latency_ms, 4),
        mae=round(mae_global, 3),
        rmse=round(rmse_global, 3),
        r2=round(r2_global, 3),
        mae_3s=round(mae_3s, 3),
        rmse_3s=round(rmse_3s, 3),
        r2_3s=round(r2_3s, 3),
    )
    return metrics, preds


def tune_trajectory_predictor(
    df_train: pd.DataFrame,
    model_type: str = "gradient_boosting",
    n_iter: int = 15,
    cv: int = 3,
    random_state: int = 42,
) -> Tuple[MLTrajectoryPredictor, Dict[str, Any]]:
    """
    Performs hyperparameter optimization using RandomizedSearchCV on trajectory regressors.
    Optimizes for negative Mean Absolute Error (neg_mean_absolute_error).
    Returns: (tuned_predictor, best_parameters_dict)
    """
    X = df_train[MLTrajectoryPredictor.FEATURE_COLS].to_numpy()
    y = df_train[MLTrajectoryPredictor.TARGET_COLS].to_numpy()

    if model_type == "gradient_boosting":
        base_estimator = MultiOutputRegressor(
            HistGradientBoostingRegressor(random_state=random_state)
        )
        param_distributions = {
            "estimator__learning_rate": [0.03, 0.05, 0.08, 0.1, 0.15],
            "estimator__max_iter": [50, 80, 100, 150],
            "estimator__max_depth": [4, 5, 6, 8],
            "estimator__min_samples_leaf": [15, 20, 30, 50],
            "estimator__l2_regularization": [0.0, 0.1, 1.0, 5.0],
        }
    elif model_type == "random_forest":
        base_estimator = RandomForestRegressor(random_state=random_state, n_jobs=-1)
        param_distributions = {
            "n_estimators": [40, 60, 80, 100],
            "max_depth": [6, 8, 10, 12, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 1.0],
        }
    else:
        raise ValueError(f"Tuning not implemented for model_type: {model_type}")

    search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_absolute_error",
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y)

    best_predictor = MLTrajectoryPredictor(
        model_type=model_type,
        name=f"Tuned {model_type.replace('_', ' ').title()}",
    )
    best_predictor.model = search.best_estimator_

    # Clean param dict
    best_params = {
        k.replace("estimator__", ""): v for k, v in search.best_params_.items()
    }
    return best_predictor, best_params


class ManeuverClassifier:
    """
    Classifies vehicle intention at horizon t+3.0s into one of:
    - STRAIGHT, TURN_LEFT, TURN_RIGHT, DECELERATING, ACCELERATING.
    """

    FEATURE_COLS = PandasFeaturePipeline.FEATURE_COLS
    TARGET_COL = PandasFeaturePipeline.TARGET_MANEUVER_COL

    def __init__(self, model_type: str = "random_forest", **kwargs):
        self.model_type = model_type
        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 80),
                max_depth=kwargs.get("max_depth", 8),
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            self.name = f"Random Forest Classifier (trees={kwargs.get('n_estimators', 80)})"
        elif model_type == "gradient_boosting":
            self.model = HistGradientBoostingClassifier(
                max_iter=kwargs.get("max_iter", 100),
                max_depth=kwargs.get("max_depth", 6),
                class_weight="balanced",
                random_state=42,
            )
            self.name = f"HistGradientBoosting Classifier (iter={kwargs.get('max_iter', 100)})"
        else:
            raise ValueError(f"Unknown classifier model_type: {model_type}")

    def fit(self, df_train: pd.DataFrame):
        X = df_train[self.FEATURE_COLS].to_numpy()
        y = df_train[self.TARGET_COL].to_numpy()
        self.model.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.FEATURE_COLS].to_numpy()
        return self.model.predict(X)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.FEATURE_COLS].to_numpy()
        return self.model.predict_proba(X)


def evaluate_maneuver_classifier(
    classifier: ManeuverClassifier, df_eval: pd.DataFrame
) -> Dict[str, Any]:
    """
    Evaluates classification performance (Accuracy, Macro-F1, Weighted-F1).
    """
    y_true = df_eval[classifier.TARGET_COL].to_numpy()
    y_pred = classifier.predict(df_eval)

    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted"))
    rep = classification_report(y_true, y_pred, output_dict=True)

    return {
        "model_name": classifier.name,
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "report": rep,
    }
