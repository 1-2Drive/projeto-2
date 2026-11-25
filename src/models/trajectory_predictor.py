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
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
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

    Two design choices (both can be switched off to reproduce the first version of this class):

    - frame="agent": features and targets are expressed in the vehicle's own frame
      (forward / left) instead of the global map frame (east / north). A vehicle driving
      straight north and one driving straight east become the same case for the model.
    - residual=True: the model does not predict the displacement itself but the CORRECTION
      to the constant-velocity displacement (v * t). Physics does the easy part; the model
      only learns the deviations (braking, accelerating, turning). Predicting a zero
      correction gives back the constant-velocity baseline.

    One model is trained per horizon, each on the samples whose target really exists
    (the 5s target is missing when the vehicle is not observed 5s later).
    predict_displacements() always returns displacements in the GLOBAL frame, so the
    planner and the evaluation code do not change.
    """

    FEATURE_COLS = PandasFeaturePipeline.FEATURE_COLS
    AGENT_FRAME_FEATURE_COLS = PandasFeaturePipeline.AGENT_FRAME_FEATURE_COLS
    TARGET_COLS = PandasFeaturePipeline.TARGET_REGRESSION_COLS
    HORIZONS = (1, 3, 5)

    def __init__(self, model_type: str = "random_forest", frame: str = "agent", residual: bool = True, **kwargs):
        if frame not in ("agent", "global"):
            raise ValueError(f"Unknown frame: {frame}")
        self.model_type = model_type
        self.frame = frame
        self.residual = residual
        self.kwargs = kwargs
        self.feature_cols = self.AGENT_FRAME_FEATURE_COLS if frame == "agent" else self.FEATURE_COLS
        self.models: Dict[int, Any] = {}

        if model_type == "decision_tree":
            base_name = "Decision Tree"
            params = f"depth={kwargs.get('max_depth', 8)}"
        elif model_type == "random_forest":
            base_name = "Random Forest"
            params = f"trees={kwargs.get('n_estimators', 50)}"
        elif model_type == "knn":
            base_name = "KNN"
            params = f"k={kwargs.get('n_neighbors', 7)}"
        elif model_type == "gradient_boosting":
            base_name = "Gradient Boosting"
            params = f"HistGB, iter={kwargs.get('max_iter', 100)}"
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        if frame == "agent" and residual:
            variant = ""
        else:
            variant = f" [{frame} frame, residual]" if residual else f" [{frame} frame]"
        self.name = f"{base_name}{variant} ({params})"

    def _new_model(self):
        kwargs = self.kwargs
        if self.model_type == "decision_tree":
            return DecisionTreeRegressor(max_depth=kwargs.get("max_depth", 8), random_state=42)
        if self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=kwargs.get("n_estimators", 50),
                max_depth=kwargs.get("max_depth", 10),
                min_samples_split=kwargs.get("min_samples_split", 2),
                min_samples_leaf=kwargs.get("min_samples_leaf", 1),
                max_features=kwargs.get("max_features", 1.0),
                random_state=kwargs.get("random_state", 42),
                n_jobs=-1,
            )
        if self.model_type == "knn":
            return KNeighborsRegressor(n_neighbors=kwargs.get("n_neighbors", 7), n_jobs=-1)
        # gradient_boosting: HistGradientBoosting predicts one output at a time
        base_gb = HistGradientBoostingRegressor(
            max_iter=kwargs.get("max_iter", 100),
            max_depth=kwargs.get("max_depth", 6),
            learning_rate=kwargs.get("learning_rate", 0.1),
            min_samples_leaf=kwargs.get("min_samples_leaf", 20),
            l2_regularization=kwargs.get("l2_regularization", 0.0),
            random_state=kwargs.get("random_state", 42),
        )
        return MultiOutputRegressor(base_gb, n_jobs=-1)

    def _features(self, df: pd.DataFrame) -> np.ndarray:
        if self.frame == "agent":
            df = PandasFeaturePipeline.add_agent_frame_features(df)
        return df[self.feature_cols].to_numpy()

    @staticmethod
    def _rotation(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        heading = df["agent_heading"].to_numpy()
        return np.cos(heading), np.sin(heading)

    def _constant_velocity(self, df: pd.DataFrame, horizon: int) -> np.ndarray:
        """Constant-velocity displacement at the horizon, in the frame the model works in."""
        vx = df["agent_vx"].to_numpy()
        vy = df["agent_vy"].to_numpy()
        if self.frame == "agent":
            cos_h, sin_h = self._rotation(df)
            vx, vy = cos_h * vx + sin_h * vy, -sin_h * vx + cos_h * vy
        return np.column_stack([vx * horizon, vy * horizon])

    def _targets(self, df: pd.DataFrame, horizon: int) -> np.ndarray:
        """What the model learns for one horizon: (N, 2), NaN where the true future is missing."""
        dx = df[f"future_dx_{horizon}s"].to_numpy(dtype=float)
        dy = df[f"future_dy_{horizon}s"].to_numpy(dtype=float)
        if self.frame == "agent":
            cos_h, sin_h = self._rotation(df)
            dx, dy = cos_h * dx + sin_h * dy, -sin_h * dx + cos_h * dy
        y = np.column_stack([dx, dy])
        if self.residual:
            y = y - self._constant_velocity(df, horizon)
        return y

    def fit(self, df_train: pd.DataFrame):
        X = self._features(df_train)
        for horizon in self.HORIZONS:
            y = self._targets(df_train, horizon)
            has_target = ~np.isnan(y).any(axis=1)
            self.models[horizon] = self._new_model().fit(X[has_target], y[has_target])
        return self

    def predict_displacements(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        X = self._features(df)
        out = {}
        for horizon in self.HORIZONS:
            pred = self.models[horizon].predict(X)
            if self.residual:
                pred = pred + self._constant_velocity(df, horizon)
            if self.frame == "agent":
                # Back to the global frame
                cos_h, sin_h = self._rotation(df)
                pred = np.column_stack([
                    cos_h * pred[:, 0] - sin_h * pred[:, 1],
                    sin_h * pred[:, 0] + cos_h * pred[:, 1],
                ])
            out[f"{horizon}s"] = pred
        return out

    def get_feature_importances(self) -> Optional[pd.Series]:
        """Feature importances averaged over the three horizon models (tree models only)."""
        if not self.models or not all(hasattr(m, "feature_importances_") for m in self.models.values()):
            return None
        mean_imp = np.mean([m.feature_importances_ for m in self.models.values()], axis=0)
        return pd.Series(mean_imp, index=self.feature_cols).sort_values(ascending=False)


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

    # The 5s target is missing (NaN) when the vehicle is not observed 5s later:
    # 5s metrics only use the samples where it exists.
    has_5s = ~np.isnan(err_5s)

    fde_1s = float(np.mean(err_1s))
    fde_3s = float(np.mean(err_3s))
    fde_5s = float(np.mean(err_5s[has_5s])) if has_5s.any() else float("nan")

    ade_1s = fde_1s
    ade_3s = float(np.mean((err_1s + err_3s) / 2.0))
    ade_5s = (
        float(np.mean((err_1s[has_5s] + err_3s[has_5s] + err_5s[has_5s]) / 3.0)) if has_5s.any() else float("nan")
    )

    gt_all = np.column_stack([gt_1s, gt_3s, gt_5s])
    preds_all = np.column_stack([preds["1s"], preds["3s"], preds["5s"]])

    # Global regression metrics across all displacement components
    valid_mask = ~np.isnan(gt_all).any(axis=1) & ~np.isnan(preds_all).any(axis=1)
    if valid_mask.any():
        mae_global = float(mean_absolute_error(gt_all[valid_mask], preds_all[valid_mask]))
        rmse_global = float(np.sqrt(mean_squared_error(gt_all[valid_mask], preds_all[valid_mask])))
        r2_global = float(r2_score(gt_all[valid_mask], preds_all[valid_mask]))
    else:
        mae_global, rmse_global, r2_global = float("nan"), float("nan"), float("nan")

    # Horizon-specific metrics at 3.0s (planning horizon target)
    valid_3s = ~np.isnan(gt_3s).any(axis=1) & ~np.isnan(preds["3s"]).any(axis=1)
    if valid_3s.any():
        mae_3s = float(mean_absolute_error(gt_3s[valid_3s], preds["3s"][valid_3s]))
        rmse_3s = float(np.sqrt(mean_squared_error(gt_3s[valid_3s], preds["3s"][valid_3s])))
        r2_3s = float(r2_score(gt_3s[valid_3s], preds["3s"][valid_3s]))
    else:
        mae_3s, rmse_3s, r2_3s = float("nan"), float("nan"), float("nan")

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
    *,
    df_refit: Optional[pd.DataFrame] = None,
) -> Tuple[MLTrajectoryPredictor, Dict[str, Any]]:
    """
    Optimizes 3s displacement MAE using scenario-grouped cross-validation.
    Fits the chosen configuration at every horizon on df_refit, or df_train if omitted.
    Returns: (tuned_predictor, best_parameters_dict)
    """
    sample_predictor = MLTrajectoryPredictor(model_type=model_type, random_state=random_state)
    X = sample_predictor._features(df_train)
    y = sample_predictor._targets(df_train, horizon=3)
    valid = ~np.isnan(y).any(axis=1)
    groups = df_train.loc[valid, "scenario_id"]
    if groups.isna().any() or groups.nunique() < cv:
        raise ValueError(f"Tuning requires at least {cv} distinct scenario IDs with valid 3s targets and no missing IDs.")

    if model_type == "gradient_boosting":
        param_distributions = {
            "estimator__learning_rate": [0.03, 0.05, 0.08, 0.1, 0.15],
            "estimator__max_iter": [50, 80, 100, 150],
            "estimator__max_depth": [4, 5, 6, 8],
            "estimator__min_samples_leaf": [15, 20, 30, 50],
            "estimator__l2_regularization": [0.0, 0.1, 1.0, 5.0],
        }
    elif model_type == "random_forest":
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
        estimator=sample_predictor._new_model(),
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=GroupKFold(n_splits=cv),
        scoring="neg_mean_absolute_error",
        random_state=random_state,
        n_jobs=-1,
        refit=False,
    )
    search.fit(X[valid], y[valid], groups=groups)

    # Clean param dict
    best_params = {key.removeprefix("estimator__"): value for key, value in search.best_params_.items()}

    best_predictor = MLTrajectoryPredictor(
        model_type=model_type,
        random_state=random_state,
        **best_params,
    )
    best_predictor.name = f"Tuned {best_predictor.name}"
    best_predictor.fit(df_train if df_refit is None else df_refit)

    return best_predictor, best_params


def cross_validate_predictors(
    predictor_factories: Dict[str, Any], df: pd.DataFrame, n_splits: int = 5
) -> pd.DataFrame:
    """
    Scenario-grouped K-fold cross-validation: every scenario is used for testing exactly once
    and never appears in train and test at the same time.
    predictor_factories maps a label to a function that builds a fresh predictor.
    Returns the mean and the standard deviation (across folds) of the final error at 1s, 3s, 5s.
    """
    groups = df["scenario_id"].to_numpy()
    rows = []
    for label, factory in predictor_factories.items():
        fold_metrics = []
        for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(df, groups=groups):
            predictor = factory()
            if hasattr(predictor, "fit"):
                predictor.fit(df.iloc[train_idx])
            metrics, _ = evaluate_trajectory_predictor(predictor, df.iloc[test_idx])
            fold_metrics.append([metrics.fde_1s, metrics.fde_3s, metrics.fde_5s])
        fold_metrics = np.array(fold_metrics)
        mean, std = fold_metrics.mean(axis=0), fold_metrics.std(axis=0)
        rows.append({
            "model": label,
            "fde_1s": f"{mean[0]:.2f} ± {std[0]:.2f}",
            "fde_3s": f"{mean[1]:.2f} ± {std[1]:.2f}",
            "fde_5s": f"{mean[2]:.2f} ± {std[2]:.2f}",
        })
    return pd.DataFrame(rows)


class ManeuverClassifier:
    """
    Classifies vehicle intention at horizon t+3.0s into one of:
    - STRAIGHT, TURN_LEFT, TURN_RIGHT, DECELERATING, ACCELERATING.

    The maneuver labels are defined in the vehicle's own frame (lateral / forward displacement),
    so by default the features are too (frame="agent"). frame="global" keeps the first version.
    """

    FEATURE_COLS = PandasFeaturePipeline.FEATURE_COLS
    AGENT_FRAME_FEATURE_COLS = PandasFeaturePipeline.AGENT_FRAME_FEATURE_COLS
    TARGET_COL = PandasFeaturePipeline.TARGET_MANEUVER_COL

    def __init__(self, model_type: str = "random_forest", frame: str = "agent", **kwargs):
        if frame not in ("agent", "global"):
            raise ValueError(f"Unknown frame: {frame}")
        self.model_type = model_type
        self.frame = frame
        self.feature_cols = self.AGENT_FRAME_FEATURE_COLS if frame == "agent" else self.FEATURE_COLS
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
        if frame == "global":
            self.name = self.name.replace(" (", " [global frame] (", 1)

    def _features(self, df: pd.DataFrame) -> np.ndarray:
        if self.frame == "agent":
            df = PandasFeaturePipeline.add_agent_frame_features(df)
        return df[self.feature_cols].to_numpy()

    def fit(self, df_train: pd.DataFrame):
        X = self._features(df_train)
        y = df_train[self.TARGET_COL].to_numpy()
        self.model.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._features(df))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._features(df))


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
