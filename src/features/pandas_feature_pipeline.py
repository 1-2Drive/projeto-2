"""
Vectorized Preprocessing and Feature Engineering using NumPy and Pandas.
Applies vectorized mathematical transformations, quality filters, maneuver labeling,
and scaling for autonomous driving motion modeling.
"""

from __future__ import annotations

import warnings

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


# Gap used when there is no vehicle ahead (same value as SequenceExtractor.LEAD_NONE_GAP)
LEAD_NONE_GAP = 100.0


class PandasFeaturePipeline:
    """
    Production Pandas & NumPy feature processing pipeline.
    Performs fully vectorized transformations on trajectory sequences.
    """

    METADATA_COLS = [
        "scenario_id",
        "track_id",
        "object_type",
        "current_time_step",
    ]

    # Global coordinates: deliberately EXCLUDED from ML features to guarantee translation/rotation invariance
    GLOBAL_COORD_COLS = [
        "ego_x",
        "ego_y",
        "ego_heading",
        "agent_x",
        "agent_y",
        "agent_heading",
    ]

    # Input Features (X): strictly ego-relative kinematics and local dynamic features
    FEATURE_COLS = [
        "agent_speed",
        "agent_vx",
        "agent_vy",
        "rel_x",
        "rel_y",
        "rel_vx",
        "rel_vy",
        "rel_heading",
        "distance_to_ego",
        "closing_speed",
        "ttc",
        "azimuth_angle",
        "longitudinal_gap",
        "lateral_gap",
        "past_dx_1s",
        "past_dy_1s",
        "past_speed_mean",
        "past_heading_change",
        "yaw_rate_1s",
        "agent_accel_est",
        "ego_speed",
    ]

    # Input features in the AGENT's own frame (x = forward, y = left of the vehicle).
    # Two vehicles doing the same thing look the same to the model whatever their compass
    # direction, which the global-frame columns above (agent_vx, agent_vy, past_dx_1s, ...) do not allow.
    # Exact duplicates in FEATURE_COLS are dropped: longitudinal_gap (= rel_x), lateral_gap (= rel_y)
    # and past_heading_change (= yaw_rate_1s).
    AGENT_FRAME_FEATURE_COLS = [
        "agent_speed",
        "v_lon",
        "v_lat",
        "agent_accel_est",
        "yaw_rate_1s",
        "past_lon_1s",
        "past_lat_1s",
        "past_speed_mean",
        "lead_gap",
        "lead_speed",
        "lead_closing_speed",
        "rel_x",
        "rel_y",
        "rel_vx",
        "rel_vy",
        "rel_heading",
        "distance_to_ego",
        "closing_speed",
        "ttc",
        "azimuth_angle",
        "ego_speed",
    ]

    TARGET_REGRESSION_COLS = [
        "future_dx_1s",
        "future_dy_1s",
        "future_dx_3s",
        "future_dy_3s",
        "future_dx_5s",
        "future_dy_5s",
    ]

    TARGET_MANEUVER_COL = "maneuver_class"

    @classmethod
    def get_feature_target_split(
        cls, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
        """
        Splits a processed DataFrame into:
        - X (features only, no IDs or global coordinates)
        - y_reg (regression displacement targets)
        - y_cls (maneuver classification target)
        - groups (scenario_id series for group-aware CV)
        - metadata (scenario_id, track_id, timesteps for traceability)
        """
        X = df[cls.FEATURE_COLS].copy()
        y_reg = df[cls.TARGET_REGRESSION_COLS].copy()
        y_cls = df[cls.TARGET_MANEUVER_COL].copy() if cls.TARGET_MANEUVER_COL in df.columns else None
        groups = df["scenario_id"].copy()
        metadata = df[cls.METADATA_COLS].copy()
        return X, y_reg, y_cls, groups, metadata

    @classmethod
    def create_scenario_based_split(
        cls, df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scenario-level split (Group-aware) to strictly prevent data leakage across train and test sets.
        Ensures all time-steps from a given scenario belong exclusively to either train or test.
        """
        unique_scenarios = pd.Series(df["scenario_id"].unique())
        test_scenarios = set(unique_scenarios.sample(frac=test_size, random_state=random_state))

        train_mask = ~df["scenario_id"].isin(test_scenarios)
        test_mask = df["scenario_id"].isin(test_scenarios)

        df_train = df[train_mask].reset_index(drop=True)
        df_test = df[test_mask].reset_index(drop=True)
        return df_train, df_test

    @staticmethod
    def clean_and_filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters noise, physically impossible measurements, and stationary artifacts
        using vectorized pandas operations.
        """
        df_clean = df.copy()

        # 1. Drop any rows with NaN or Inf in crucial columns
        required_cols = [
            "agent_vx", "agent_vy", "agent_speed", "ego_speed",
            "rel_x", "rel_y", "distance_to_ego",
            "future_dx_1s", "future_dy_1s", "future_dx_3s", "future_dy_3s"
        ]
        df_clean = df_clean.dropna(subset=required_cols)
        df_clean = df_clean[~np.isinf(df_clean[required_cols]).any(axis=1)]

        # 2. Vectorized physical bounds (speeds up to 50 m/s = 180 km/h, distance < 80m)
        valid_mask = (
            (df_clean["agent_speed"] >= 0.2) &
            (df_clean["agent_speed"] <= 50.0) &
            (df_clean["distance_to_ego"] > 0.5) &
            (df_clean["distance_to_ego"] <= 75.0) &
            (df_clean["future_dx_3s"].abs() <= 60.0) &
            (df_clean["future_dy_3s"].abs() <= 60.0)
        )
        return df_clean[valid_mask].reset_index(drop=True)

    @staticmethod
    def add_vectorized_kinematic_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes interaction and dynamic features using vectorized NumPy arrays:
        - Time-To-Collision (TTC)
        - Closing speed (approach rate)
        - Bearing angle (relative azimuth)
        - Lateral and longitudinal accelerations
        - Curvature proxy (angular yaw rate)
        """
        df_feat = df.copy()

        # Closing speed: rate of distance reduction along line of sight
        # rel_x, rel_y are agent position relative to ego
        rel_pos = df_feat[["rel_x", "rel_y"]].to_numpy()
        rel_vel = df_feat[["rel_vx", "rel_vy"]].to_numpy()
        dists = df_feat["distance_to_ego"].to_numpy()

        # Dot product: rel_pos . rel_vel / dist
        closing_rates = -np.sum(rel_pos * rel_vel, axis=1) / np.maximum(dists, 1e-4)
        df_feat["closing_speed"] = closing_rates

        # Vectorized Time-To-Collision (TTC): dist / closing_rate where closing_rate > 0
        ttc = np.where(closing_rates > 0.1, dists / closing_rates, 99.0)
        df_feat["ttc"] = np.clip(ttc, 0.0, 99.0)

        # Vectorized Azimuth / Bearing Angle: arctan2(rel_y, rel_x)
        df_feat["azimuth_angle"] = np.arctan2(df_feat["rel_y"], df_feat["rel_x"])

        # Longitudinal and Lateral relative velocities in ego frame
        # If ego heading is theta_ego, rel_vx, rel_vy are already aligned
        df_feat["longitudinal_gap"] = df_feat["rel_x"]  # ahead (+) or behind (-)
        df_feat["lateral_gap"] = df_feat["rel_y"]       # left (+) or right (-)

        # Curvature proxy from past heading change (yaw rate in rad/s over 1s)
        df_feat["yaw_rate_1s"] = df_feat["past_heading_change"] / 1.0

        # Estimated longitudinal acceleration over past 1s: (v_curr - v_past)
        df_feat["agent_accel_est"] = df_feat["agent_speed"] - df_feat["past_speed_mean"]

        return df_feat

    @staticmethod
    def add_agent_frame_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Rotates the agent's velocity and past displacement from the global frame (east/north)
        into the agent's own frame (forward/left), using its heading at the current instant.
        """
        df_feat = df.copy()
        cos_h = np.cos(df_feat["agent_heading"].to_numpy())
        sin_h = np.sin(df_feat["agent_heading"].to_numpy())
        vx = df_feat["agent_vx"].to_numpy()
        vy = df_feat["agent_vy"].to_numpy()
        px = df_feat["past_dx_1s"].to_numpy()
        py = df_feat["past_dy_1s"].to_numpy()

        df_feat["v_lon"] = cos_h * vx + sin_h * vy
        df_feat["v_lat"] = -sin_h * vx + cos_h * vy
        df_feat["past_lon_1s"] = cos_h * px + sin_h * py
        df_feat["past_lat_1s"] = -sin_h * px + cos_h * py

        # Tables extracted before the lead-vehicle features existed do not have these columns.
        # Fill them with the same values the extractor uses when no vehicle is ahead, so that
        # older CSV files keep working (the models just cannot use the lead vehicle).
        lead_cols = ["lead_gap", "lead_speed", "lead_closing_speed"]
        if any(col not in df_feat.columns for col in lead_cols):
            warnings.warn(
                "Lead-vehicle columns are missing: assuming no vehicle ahead. "
                "Extract the samples again to use the lead-vehicle features.",
                stacklevel=2,
            )
            if "lead_gap" not in df_feat.columns:
                df_feat["lead_gap"] = LEAD_NONE_GAP
            if "lead_speed" not in df_feat.columns:
                df_feat["lead_speed"] = df_feat["v_lon"]
            if "lead_closing_speed" not in df_feat.columns:
                df_feat["lead_closing_speed"] = 0.0
        return df_feat

    @staticmethod
    def assign_maneuver_labels(df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized classification of dynamic vehicle maneuvers at horizon t+3.0s:
        - 0: Straight (Cruising)
        - 1: Turn Left
        - 2: Turn Right
        - 3: Decelerating / Braking
        - 4: Accelerating
        """
        df_labeled = df.copy()

        # Displacement in agent's own local reference frame
        # future_agent_dx_3s (forward), future_agent_dy_3s (lateral left + / right -)
        lat_disp = df_labeled["future_agent_dy_3s"].to_numpy()
        fwd_disp = df_labeled["future_agent_dx_3s"].to_numpy()
        exp_disp = df_labeled["agent_speed"].to_numpy() * 3.0

        disp_ratio = fwd_disp / np.maximum(exp_disp, 1.0)

        conditions = [
            (lat_disp > 2.0),                      # Turn Left
            (lat_disp < -2.0),                     # Turn Right
            (disp_ratio < 0.65),                   # Decelerating / Stopping
            (disp_ratio > 1.35),                   # Accelerating
        ]
        choices = [
            "TURN_LEFT",
            "TURN_RIGHT",
            "DECELERATING",
            "ACCELERATING",
        ]

        df_labeled["maneuver_class"] = np.select(conditions, choices, default="STRAIGHT")
        return df_labeled

    @staticmethod
    def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates descriptive distribution statistics via pandas.agg.
        """
        metric_cols = [
            "agent_speed", "ego_speed", "distance_to_ego", "closing_speed",
            "ttc", "future_dx_3s", "future_dy_3s"
        ]
        stats = df[metric_cols].agg([
            "count", "mean", "std", "min",
            lambda s: s.quantile(0.25),
            "median",
            lambda s: s.quantile(0.75),
            "max"
        ]).rename(index={"<lambda_0>": "p25", "<lambda_1>": "p75"})
        return stats.round(3)
