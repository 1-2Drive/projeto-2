"""
Vectorized Preprocessing and Feature Engineering using NumPy and Pandas.
Applies vectorized mathematical transformations, quality filters, maneuver labeling,
and scaling for autonomous driving motion modeling.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


class PandasFeaturePipeline:
    """
    Production Pandas & NumPy feature processing pipeline.
    Performs fully vectorized transformations on trajectory sequences.
    """

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
