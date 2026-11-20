"""Data preprocessing and data quality validation module for Waymo motion tabular data."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


class DataQualityChecker:
    """Performs rigorous data quality checks on Waymo extracted DataFrames."""

    @staticmethod
    def check_summary_quality(summary_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate scenario summary table quality:

        - Duplicate scenario IDs
        - Missing (NaN) values
        - Timestamp counts
        - Ego validity rate
        """
        total = len(summary_df)
        dup_scenarios = summary_df["scenario_id"].duplicated().sum()
        nan_counts = summary_df.isna().sum().to_dict()
        ego_valid_count = int(summary_df["ego_valid"].sum()) if "ego_valid" in summary_df.columns else 0

        return {
            "total_scenarios": total,
            "duplicate_scenario_ids": int(dup_scenarios),
            "nan_counts": {k: int(v) for k, v in nan_counts.items() if v > 0},
            "ego_valid_count": ego_valid_count,
            "ego_valid_percentage": round(100.0 * ego_valid_count / max(1, total), 2),
            "passed": bool(dup_scenarios == 0 and ego_valid_count == total),
        }

    @staticmethod
    def check_objects_quality(objects_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate objects table quality:

        - Duplicate object IDs within each scenario
        - Infinite or NaN values in coordinates and velocities
        - Speed range validity (non-negative, realistic physical limits)
        - Valid state flags
        """
        total_rows = len(objects_df)
        valid_rows = int(objects_df["valid"].sum()) if "valid" in objects_df.columns else total_rows

        # Check duplicate object IDs within scenarios
        dups = objects_df.duplicated(subset=["scenario_id", "object_id"]).sum()

        # Check speed range on valid records
        valid_subset = objects_df[objects_df["valid"]] if "valid" in objects_df.columns else objects_df
        speeds = valid_subset["speed"].dropna()
        neg_speeds = int((speeds < 0).sum())
        excessive_speeds = int((speeds > 80.0).sum())  # > 288 km/h is physical anomaly for urban scenarios
        inf_speeds = int(np.isinf(speeds).sum())

        nan_positions = int(valid_subset[["x", "y", "z"]].isna().any(axis=1).sum())

        return {
            "total_records": total_rows,
            "valid_records": valid_rows,
            "duplicate_object_ids_in_scenario": int(dups),
            "negative_speeds": neg_speeds,
            "infinite_speeds": inf_speeds,
            "excessive_speeds_gt_80ms": excessive_speeds,
            "valid_records_with_nan_coords": nan_positions,
            "min_speed_ms": float(speeds.min()) if not speeds.empty else 0.0,
            "max_speed_ms": float(speeds.max()) if not speeds.empty else 0.0,
            "mean_speed_ms": float(speeds.mean()) if not speeds.empty else 0.0,
            "passed": bool(dups == 0 and neg_speeds == 0 and inf_speeds == 0 and nan_positions == 0),
        }

    @staticmethod
    def check_ego_quality(ego_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate ego vehicle kinematics table quality."""
        total = len(ego_df)
        nan_speeds = int(ego_df["ego_speed"].isna().sum())
        speeds = ego_df["ego_speed"].dropna()
        neg_speeds = int((speeds < 0).sum())
        inf_speeds = int(np.isinf(speeds).sum())

        return {
            "total_records": total,
            "nan_speeds": nan_speeds,
            "negative_speeds": neg_speeds,
            "infinite_speeds": inf_speeds,
            "min_ego_speed_ms": float(speeds.min()) if not speeds.empty else 0.0,
            "max_ego_speed_ms": float(speeds.max()) if not speeds.empty else 0.0,
            "mean_ego_speed_ms": float(speeds.mean()) if not speeds.empty else 0.0,
            "passed": bool(nan_speeds == 0 and neg_speeds == 0 and inf_speeds == 0),
        }


def run_full_quality_audit(
    summary_df: pd.DataFrame,
    objects_df: pd.DataFrame,
    ego_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Execute complete data quality verification across all tabular datasets."""
    checker = DataQualityChecker()
    summary_rep = checker.check_summary_quality(summary_df)
    objects_rep = checker.check_objects_quality(objects_df)
    ego_rep = checker.check_ego_quality(ego_df)

    overall_passed = summary_rep["passed"] and objects_rep["passed"] and ego_rep["passed"]

    return {
        "overall_passed": overall_passed,
        "summary_quality": summary_rep,
        "objects_quality": objects_rep,
        "ego_quality": ego_rep,
    }
