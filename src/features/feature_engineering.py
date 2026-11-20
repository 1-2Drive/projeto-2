"""Feature engineering definitions and geometric helpers for Phase 2 tactical decision modeling.

Note: In Phase 1, this module defines the mathematical definitions, geometric transformations,
and feature signatures planned for Phase 2. Final target labels and trained models will be
implemented in Phase 2.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


def transform_to_ego_frame(
    x: float,
    y: float,
    ego_x: float,
    ego_y: float,
    ego_heading: float,
) -> Tuple[float, float]:
    """Transform global 2D coordinates into an ego-centric coordinate frame.

    In the ego frame:
    - Longitudinally (+X_ego): Ahead along the vehicle heading.
    - Laterally (+Y_ego): To the left perpendicular to heading.
    """
    dx = x - ego_x
    dy = y - ego_y
    cos_h = math.cos(-ego_heading)
    sin_h = math.sin(-ego_heading)

    x_ego = dx * cos_h - dy * sin_h
    y_ego = dx * sin_h + dy * cos_h
    return float(x_ego), float(y_ego)


def compute_ttc(distance: float, relative_speed: float) -> float:
    """Calculate Time-To-Collision (TTC) in seconds.

    Args:
        distance: Longitudinal distance to lead vehicle in meters (must be > 0).
        relative_speed: v_ego - v_lead in m/s (closing speed).

    Returns:
        TTC in seconds if closing (relative_speed > 0), else inf.
    """
    if distance <= 0:
        return 0.0
    if relative_speed <= 0.001:
        return float("inf")
    return float(distance / relative_speed)


def get_candidate_feature_names() -> List[str]:
    """Return the planned list of engineered feature names for Phase 2."""
    return [
        # Ego Kinematics
        "ego_speed",
        "ego_acceleration",
        "ego_heading_change",
        # Front interactions
        "front_vehicle_distance",
        "front_relative_speed",
        "front_ttc",
        # Surrounding agent distances
        "left_front_distance",
        "left_rear_distance",
        "right_front_distance",
        "right_rear_distance",
        "nearest_pedestrian_distance",
        "nearest_cyclist_distance",
        "nearby_vehicle_count",
        # Map & traffic control context
        "traffic_light_state",
        "distance_to_crosswalk",
        "distance_to_stop_sign",
        "distance_to_intersection",
    ]


def get_planned_target_classes() -> Dict[str, List[str]]:
    """Return planned classification target categories."""
    return {
        "lateral": ["KEEP_LANE", "CHANGE_LEFT", "CHANGE_RIGHT", "TURN_LEFT", "TURN_RIGHT"],
        "longitudinal": ["ACCELERATE", "MAINTAIN", "DECELERATE", "STOP"],
    }
