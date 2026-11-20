"""Parser and extractor for Waymo Open Motion Dataset Scenario protobufs.

Extracts tabular representations, summary statistics, current-step agent states,
and ego vehicle kinematics.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from waymo_open_dataset.protos import scenario_pb2


OBJECT_TYPE_NAMES = {
    scenario_pb2.Track.ObjectType.TYPE_UNSET: "UNSET",
    scenario_pb2.Track.ObjectType.TYPE_VEHICLE: "VEHICLE",
    scenario_pb2.Track.ObjectType.TYPE_PEDESTRIAN: "PEDESTRIAN",
    scenario_pb2.Track.ObjectType.TYPE_CYCLIST: "CYCLIST",
    scenario_pb2.Track.ObjectType.TYPE_OTHER: "OTHER",
}


def calculate_speed(vx: float, vy: float) -> float:
    """Calculate 2D planar speed from velocity components."""
    if math.isnan(vx) or math.isnan(vy):
        return 0.0
    return float(math.hypot(vx, vy))


def parse_scenario_summary(scenario: scenario_pb2.Scenario) -> Dict[str, Any]:
    """Parse top-level metadata and track counts for a single scenario."""
    num_vehicles = 0
    num_pedestrians = 0
    num_cyclists = 0
    num_other = 0

    for track in scenario.tracks:
        if track.object_type == scenario_pb2.Track.ObjectType.TYPE_VEHICLE:
            num_vehicles += 1
        elif track.object_type == scenario_pb2.Track.ObjectType.TYPE_PEDESTRIAN:
            num_pedestrians += 1
        elif track.object_type == scenario_pb2.Track.ObjectType.TYPE_CYCLIST:
            num_cyclists += 1
        else:
            num_other += 1

    t_idx = scenario.current_time_index if scenario.current_time_index is not None else 0
    ego_valid = False
    if 0 <= scenario.sdc_track_index < len(scenario.tracks):
        sdc_track = scenario.tracks[scenario.sdc_track_index]
        if 0 <= t_idx < len(sdc_track.states):
            ego_valid = bool(sdc_track.states[t_idx].valid)

    # Breakdown map feature types
    map_feature_counts: Dict[str, int] = {}
    for mf in scenario.map_features:
        feat_type = mf.WhichOneof("feature_data") or "unknown"
        map_feature_counts[feat_type] = map_feature_counts.get(feat_type, 0) + 1

    return {
        "scenario_id": scenario.scenario_id,
        "timestamps_count": len(scenario.timestamps_seconds),
        "current_time_index": t_idx,
        "num_tracks": len(scenario.tracks),
        "num_vehicles": num_vehicles,
        "num_pedestrians": num_pedestrians,
        "num_cyclists": num_cyclists,
        "num_other": num_other,
        "num_map_features": len(scenario.map_features),
        "num_dynamic_map_states": len(scenario.dynamic_map_states),
        "ego_valid": ego_valid,
        "sdc_track_index": scenario.sdc_track_index,
        "num_lanes": map_feature_counts.get("lane", 0),
        "num_road_edges": map_feature_counts.get("road_edge", 0),
        "num_road_lines": map_feature_counts.get("road_line", 0),
        "num_crosswalks": map_feature_counts.get("crosswalk", 0),
        "num_speed_bumps": map_feature_counts.get("speed_bump", 0),
        "num_stop_signs": map_feature_counts.get("stop_sign", 0),
        "num_driveways": map_feature_counts.get("driveway", 0),
    }


def parse_objects_at_timestamp(
    scenario: scenario_pb2.Scenario,
    time_index: Optional[int] = None,
    only_valid: bool = False,
) -> List[Dict[str, Any]]:
    """Extract object states at a specific timestamp (default: current_time_index)."""
    if time_index is None:
        time_index = scenario.current_time_index if scenario.current_time_index is not None else 0

    objects = []
    for idx, track in enumerate(scenario.tracks):
        is_ego = (idx == scenario.sdc_track_index)
        obj_type_str = OBJECT_TYPE_NAMES.get(track.object_type, "OTHER")

        if time_index >= len(track.states):
            continue

        state = track.states[time_index]
        if only_valid and not state.valid:
            continue

        vx = float(state.velocity_x) if state.valid else 0.0
        vy = float(state.velocity_y) if state.valid else 0.0
        spd = calculate_speed(vx, vy) if state.valid else 0.0

        objects.append({
            "scenario_id": scenario.scenario_id,
            "object_id": track.id,
            "object_type": obj_type_str,
            "is_ego": is_ego,
            "x": float(state.center_x) if state.valid else np.nan,
            "y": float(state.center_y) if state.valid else np.nan,
            "z": float(state.center_z) if state.valid else np.nan,
            "velocity_x": vx if state.valid else np.nan,
            "velocity_y": vy if state.valid else np.nan,
            "speed": spd if state.valid else np.nan,
            "heading": float(state.heading) if state.valid else np.nan,
            "length": float(state.length) if state.valid else np.nan,
            "width": float(state.width) if state.valid else np.nan,
            "height": float(state.height) if state.valid else np.nan,
            "valid": bool(state.valid),
        })

    return objects


def parse_ego_at_timestamp(
    scenario: scenario_pb2.Scenario,
    time_index: Optional[int] = None,
    nearby_radius: float = 50.0,
) -> Dict[str, Any]:
    """Extract ego kinematics and surrounding agent counts within radius at timestamp."""
    if time_index is None:
        time_index = scenario.current_time_index if scenario.current_time_index is not None else 0

    ego_data = {
        "scenario_id": scenario.scenario_id,
        "ego_x": np.nan,
        "ego_y": np.nan,
        "ego_speed": np.nan,
        "ego_velocity_x": np.nan,
        "ego_velocity_y": np.nan,
        "ego_heading": np.nan,
        "nearby_vehicle_count": 0,
        "nearby_pedestrian_count": 0,
        "nearby_cyclist_count": 0,
    }

    if not (0 <= scenario.sdc_track_index < len(scenario.tracks)):
        return ego_data

    sdc_track = scenario.tracks[scenario.sdc_track_index]
    if time_index >= len(sdc_track.states):
        return ego_data

    ego_state = sdc_track.states[time_index]
    if not ego_state.valid:
        return ego_data

    ex = float(ego_state.center_x)
    ey = float(ego_state.center_y)
    evx = float(ego_state.velocity_x)
    evy = float(ego_state.velocity_y)
    e_spd = calculate_speed(evx, evy)

    ego_data.update({
        "ego_x": ex,
        "ego_y": ey,
        "ego_speed": e_spd,
        "ego_velocity_x": evx,
        "ego_velocity_y": evy,
        "ego_heading": float(ego_state.heading),
    })

    # Count nearby valid objects by type
    nearby_veh = 0
    nearby_ped = 0
    nearby_cyc = 0

    for idx, track in enumerate(scenario.tracks):
        if idx == scenario.sdc_track_index:
            continue
        if time_index >= len(track.states):
            continue
        st = track.states[time_index]
        if not st.valid:
            continue

        dist = math.hypot(st.center_x - ex, st.center_y - ey)
        if dist <= nearby_radius:
            if track.object_type == scenario_pb2.Track.ObjectType.TYPE_VEHICLE:
                nearby_veh += 1
            elif track.object_type == scenario_pb2.Track.ObjectType.TYPE_PEDESTRIAN:
                nearby_ped += 1
            elif track.object_type == scenario_pb2.Track.ObjectType.TYPE_CYCLIST:
                nearby_cyc += 1

    ego_data["nearby_vehicle_count"] = nearby_veh
    ego_data["nearby_pedestrian_count"] = nearby_ped
    ego_data["nearby_cyclist_count"] = nearby_cyc

    return ego_data


class ScenarioParser:
    """Batch parser transforming scenarios into Pandas DataFrames."""

    def __init__(self, nearby_radius: float = 50.0):
        self.nearby_radius = nearby_radius

    def parse_scenarios(
        self,
        scenarios: List[scenario_pb2.Scenario],
        time_index: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Convert a list of scenarios into 3 DataFrames:

        1. scenario_summary_df
        2. objects_current_df
        3. ego_current_df
        """
        summary_rows = []
        objects_rows = []
        ego_rows = []

        for sc in scenarios:
            summary_rows.append(parse_scenario_summary(sc))
            objects_rows.extend(parse_objects_at_timestamp(sc, time_index=time_index))
            ego_rows.append(
                parse_ego_at_timestamp(sc, time_index=time_index, nearby_radius=self.nearby_radius)
            )

        summary_df = pd.DataFrame(summary_rows)
        objects_df = pd.DataFrame(objects_rows)
        ego_df = pd.DataFrame(ego_rows)

        return summary_df, objects_df, ego_df
