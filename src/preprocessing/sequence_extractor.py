"""
Sequence Extractor for Waymo Open Motion Dataset.
Extracts ego-relative trajectory observation windows (history) and future target displacements
for interactive obstacle vehicles to feed ML trajectory predictors.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class TrajectorySample:
    scenario_id: str
    track_id: int
    object_type: int
    current_time_step: int
    # Ego state at current_time_step
    ego_x: float
    ego_y: float
    ego_speed: float
    ego_heading: float
    # Target vehicle state
    agent_x: float
    agent_y: float
    agent_vx: float
    agent_vy: float
    agent_speed: float
    agent_heading: float
    # Ego-relative features
    rel_x: float
    rel_y: float
    rel_vx: float
    rel_vy: float
    rel_heading: float
    distance_to_ego: float
    # Historical motion features over past 10 steps (1.0 s)
    past_dx_1s: float
    past_dy_1s: float
    past_speed_mean: float
    past_heading_change: float
    # Future displacement ground truth (1s = 10 steps, 3s = 30 steps, 5s = 50 steps)
    future_dx_1s: float
    future_dy_1s: float
    future_dx_3s: float
    future_dy_3s: float
    future_dx_5s: float
    future_dy_5s: float
    # Future trajectory waypoints relative to agent's current position (agent frame)
    future_agent_dx_1s: float
    future_agent_dy_1s: float
    future_agent_dx_3s: float
    future_agent_dy_3s: float
    future_agent_dx_5s: float
    future_agent_dy_5s: float


class SequenceExtractor:
    """
    Extracts structured ML samples from scenario protobuf objects.
    Filters for relevant dynamic obstacle vehicles in the ego vehicle's neighborhood.
    """

    def __init__(
        self,
        history_steps: int = 10,
        future_steps: int = 30,
        max_distance_to_ego: float = 60.0,
        min_agent_speed: float = 0.5,
        step_stride: int = 5,
    ):
        """
        Args:
            history_steps: Number of past steps to observe (10 steps = 1.0s at 10Hz).
            future_steps: Number of future steps to predict (30 steps = 3.0s, 50 steps = 5.0s).
            max_distance_to_ego: Radius in meters around SDC to consider obstacles.
            min_agent_speed: Threshold in m/s to filter out stationary/parked vehicles.
            step_stride: Stride between sampled time steps to avoid excessive temporal redundancy.
        """
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.max_distance_to_ego = max_distance_to_ego
        self.min_agent_speed = min_agent_speed
        self.step_stride = step_stride

    def extract_from_scenario(self, scenario) -> List[TrajectorySample]:
        """
        Parses a single Waymo scenario protobuf into a list of TrajectorySamples.
        """
        samples = []
        sdc_index = scenario.sdc_track_index
        tracks = scenario.tracks
        if sdc_index < 0 or sdc_index >= len(tracks):
            return samples

        sdc_track = tracks[sdc_index]
        sdc_states = list(sdc_track.states)
        num_steps = len(sdc_states)

        # Ensure scenario has enough temporal depth
        min_required_steps = self.history_steps + self.future_steps
        if num_steps < min_required_steps:
            return samples

        # Loop through valid observation timestamps
        for t_curr in range(self.history_steps, num_steps - self.future_steps, self.step_stride):
            sdc_state = sdc_states[t_curr]
            if not sdc_state.valid:
                continue

            ego_x = sdc_state.center_x
            ego_y = sdc_state.center_y
            ego_heading = sdc_state.heading
            ego_vx = sdc_state.velocity_x
            ego_vy = sdc_state.velocity_y
            ego_speed = float(np.hypot(ego_vx, ego_vy))

            # Iterate over all non-SDC tracks (focusing on vehicles: object_type == 1)
            for idx, track in enumerate(tracks):
                if idx == sdc_index:
                    continue
                # object_type 1 = TYPE_VEHICLE
                if track.object_type != 1:
                    continue

                states = list(track.states)
                curr_st = states[t_curr]
                if not curr_st.valid:
                    continue

                # Distance check to SDC
                dx = curr_st.center_x - ego_x
                dy = curr_st.center_y - ego_y
                dist = float(np.hypot(dx, dy))
                if dist > self.max_distance_to_ego or dist < 1.0:
                    continue

                agent_vx = curr_st.velocity_x
                agent_vy = curr_st.velocity_y
                agent_speed = float(np.hypot(agent_vx, agent_vy))
                if agent_speed < self.min_agent_speed:
                    continue

                # Check if history is valid
                past_valid = all(states[t_curr - k].valid for k in range(1, self.history_steps + 1))
                if not past_valid:
                    continue

                # Check future validity (at least at 1s, 3s)
                t_1s = t_curr + 10
                t_3s = t_curr + 30
                t_5s = min(t_curr + 50, num_steps - 1)

                if not (states[t_1s].valid and states[t_3s].valid):
                    continue

                # Compute past motion features
                past_st = states[t_curr - self.history_steps]
                past_dx = curr_st.center_x - past_st.center_x
                past_dy = curr_st.center_y - past_st.center_y
                past_speeds = [
                    np.hypot(states[t_curr - k].velocity_x, states[t_curr - k].velocity_y)
                    for k in range(self.history_steps)
                ]
                past_speed_mean = float(np.mean(past_speeds))
                past_heading_change = float(curr_st.heading - past_st.heading)
                # Normalize angle to [-pi, pi]
                past_heading_change = (past_heading_change + np.pi) % (2 * np.pi) - np.pi

                # Ego-relative transforms
                cos_ego = np.cos(ego_heading)
                sin_ego = np.sin(ego_heading)
                rel_x = cos_ego * dx + sin_ego * dy
                rel_y = -sin_ego * dx + cos_ego * dy

                dvx = agent_vx - ego_vx
                dvy = agent_vy - ego_vy
                rel_vx = cos_ego * dvx + sin_ego * dvy
                rel_vy = -sin_ego * dvx + cos_ego * dvy

                rel_heading = (curr_st.heading - ego_heading + np.pi) % (2 * np.pi) - np.pi

                # Global future displacements
                fut_dx_1s = states[t_1s].center_x - curr_st.center_x
                fut_dy_1s = states[t_1s].center_y - curr_st.center_y
                fut_dx_3s = states[t_3s].center_x - curr_st.center_x
                fut_dy_3s = states[t_3s].center_y - curr_st.center_y

                if states[t_5s].valid:
                    fut_dx_5s = states[t_5s].center_x - curr_st.center_x
                    fut_dy_5s = states[t_5s].center_y - curr_st.center_y
                else:
                    fut_dx_5s = fut_dx_3s * (5.0 / 3.0)
                    fut_dy_5s = fut_dy_3s * (5.0 / 3.0)

                # Agent-relative future displacements (rotated into agent's heading)
                cos_ag = np.cos(curr_st.heading)
                sin_ag = np.sin(curr_st.heading)

                fut_ag_dx_1s = cos_ag * fut_dx_1s + sin_ag * fut_dy_1s
                fut_ag_dy_1s = -sin_ag * fut_dx_1s + cos_ag * fut_dy_1s
                fut_ag_dx_3s = cos_ag * fut_dx_3s + sin_ag * fut_dy_3s
                fut_ag_dy_3s = -sin_ag * fut_dx_3s + cos_ag * fut_dy_3s
                fut_ag_dx_5s = cos_ag * fut_dx_5s + sin_ag * fut_dy_5s
                fut_ag_dy_5s = -sin_ag * fut_dx_5s + cos_ag * fut_dy_5s

                sample = TrajectorySample(
                    scenario_id=scenario.scenario_id,
                    track_id=track.id,
                    object_type=track.object_type,
                    current_time_step=t_curr,
                    ego_x=ego_x,
                    ego_y=ego_y,
                    ego_speed=ego_speed,
                    ego_heading=ego_heading,
                    agent_x=curr_st.center_x,
                    agent_y=curr_st.center_y,
                    agent_vx=agent_vx,
                    agent_vy=agent_vy,
                    agent_speed=agent_speed,
                    agent_heading=curr_st.heading,
                    rel_x=rel_x,
                    rel_y=rel_y,
                    rel_vx=rel_vx,
                    rel_vy=rel_vy,
                    rel_heading=rel_heading,
                    distance_to_ego=dist,
                    past_dx_1s=past_dx,
                    past_dy_1s=past_dy,
                    past_speed_mean=past_speed_mean,
                    past_heading_change=past_heading_change,
                    future_dx_1s=fut_dx_1s,
                    future_dy_1s=fut_dy_1s,
                    future_dx_3s=fut_dx_3s,
                    future_dy_3s=fut_dy_3s,
                    future_dx_5s=fut_dx_5s,
                    future_dy_5s=fut_dy_5s,
                    future_agent_dx_1s=fut_ag_dx_1s,
                    future_agent_dy_1s=fut_ag_dy_1s,
                    future_agent_dx_3s=fut_ag_dx_3s,
                    future_agent_dy_3s=fut_ag_dy_3s,
                    future_agent_dx_5s=fut_ag_dx_5s,
                    future_agent_dy_5s=fut_ag_dy_5s,
                )
                samples.append(sample)

        return samples

    def samples_to_dataframe(self, samples: List[TrajectorySample]) -> pd.DataFrame:
        """
        Converts a list of TrajectorySample dataclasses into a tabular pandas DataFrame.
        """
        if not samples:
            return pd.DataFrame()
        return pd.DataFrame([s.__dict__ for s in samples])
