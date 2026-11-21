"""
Spatiotemporal A* Trajectory and Speed Planner for Autonomous Driving.
Navigates the ego vehicle along the lane centerline while avoiding dynamic obstacles
under different prediction modalities:
- Uninformed (Reactive, assumes stationary obstacles)
- Constant Velocity (Physics baseline)
- ML-Predicted (Predict-to-Plan via Decision Tree / Random Forest)
- Oracle (True Ground Truth future obstacles)
"""

from dataclasses import dataclass, field
import heapq
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ObstacleState:
    track_id: int
    x: float
    y: float
    vx: float
    vy: float
    length: float = 4.8
    width: float = 2.0


@dataclass
class PlanResult:
    mode: str
    success: bool
    collision_occurred: bool
    min_distance_to_obstacle: float
    emergency_braking_events: int
    total_travel_distance: float
    average_speed: float
    trajectory_s: List[float] = field(default_factory=list)
    trajectory_v: List[float] = field(default_factory=list)
    trajectory_a: List[float] = field(default_factory=list)


class AStarTrajectoryPlanner:
    """
    Search-based spatiotemporal speed planner along the ego trajectory centerline.
    Evaluates safety against predicted dynamic obstacle positions.
    """

    def __init__(
        self,
        horizon_steps: int = 30,  # 3.0 seconds at 10Hz (dt = 0.1s)
        dt: float = 0.1,
        max_speed: float = 20.0,  # 72 km/h
        accel_options: Optional[List[float]] = None,
        safety_radius: float = 3.5,  # meters center-to-center
        w_progress: float = 2.0,
        w_comfort: float = 0.5,
        w_collision: float = 1000.0,
        max_expansions: int = 3000,
    ):
        self.horizon_steps = horizon_steps
        self.dt = dt
        self.max_speed = max_speed
        self.accel_options = accel_options or [-3.0, -1.5, 0.0, 1.5]
        self.safety_radius = safety_radius
        self.w_progress = w_progress
        self.w_comfort = w_comfort
        self.w_collision = w_collision
        self.max_expansions = max_expansions

    def plan_trajectory(
        self,
        ego_init_v: float,
        ego_path_xy: np.ndarray,  # shape (M, 2)
        predicted_obstacles: Dict[int, List[Tuple[float, float]]],  # track_id -> [(x, y) for step 0..T]
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Plans ego (s, v) profile along ego_path_xy to avoid predicted obstacle footprints.
        Returns: (s_profile, v_profile, a_profile)
        """
        # Arc length along centerline path
        dists = np.linalg.norm(np.diff(ego_path_xy, axis=0), axis=1)
        s_coords = np.insert(np.cumsum(dists), 0, 0.0)
        total_path_length = s_coords[-1] if len(s_coords) > 0 else 50.0

        def s_to_xy(s_val: float) -> Tuple[float, float]:
            if s_val <= 0:
                return float(ego_path_xy[0, 0]), float(ego_path_xy[0, 1])
            if s_val >= total_path_length:
                return float(ego_path_xy[-1, 0]), float(ego_path_xy[-1, 1])
            idx = np.searchsorted(s_coords, s_val)
            idx = max(1, min(idx, len(s_coords) - 1))
            ratio = (s_val - s_coords[idx - 1]) / max(1e-4, s_coords[idx] - s_coords[idx - 1])
            pt = ego_path_xy[idx - 1] + ratio * (ego_path_xy[idx] - ego_path_xy[idx - 1])
            return float(pt[0]), float(pt[1])

        # Priority queue for A*: (f_score, step, s, v, cost_so_far, parent_idx, accel)
        start_state = (0.0, 0, 0.0, ego_init_v, 0.0, -1, 0.0)
        open_set = [start_state]
        visited = {}
        nodes_history = []

        best_terminal = None
        best_cost = float("inf")

        while open_set and len(nodes_history) < self.max_expansions:
            f, step, s, v, g, parent, a = heapq.heappop(open_set)

            # Discretize state key to 0.5m and 0.5m/s resolution
            state_key = (step, round(s * 2) / 2, round(v * 2) / 2)
            if state_key in visited and visited[state_key] <= g:
                continue
            visited[state_key] = g

            node_idx = len(nodes_history)
            nodes_history.append((step, s, v, g, parent, a))

            if step == self.horizon_steps:
                if g < best_cost:
                    best_cost = g
                    best_terminal = node_idx
                continue

            for accel in self.accel_options:
                v_next = float(np.clip(v + accel * self.dt, 0.0, self.max_speed))
                s_next = float(s + 0.5 * (v + v_next) * self.dt)

                # Check collision with predicted obstacles at step + 1
                ego_x, ego_y = s_to_xy(s_next)
                collision = False
                for track_id, obs_traj in predicted_obstacles.items():
                    step_idx = min(step + 1, len(obs_traj) - 1)
                    ox, oy = obs_traj[step_idx]
                    dist_to_obs = float(np.hypot(ego_x - ox, ego_y - oy))
                    if dist_to_obs < self.safety_radius:
                        collision = True
                        break

                step_cost = self.w_comfort * (accel**2) * self.dt
                if collision:
                    step_cost += self.w_collision

                g_next = g + step_cost
                remaining_steps = self.horizon_steps - (step + 1)
                h_next = -self.w_progress * (s_next + v_next * remaining_steps * self.dt)
                f_next = g_next + h_next

                heapq.heappush(open_set, (f_next, step + 1, s_next, v_next, g_next, node_idx, accel))

        if best_terminal is None and nodes_history:
            best_terminal = max(range(len(nodes_history)), key=lambda i: nodes_history[i][1])

        curr_idx = best_terminal
        s_traj, v_traj, a_traj = [], [], []
        while curr_idx >= 0:
            st, s, v, g, p, a = nodes_history[curr_idx]
            s_traj.append(s)
            v_traj.append(v)
            a_traj.append(a)
            curr_idx = p

        s_traj.reverse()
        v_traj.reverse()
        a_traj.reverse()
        return s_traj, v_traj, a_traj

    def evaluate_plan_against_ground_truth(
        self,
        mode: str,
        ego_path_xy: np.ndarray,
        s_traj: List[float],
        v_traj: List[float],
        a_traj: List[float],
        gt_obstacles: Dict[int, List[Tuple[float, float]]],
    ) -> PlanResult:
        """
        Evaluates the executed plan against true ground truth obstacles.
        Computes safety, minimum TTC/distance, and emergency braking occurrences.
        """
        dists = np.linalg.norm(np.diff(ego_path_xy, axis=0), axis=1)
        s_coords = np.insert(np.cumsum(dists), 0, 0.0)
        total_length = s_coords[-1] if len(s_coords) > 0 else 50.0

        def s_to_xy(s_val: float) -> Tuple[float, float]:
            if s_val <= 0:
                return float(ego_path_xy[0, 0]), float(ego_path_xy[0, 1])
            if s_val >= total_length:
                return float(ego_path_xy[-1, 0]), float(ego_path_xy[-1, 1])
            idx = np.searchsorted(s_coords, s_val)
            idx = max(1, min(idx, len(s_coords) - 1))
            ratio = (s_val - s_coords[idx - 1]) / max(1e-4, s_coords[idx] - s_coords[idx - 1])
            pt = ego_path_xy[idx - 1] + ratio * (ego_path_xy[idx] - ego_path_xy[idx - 1])
            return float(pt[0]), float(pt[1])

        collision = False
        min_dist = float("inf")
        emergency_brakes = 0

        for t_idx, (s, a) in enumerate(zip(s_traj, a_traj)):
            ego_x, ego_y = s_to_xy(s)
            for track_id, gt_traj in gt_obstacles.items():
                step_idx = min(t_idx, len(gt_traj) - 1)
                ox, oy = gt_traj[step_idx]
                d = float(np.hypot(ego_x - ox, ego_y - oy))
                if d < min_dist:
                    min_dist = d
                if d < self.safety_radius:
                    collision = True

            if a <= -3.0:
                emergency_brakes += 1

        total_distance = s_traj[-1] - s_traj[0] if len(s_traj) > 1 else 0.0
        avg_speed = float(np.mean(v_traj)) if v_traj else 0.0

        return PlanResult(
            mode=mode,
            success=not collision,
            collision_occurred=collision,
            min_distance_to_obstacle=round(min_dist, 2),
            emergency_braking_events=emergency_brakes,
            total_travel_distance=round(total_distance, 2),
            average_speed=round(avg_speed, 2),
            trajectory_s=s_traj,
            trajectory_v=v_traj,
            trajectory_a=a_traj,
        )
