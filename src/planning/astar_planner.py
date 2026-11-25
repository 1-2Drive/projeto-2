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
import time
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
    obb_collision_occurred: bool = False
    initial_overlap_detected: bool = False
    trajectory_s: List[float] = field(default_factory=list)
    trajectory_v: List[float] = field(default_factory=list)
    trajectory_a: List[float] = field(default_factory=list)
    trajectory_d: List[float] = field(default_factory=list)
    expansions: int = 0
    planning_time_ms: float = 0.0
    max_lateral_displacement: float = 0.0


def check_obb_collision(
    ax: float,
    ay: float,
    a_head: float,
    a_l: float,
    a_w: float,
    bx: float,
    by: float,
    b_head: float,
    b_l: float,
    b_w: float,
    safety_margin: float = 0.1,
) -> bool:
    """
    Checks collision between two 2D oriented bounding boxes using the Separating Axis Theorem (SAT).
    a_l, a_w: length and width of box A (meters).
    b_l, b_w: length and width of box B (meters).
    safety_margin: expansion buffer added to half-dimensions.
    Returns True if boxes overlap (collision), False if a separating axis exists.
    """
    hla = a_l * 0.5 + safety_margin
    hwa = a_w * 0.5 + safety_margin
    hlb = b_l * 0.5 + safety_margin
    hwb = b_w * 0.5 + safety_margin

    ca, sa = np.cos(a_head), np.sin(a_head)
    cb, sb = np.cos(b_head), np.sin(b_head)

    axes = ((ca, sa), (-sa, ca), (cb, sb), (-sb, cb))
    dx = bx - ax
    dy = by - ay

    for ux, uy in axes:
        dist = abs(dx * ux + dy * uy)
        ra = hla * abs(ca * ux + sa * uy) + hwa * abs(-sa * ux + ca * uy)
        rb = hlb * abs(cb * ux + sb * uy) + hwb * abs(-sb * ux + cb * uy)
        if dist > (ra + rb):
            return False
    return True


def compute_wilson_score_interval(
    successes: int, total: int, confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Computes the Wilson score confidence interval for a binomial proportion.
    Returns: (rate_pct, lower_bound_pct, upper_bound_pct).
    """
    if total == 0:
        return 0.0, 0.0, 0.0
    p = successes / total
    z = 1.959964  # 95% two-tailed confidence z-score
    denominator = 1.0 + (z**2) / total
    centre_adjusted_probability = p + (z**2) / (2.0 * total)
    adjusted_std_dev = np.sqrt(max(0.0, (p * (1.0 - p) + (z**2) / (4.0 * total)) / total))
    lower = max(0.0, (centre_adjusted_probability - z * adjusted_std_dev) / denominator)
    upper = min(1.0, (centre_adjusted_probability + z * adjusted_std_dev) / denominator)
    return p * 100.0, lower * 100.0, upper * 100.0


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
        self.last_expansions = 0
        self.last_planning_time_ms = 0.0

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
        t0 = time.perf_counter()
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

        self.last_expansions = len(nodes_history)
        self.last_planning_time_ms = (time.perf_counter() - t0) * 1000.0

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

    def plan_trajectory_2d(
        self,
        ego_init_v: float,
        ego_path_xy: np.ndarray,
        predicted_obstacles: Dict[int, List[Tuple[float, float]]],
        ego_init_d: float = 0.0,
        lat_options: Optional[List[float]] = None,
        max_d: float = 3.5,
        w_lateral: float = 0.3,
        w_lane: float = 0.02,
        gt_headings: Optional[Dict[int, float]] = None,
        use_obb: bool = True,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        2D Spatiotemporal Frenet A* Trajectory Planner (s, d, v, t).
        Optimizes both longitudinal acceleration and lateral evasion maneuvers
        around dynamic obstacles.
        Returns: (s_profile, v_profile, a_profile, d_profile)
        """
        t0 = time.perf_counter()
        lat_options = lat_options or [-1.5, 0.0, 1.5]

        # Calculate path arc lengths and local headings
        dists = np.linalg.norm(np.diff(ego_path_xy, axis=0), axis=1)
        s_coords = np.insert(np.cumsum(dists), 0, 0.0)
        total_path_length = s_coords[-1] if len(s_coords) > 0 else 50.0

        def sd_to_xy_heading(s_val: float, d_val: float) -> Tuple[float, float, float]:
            s_val = max(0.0, min(s_val, total_path_length))
            idx = np.searchsorted(s_coords, s_val)
            idx = max(1, min(idx, len(s_coords) - 1))
            ds = max(1e-4, s_coords[idx] - s_coords[idx - 1])
            ratio = (s_val - s_coords[idx - 1]) / ds
            p_prev = ego_path_xy[idx - 1]
            p_curr = ego_path_xy[idx]
            c_pt = p_prev + ratio * (p_curr - p_prev)

            # Tangent & normal vectors
            tx = (p_curr[0] - p_prev[0]) / ds
            ty = (p_curr[1] - p_prev[1]) / ds
            heading = float(np.arctan2(ty, tx))
            # Normal to the left is (-ty, tx)
            nx, ny = -ty, tx
            x = float(c_pt[0] + d_val * nx)
            y = float(c_pt[1] + d_val * ny)
            return x, y, heading

        # Open set: (f, step, s, d, v, g, parent, accel, lat_v)
        start_state = (0.0, 0, 0.0, ego_init_d, ego_init_v, 0.0, -1, 0.0, 0.0)
        open_set = [start_state]
        visited = {}
        nodes_history = []

        best_terminal = None
        best_cost = float("inf")

        while open_set and len(nodes_history) < self.max_expansions:
            f, step, s, d, v, g, parent, a, vd = heapq.heappop(open_set)

            # Quantize state key: step, s by 1m, d by 0.15m (matches lat_v*dt=0.15m), v by 1m/s
            state_key = (step, round(s), round(d / 0.15), round(v))
            if state_key in visited and visited[state_key] <= g:
                continue
            visited[state_key] = g

            node_idx = len(nodes_history)
            nodes_history.append((step, s, d, v, g, parent, a, vd))

            if step == self.horizon_steps:
                if g < best_cost:
                    best_cost = g
                    best_terminal = node_idx
                break  # A* admissible heuristic terminates on first reached goal

            for accel in self.accel_options:
                v_next = float(np.clip(v + accel * self.dt, 0.0, self.max_speed))
                s_next = float(s + 0.5 * (v + v_next) * self.dt)

                for lat_v in lat_options:
                    d_next = float(np.clip(d + lat_v * self.dt, -max_d, max_d))
                    ego_x, ego_y, ego_head = sd_to_xy_heading(s_next, d_next)

                    # Collision checking against predicted obstacles
                    collision = False
                    for track_id, obs_traj in predicted_obstacles.items():
                        step_idx = min(step + 1, len(obs_traj) - 1)
                        ox, oy = obs_traj[step_idx]

                        if use_obb:
                            o_head = gt_headings.get(track_id, ego_head) if gt_headings else ego_head
                            if check_obb_collision(
                                ego_x, ego_y, ego_head, 4.88, 2.02, ox, oy, o_head, 4.7, 1.9, safety_margin=0.1
                            ):
                                collision = True
                                break
                        else:
                            if np.hypot(ego_x - ox, ego_y - oy) < self.safety_radius:
                                collision = True
                                break

                    if collision:
                        continue  # Hard collision prune in 2D evasion mode

                    step_cost = (
                        self.w_comfort * (accel**2)
                        + w_lateral * (lat_v**2)
                        + w_lane * abs(d_next)
                    ) * self.dt
                    g_next = g + step_cost

                    remaining_steps = self.horizon_steps - (step + 1)
                    # Admissible heuristic: maximum distance progress
                    h_next = -self.w_progress * (s_next + self.max_speed * remaining_steps * self.dt)
                    f_next = g_next + h_next

                    heapq.heappush(open_set, (f_next, step + 1, s_next, d_next, v_next, g_next, node_idx, accel, lat_v))

        self.last_expansions = len(nodes_history)
        self.last_planning_time_ms = (time.perf_counter() - t0) * 1000.0

        if best_terminal is None and nodes_history:
            best_terminal = max(range(len(nodes_history)), key=lambda i: nodes_history[i][1])

        curr_idx = best_terminal
        s_traj, d_traj, v_traj, a_traj = [], [], [], []
        while curr_idx >= 0:
            st, s, d, v, g, p, a, vd = nodes_history[curr_idx]
            s_traj.append(s)
            d_traj.append(d)
            v_traj.append(v)
            a_traj.append(a)
            curr_idx = p

        s_traj.reverse()
        d_traj.reverse()
        v_traj.reverse()
        a_traj.reverse()
        return s_traj, v_traj, a_traj, d_traj

    def evaluate_plan_against_ground_truth(
        self,
        mode: str,
        ego_path_xy: np.ndarray,
        s_traj: List[float],
        v_traj: List[float],
        a_traj: List[float],
        gt_obstacles: Dict[int, List[Tuple[float, float]]],
        ego_heading: float = 0.0,
        gt_headings: Optional[Dict[int, float]] = None,
        obb_safety_margin: float = 0.1,
        d_traj: Optional[List[float]] = None,
    ) -> PlanResult:
        """
        Evaluates the executed plan against true ground truth obstacles.
        Computes safety, minimum TTC/distance, and emergency braking occurrences.
        If gt_headings is provided, computes exact Oriented Bounding Box (OBB / SAT) collision.
        Supports 2D lateral deviation via d_traj.
        """
        dists = np.linalg.norm(np.diff(ego_path_xy, axis=0), axis=1)
        s_coords = np.insert(np.cumsum(dists), 0, 0.0)
        total_length = s_coords[-1] if len(s_coords) > 0 else 50.0

        def sd_to_xy(s_val: float, d_val: float = 0.0) -> Tuple[float, float]:
            if s_val <= 0:
                c_pt = ego_path_xy[0]
                idx = 1
            elif s_val >= total_length:
                c_pt = ego_path_xy[-1]
                idx = len(s_coords) - 1
            else:
                idx = np.searchsorted(s_coords, s_val)
                idx = max(1, min(idx, len(s_coords) - 1))
                ratio = (s_val - s_coords[idx - 1]) / max(1e-4, s_coords[idx] - s_coords[idx - 1])
                c_pt = ego_path_xy[idx - 1] + ratio * (ego_path_xy[idx] - ego_path_xy[idx - 1])

            if abs(d_val) < 1e-4:
                return float(c_pt[0]), float(c_pt[1])

            p_prev = ego_path_xy[idx - 1]
            p_curr = ego_path_xy[idx]
            ds = max(1e-4, s_coords[idx] - s_coords[idx - 1])
            tx = (p_curr[0] - p_prev[0]) / ds
            ty = (p_curr[1] - p_prev[1]) / ds
            nx, ny = -ty, tx
            return float(c_pt[0] + d_val * nx), float(c_pt[1] + d_val * ny)

        collision = False
        obb_collision = False
        initial_overlap = False
        min_dist = float("inf")
        emergency_brakes = 0

        # Check initial overlap at t=0
        if gt_headings is not None:
            d0 = d_traj[0] if d_traj else 0.0
            ex0, ey0 = sd_to_xy(s_traj[0], d0) if s_traj else (float(ego_path_xy[0, 0]), float(ego_path_xy[0, 1]))
            for track_id, gt_traj in gt_obstacles.items():
                ox0, oy0 = gt_traj[0]
                o_head = gt_headings.get(track_id, ego_heading)
                if check_obb_collision(ex0, ey0, ego_heading, 4.88, 2.02, ox0, oy0, o_head, 4.7, 1.9, safety_margin=0.0):
                    initial_overlap = True
                    break

        for t_idx, (s, a) in enumerate(zip(s_traj, a_traj)):
            d_val = d_traj[t_idx] if d_traj and t_idx < len(d_traj) else 0.0
            ego_x, ego_y = sd_to_xy(s, d_val)
            for track_id, gt_traj in gt_obstacles.items():
                step_idx = min(t_idx, len(gt_traj) - 1)
                ox, oy = gt_traj[step_idx]
                d = float(np.hypot(ego_x - ox, ego_y - oy))
                if d < min_dist:
                    min_dist = d
                if d < self.safety_radius:
                    collision = True

                if gt_headings is not None and not obb_collision:
                    o_head = gt_headings.get(track_id, ego_heading)
                    if check_obb_collision(ego_x, ego_y, ego_heading, 4.88, 2.02, ox, oy, o_head, 4.7, 1.9, safety_margin=obb_safety_margin):
                        obb_collision = True

            if a <= -3.0:
                emergency_brakes += 1

        total_distance = s_traj[-1] - s_traj[0] if len(s_traj) > 1 else 0.0
        avg_speed = float(np.mean(v_traj)) if v_traj else 0.0
        max_lat = float(max(abs(x) for x in d_traj)) if d_traj else 0.0

        return PlanResult(
            mode=mode,
            success=not collision,
            collision_occurred=collision,
            min_distance_to_obstacle=round(min_dist, 2),
            emergency_braking_events=emergency_brakes,
            total_travel_distance=round(total_distance, 2),
            average_speed=round(avg_speed, 2),
            obb_collision_occurred=obb_collision,
            initial_overlap_detected=initial_overlap,
            trajectory_s=s_traj,
            trajectory_v=v_traj,
            trajectory_a=a_traj,
            trajectory_d=d_traj or [],
            expansions=self.last_expansions,
            planning_time_ms=round(self.last_planning_time_ms, 2),
            max_lateral_displacement=round(max_lat, 2),
        )


class DijkstraTrajectoryPlanner(AStarTrajectoryPlanner):
    """
    Spatiotemporal Dijkstra Planner baseline (Uniform-Cost Search).
    Sets the heuristic h(n) = 0 to evaluate the search efficiency
    gained by A*'s spatiotemporal progress heuristic.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w_progress = 0.0  # h(n) = 0, purely cumulative cost


class GreedyTrajectoryPlanner(AStarTrajectoryPlanner):
    """
    Greedy Best-First Trajectory Planner baseline.
    Prioritizes nodes purely by heuristic distance to goal h(n),
    ignoring accumulated control cost g(n).
    """

    def plan_trajectory(
        self,
        ego_init_v: float,
        ego_path_xy: np.ndarray,
        predicted_obstacles: Dict[int, List[Tuple[float, float]]],
    ) -> Tuple[List[float], List[float], List[float]]:
        t0 = time.perf_counter()
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

        # Priority is purely h_score
        start_state = (0.0, 0, 0.0, ego_init_v, 0.0, -1, 0.0)
        open_set = [start_state]
        visited = {}
        nodes_history = []
        best_terminal = None

        while open_set and len(nodes_history) < self.max_expansions:
            h_score, step, s, v, g, parent, a = heapq.heappop(open_set)

            state_key = (step, round(s * 2) / 2, round(v * 2) / 2)
            if state_key in visited and visited[state_key] <= g:
                continue
            visited[state_key] = g

            node_idx = len(nodes_history)
            nodes_history.append((step, s, v, g, parent, a))

            if step == self.horizon_steps:
                best_terminal = node_idx
                break

            for accel in self.accel_options:
                v_next = float(np.clip(v + accel * self.dt, 0.0, self.max_speed))
                s_next = float(s + 0.5 * (v + v_next) * self.dt)

                ego_x, ego_y = s_to_xy(s_next)
                collision = False
                for track_id, obs_traj in predicted_obstacles.items():
                    step_idx = min(step + 1, len(obs_traj) - 1)
                    ox, oy = obs_traj[step_idx]
                    if np.hypot(ego_x - ox, ego_y - oy) < self.safety_radius:
                        collision = True
                        break

                step_cost = self.w_comfort * (accel**2) * self.dt
                if collision:
                    step_cost += self.w_collision

                g_next = g + step_cost
                remaining_steps = self.horizon_steps - (step + 1)
                # Greedy: priority is solely h_next
                h_next = -self.w_progress * (s_next + v_next * remaining_steps * self.dt)
                heapq.heappush(open_set, (h_next, step + 1, s_next, v_next, g_next, node_idx, accel))

        self.last_expansions = len(nodes_history)
        self.last_planning_time_ms = (time.perf_counter() - t0) * 1000.0

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

