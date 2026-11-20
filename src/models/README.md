# Models (Phase 2 & 3 Roadmap)

> **Notice:** Per Phase 1 project specifications, no machine learning models are trained in this phase. This module will house classical Machine Learning model implementations and training pipelines in Phase 2/3.

## Planned Classical ML Models

In accordance with project scope (comparing classical ML approaches for tactical decision prediction without heavy deep learning), the following models will be evaluated:

1. **Decision Tree Classifier (`DecisionTreeClassifier`)**
   - Serves as interpretable baseline with explicit rule branching and feature importance.
2. **K-Nearest Neighbors (`KNeighborsClassifier`)**
   - Instance-based spatial-temporal similarity baseline.
3. **Support Vector Machine (`SVC` with RBF/Linear kernels)**
   - High-dimensional margin maximization with kernel tricks for complex decision boundaries.
4. **Random Forest Classifier (`RandomForestClassifier`)**
   - Ensemble bagging model providing high robustness against feature variance and non-linear interactions.

## Planned Target Labels (Phase 2 Target Engineering)

Derived from the scenario's future trajectory timeline (future 3-5 seconds):

### Lateral Decision Targets
- `KEEP_LANE`: Vehicle maintains current lane corridor.
- `CHANGE_LEFT`: Vehicle executes lane change to the left.
- `CHANGE_RIGHT`: Vehicle executes lane change to the right.
- `TURN_LEFT`: Vehicle negotiates an intersection or turn to the left.
- `TURN_RIGHT`: Vehicle negotiates an intersection or turn to the right.

### Longitudinal Decision Targets
- `ACCELERATE`: Longitudinal acceleration above positive threshold ($\Delta v > \epsilon$).
- `MAINTAIN`: Cruising speed within steady threshold ($|\Delta v| \le \epsilon$).
- `DECELERATE`: Gradual braking / speed reduction ($\Delta v < -\epsilon$).
- `STOP`: Vehicle reaches near-zero speed ($v \approx 0$).

> **Important Data Leakage Rule:** Future trajectories are used **strictly** to compute ground-truth target labels. They must **never** be included in feature sets.

## Candidate Engineered Features (Phase 2)

Features derived strictly from current and past observation windows ($t \le t_{\text{current}}$):

- **Ego Kinematics:**
  - `ego_speed` ($v = \sqrt{v_x^2 + v_y^2}$)
  - `ego_acceleration` ($a = \frac{\Delta v}{\Delta t}$)
  - `ego_heading_change` ($\Delta \theta$)
- **Front Agent Interactions:**
  - `front_vehicle_distance`: Distance to lead vehicle along heading corridor.
  - `front_relative_speed`: $v_{\text{ego}} - v_{\text{front}}$
  - `front_TTC`: Time-To-Collision ($\frac{d_{\text{front}}}{\Delta v}$)
- **Surrounding Agent Distances:**
  - `left_front_distance`, `left_rear_distance`
  - `right_front_distance`, `right_rear_distance`
  - `nearest_pedestrian_distance`, `nearest_cyclist_distance`
  - `nearby_vehicle_count` (within 50m radius)
- **Map & Traffic Controls:**
  - `traffic_light_state` (stop, caution, go, unknown)
  - `distance_to_crosswalk`
  - `distance_to_stop_sign`
  - `distance_to_intersection`

## Evaluation Metrics

To ensure robust evaluation on potentially imbalanced tactical driving decisions:

- **Accuracy**: Overall classification accuracy.
- **Precision (Macro & Per-Class)**: To assess false positive rates per maneuver.
- **Recall (Macro & Per-Class)**: Critical for safety-critical maneuvers (e.g. stopping, decelerating).
- **F1 Macro**: Unweighted mean of class F1 scores to prevent majority-class bias.
- **Confusion Matrix**: Multi-class matrix analysis to identify error patterns between adjacent maneuvers.
