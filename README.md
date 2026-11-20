# Machine Learning-Based Tactical Decision Prediction for Autonomous Driving

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-Waymo_Non--Commercial_Research-lightgrey.svg)](https://waymo.com/open/terms-of-use/)
[![Tests](https://img.shields.io/badge/tests-16%20passed-brightgreen.svg)]()

## 1. Project Overview & Goal

Autonomous driving systems require safe, explainable, and robust tactical decision-making when interacting with complex urban environments. While deep learning models often operate as end-to-end "black boxes", classical Machine Learning algorithms (such as Decision Trees, Random Forests, Support Vector Machines, and K-Nearest Neighbors) offer interpretability, bounded execution latency, and inspectable decision boundaries.

**Goal:**  
Utilize rich 20-second multi-agent driving scenarios from the **Waymo Open Motion Dataset** to extract structured spatio-temporal, topological, and agent-interaction tabular features, enabling rigorous comparison of classical Machine Learning algorithms for tactical decision prediction (e.g., lane keeping, lane change, stopping, deceleration, and intersection turning).

---

## 2. Architecture Pipeline

```text
Waymo Open Motion Dataset (TFRecords)
               ↓
Incremental Streaming & Reading (TensorFlow tf.data)
               ↓
Protobuf Scenario Decoding (waymo_open_dataset.protos)
               ↓
Tabular Extraction & Data Quality Auditing (Pandas / NumPy)
               ↓
Dataset Exploration & BEV Top-Down Visualization (Matplotlib)
               ↓
Spatial-Temporal Feature Engineering (Ego-Centric Frame & Interactions)
               ↓
Target Labeling (Future Trajectory Horizon)
               ↓
Classical Machine Learning Classifiers (scikit-learn)
┌─────────────────────────────────────────────────────────────┐
│  Decision Tree  │  K-Nearest Neighbors  │  SVM  │  Random Forest │
└─────────────────────────────────────────────────────────────┘
               ↓
Evaluation & Benchmarking (Accuracy, Precision, Recall, Macro F1, Confusion Matrix)
```

---

## 3. Project Status

| Phase | Description | Status |
| :--- | :--- | :---: |
| **Phase 1** | **Dataset Exploration, Protobuf Decoding & Tabular Extraction** | **COMPLETED & VALIDATED** |
| **Phase 2** | Target Engineering, Ego-Centric Feature Engineering & Class Distribution Analysis | Planned (Next) |
| **Phase 3** | Classical ML Model Training, Hyperparameter Tuning & Cross-Validation | Planned |
| **Phase 4** | Model Benchmarking, Evaluation & Comparative Analysis | Planned |

---

## 4. Dataset Information & Access Instructions

> **IMPORTANT NOTICE:**  
> Raw Waymo TFRecord files are **intentionally not tracked in Git** due to licensing and distribution restrictions. All raw dataset files must remain local.

### Obtaining the Waymo Open Motion Dataset
1. Visit the official [Waymo Open Dataset Portal](https://waymo.com/open/).
2. Register an account and agree to the [Waymo Open Dataset Terms of Use](https://waymo.com/open/terms-of-use/).
3. Download the Motion Dataset (v1.2) TFRecord shards (for development, 2 shards are utilized: `training_20s.tfrecord-00000-of-01000` and `training_20s.tfrecord-00001-of-01000`).
4. Place the downloaded files locally:
   - Default local project path: `data/raw/`
   - Or external archive path: `~/Downloads/archive`
5. Configure your local environment:
   ```bash
   cp .env.example .env
   # Edit .env and set:
   # WAYMO_DATA_DIR=/Users/<user>/Downloads/archive
   ```

### Development Sample Statistics (Local 2-Shard Sample)
- **Files Discovered:** 2 TFRecord shards
  - `training_20s.tfrecord-00000-of-01000` (90.20 MB)
  - `training_20s.tfrecord-00001-of-01000` (93.67 MB)
- **Total Local Sample Size:** 183.87 MB (192,800,360 bytes)
- **Total Scenarios Available:** 149 scenarios (100 explored in Phase 1 benchmark)
- **Timestamps per Scenario:** ~198 to 200 time steps at 10 Hz (~20 seconds continuous driving)
- **Agents Observed (Current Step):** 8,998 tracked agents
  - Vehicles: 8,113 (90.2%)
  - Pedestrians: 832 (9.2%)
  - Cyclists: 53 (0.6%)
- **HD Vector Map Elements:** 31,966 map features (19,754 lanes, 4,109 road edges, 3,700 road lines, 768 crosswalks, 485 speed bumps, 345 stop signs, 2,805 driveways)
- **SDC / Ego Vehicle Availability:** 100% valid across all scenarios

### Waymo Dataset Attribution
This project uses data from the **Waymo Open Dataset**, provided by Waymo LLC under license. Waymo LLC is not affiliated with and does not endorse this project.

---

## 5. Repository Structure

```text
projeto-2/
├── README.md                          # Project documentation and architectural overview
├── requirements.txt                   # Phase 1 runtime dependencies
├── .gitignore                         # Strict protection against raw dataset commits
├── .env.example                       # Environment variable templates
├── notebooks/
│   ├── 01_dataset_exploration.ipynb   # Executed exploration notebook with full visualizations
│   ├── 02_preprocessing.ipynb         # Phase 2: Data cleaning & preprocessing scaffold
│   ├── 03_feature_engineering.ipynb   # Phase 2: Interaction features & target derivation scaffold
│   └── 04_modeling.ipynb              # Phase 3: Classical ML benchmarking scaffold
├── src/
│   ├── data/
│   │   ├── tfrecord_reader.py         # Incremental TFRecord reader and proto decoding
│   │   ├── scenario_parser.py         # Waymo Scenario protobuf to Pandas DataFrames
│   │   └── dataset_exporter.py        # CSV tabular serializer and figure generator
│   ├── preprocessing/
│   │   └── preprocessing.py           # Automated data quality auditing & validation
│   ├── features/
│   │   └── feature_engineering.py     # Ego-frame transform, TTC, and feature signatures
│   └── models/
│       └── README.md                  # Phase 2 & 3 ML architecture and target design
├── scripts/
│   ├── explore_dataset.py             # CLI pipeline for exploration, audit, and plots
│   └── export_csv.py                  # CLI pipeline for generating export CSV tables
├── outputs/
│   ├── figures/                       # Generated exploratory visualizations
│   │   ├── object_distribution.png
│   │   ├── speed_distribution.png
│   │   ├── objects_per_scenario.png
│   │   └── topdown_scenario.png
│   ├── tables/                        # Generated tabular CSV exports
│   │   ├── scenario_summary.csv
│   │   ├── objects_current.csv
│   │   ├── ego_current.csv
│   │   ├── waymo_excel_preview.csv    # Human-readable Excel preview (~800 rows)
│   │   └── waymo_data_dictionary.csv  # Detailed schema definitions
│   └── reports/
│       └── dataset_exploration_summary.json
├── tests/                             # Automated test suite (pytest)
│   ├── conftest.py
│   ├── test_tfrecord_reader.py
│   ├── test_scenario_parser.py
│   ├── test_dataset_exporter.py
│   └── test_schemas_and_quality.py
└── data/
    └── raw/
        └── README.md                  # Instructions for local dataset placement
```

---

## 6. Installation & Quickstart

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/1-2Drive/projeto-2.git
cd projeto-2

python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Dataset Location
Set the path to your local Waymo dataset directory:
```bash
export WAYMO_DATA_DIR=~/Downloads/archive
```

### 3. Run Exploration Pipeline
Execute the complete Phase 1 exploration and CSV export pipeline:
```bash
python scripts/explore_dataset.py
```

### 4. Run Automated Tests
```bash
pytest -v
```

### 5. Launch Jupyter Notebook
```bash
jupyter notebook notebooks/01_dataset_exploration.ipynb
```

---

## 7. Data Quality & Validation Summary

Automated checks in `src/preprocessing/preprocessing.py` verified the following metrics across the dataset sample:

- **Duplicate Scenario IDs:** 0 (100% unique)
- **Duplicate Object IDs:** 0 within scenario coordinate spaces
- **Speed Bounds:** Speeds strictly non-negative, ranging between $0.00\text{ m/s}$ and $27.72\text{ m/s}$ ($\approx 99.8\text{ km/h}$, realistic for urban driving)
- **Infinite / NaN Values:** 0 invalid/unobserved coordinate values in valid states
- **Autonomous Vehicle (SDC) State:** Valid in 100% of tested scenarios

---

## 8. Phase 2 Roadmap: Target & Feature Engineering

1. **Tactical Target Derivation (from Future Trajectory):**
   - Lateral maneuvers: `KEEP_LANE`, `CHANGE_LEFT`, `CHANGE_RIGHT`, `TURN_LEFT`, `TURN_RIGHT`.
   - Longitudinal maneuvers: `ACCELERATE`, `MAINTAIN`, `DECELERATE`, `STOP`.
   - *Strict Data Leakage Prevention:* Future trajectories are used exclusively for target labeling and never as input features.
2. **Ego-Centric Feature Engineering:**
   - Kinematics: `ego_speed`, `ego_acceleration`, `ego_heading_change`.
   - Interactions: `front_vehicle_distance`, `front_relative_speed`, `front_ttc` (Time-To-Collision).
   - Spatial Proximity: `left_front_distance`, `right_front_distance`, `nearby_vehicle_count`.
   - Context: Distance to crosswalks, stop signs, and traffic light states.
3. **Model Benchmarking:**
   - Training and evaluating **Decision Tree**, **KNN**, **SVM**, and **Random Forest** with stratified cross-validation and hyperparameter tuning.
