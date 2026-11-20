# Raw Dataset Storage

Raw Waymo Open Motion Dataset files are **intentionally not tracked in Git** due to Waymo Open Dataset license terms and distribution restrictions.

## Instructions to Obtain and Use the Dataset

1. **Register and Request Access:**
   - Visit the official [Waymo Open Dataset](https://waymo.com/open/) portal.
   - Register an account and agree to the [Waymo Open Dataset Terms of Use](https://waymo.com/open/terms-of-use/).
2. **Download Motion Dataset Shards:**
   - Download the motion dataset TFRecord shards (for example, `training_20s.tfrecord-00000-of-01000` and `training_20s.tfrecord-00001-of-01000`).
3. **Place Locally or Configure Path:**
   - Place downloaded shards either in this directory (`data/raw/`) or anywhere locally (e.g. `~/Downloads/archive`).
   - Configure the environment variable:
     ```bash
     export WAYMO_DATA_DIR=~/Downloads/archive
     # or create a .env file based on .env.example
     ```
4. **Run Exploration Pipeline:**
   - Execute the exploration pipeline or notebook:
     ```bash
     python scripts/explore_dataset.py
     python scripts/export_csv.py
     ```

## Waymo Attribution & License Notice

This project utilizes data from the **Waymo Open Dataset**, provided by Waymo LLC under the Waymo Open Dataset Terms of Use. Data is used strictly for non-commercial educational and research purposes.
