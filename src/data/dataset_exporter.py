"""Exporter for tabular CSVs, data dictionaries, Excel previews, and visualization figures."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from waymo_open_dataset.protos import scenario_pb2


DATA_DICTIONARY_ENTRIES = [
    # scenario_summary
    ("scenario_summary", "scenario_id", "string", "Hex string", "Unique identifier for the 20-second driving scenario", "41600af30ab8cc55"),
    ("scenario_summary", "timestamps_count", "integer", "Count", "Total recorded time steps in scenario (~10 Hz)", "198"),
    ("scenario_summary", "current_time_index", "integer", "Index", "Index of current observation timestamp", "0"),
    ("scenario_summary", "num_tracks", "integer", "Count", "Total unique agent tracks recorded across the scenario", "54"),
    ("scenario_summary", "num_vehicles", "integer", "Count", "Number of vehicle agents identified", "42"),
    ("scenario_summary", "num_pedestrians", "integer", "Count", "Number of pedestrian agents identified", "8"),
    ("scenario_summary", "num_cyclists", "integer", "Count", "Number of cyclist agents identified", "2"),
    ("scenario_summary", "num_other", "integer", "Count", "Number of other/unset dynamic objects", "2"),
    ("scenario_summary", "num_map_features", "integer", "Count", "Total HD vector map elements associated with scenario", "217"),
    ("scenario_summary", "num_dynamic_map_states", "integer", "Count", "Number of dynamic map states (traffic signals)", "198"),
    ("scenario_summary", "ego_valid", "boolean", "True/False", "Whether Autonomous/SDC vehicle state is valid at current step", "True"),
    ("scenario_summary", "sdc_track_index", "integer", "Index", "Index of SDC track in tracks array", "53"),
    ("scenario_summary", "num_lanes", "integer", "Count", "Number of vector lane centerline features", "180"),
    ("scenario_summary", "num_road_edges", "integer", "Count", "Number of road boundary/edge features", "25"),
    ("scenario_summary", "num_road_lines", "integer", "Count", "Number of road paint line features", "12"),
    ("scenario_summary", "num_crosswalks", "integer", "Count", "Number of pedestrian crosswalk features", "4"),
    ("scenario_summary", "num_speed_bumps", "integer", "Count", "Number of speed bump features", "2"),
    ("scenario_summary", "num_stop_signs", "integer", "Count", "Number of stop sign features", "2"),
    ("scenario_summary", "num_driveways", "integer", "Count", "Number of driveway boundary features", "8"),

    # objects_current
    ("objects_current", "scenario_id", "string", "Hex string", "Identifier of scenario containing the object", "41600af30ab8cc55"),
    ("objects_current", "object_id", "integer", "Integer ID", "Unique track ID within the scenario", "201"),
    ("objects_current", "object_type", "string", "VEHICLE/PEDESTRIAN/CYCLIST/OTHER", "Classification of the tracked object", "VEHICLE"),
    ("objects_current", "is_ego", "boolean", "True/False", "Flag indicating whether track is the autonomous vehicle (SDC)", "False"),
    ("objects_current", "x", "float", "Meters (UTM/local)", "Bounding box center X coordinate", "8435.12"),
    ("objects_current", "y", "float", "Meters (UTM/local)", "Bounding box center Y coordinate", "660.45"),
    ("objects_current", "z", "float", "Meters (UTM/local)", "Bounding box center Z coordinate", "-134.10"),
    ("objects_current", "velocity_x", "float", "m/s", "Instantaneous linear velocity along X axis", "4.25"),
    ("objects_current", "velocity_y", "float", "m/s", "Instantaneous linear velocity along Y axis", "-1.10"),
    ("objects_current", "speed", "float", "m/s", "Planar speed magnitude: sqrt(vx^2 + vy^2)", "4.39"),
    ("objects_current", "heading", "float", "Radians [-pi, pi]", "Yaw heading angle of bounding box", "-0.25"),
    ("objects_current", "length", "float", "Meters", "Bounding box extent along longitudinal heading axis", "4.80"),
    ("objects_current", "width", "float", "Meters", "Bounding box extent along lateral axis", "2.10"),
    ("objects_current", "height", "float", "Meters", "Bounding box vertical height", "1.75"),
    ("objects_current", "valid", "boolean", "True/False", "Whether bounding box state is valid/observed at this timestamp", "True"),

    # ego_current
    ("ego_current", "scenario_id", "string", "Hex string", "Scenario identifier", "41600af30ab8cc55"),
    ("ego_current", "ego_x", "float", "Meters", "Ego vehicle center X coordinate", "8431.89"),
    ("ego_current", "ego_y", "float", "Meters", "Ego vehicle center Y coordinate", "654.18"),
    ("ego_current", "ego_speed", "float", "m/s", "Ego planar speed: sqrt(vx^2 + vy^2)", "0.35"),
    ("ego_current", "ego_velocity_x", "float", "m/s", "Ego longitudinal/lateral velocity X component", "0.002"),
    ("ego_current", "ego_velocity_y", "float", "m/s", "Ego longitudinal/lateral velocity Y component", "-0.348"),
    ("ego_current", "ego_heading", "float", "Radians [-pi, pi]", "Ego vehicle yaw heading orientation", "-1.559"),
    ("ego_current", "nearby_vehicle_count", "integer", "Count within 50m", "Number of active vehicles within 50-meter radius of Ego", "5"),
    ("ego_current", "nearby_pedestrian_count", "integer", "Count within 50m", "Number of active pedestrians within 50-meter radius of Ego", "1"),
    ("ego_current", "nearby_cyclist_count", "integer", "Count within 50m", "Number of active cyclists within 50-meter radius of Ego", "0"),
]


class DatasetExporter:
    """Manages table serialization and exploratory figure rendering."""

    def __init__(self, output_dir: Union[str, Path] = "outputs"):
        self.output_dir = Path(output_dir).resolve()
        self.tables_dir = self.output_dir / "tables"
        self.figures_dir = self.output_dir / "figures"
        self.reports_dir = self.output_dir / "reports"

        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def export_tables(
        self,
        summary_df: pd.DataFrame,
        objects_df: pd.DataFrame,
        ego_df: pd.DataFrame,
        preview_max_rows: int = 800,
    ) -> dict:
        """Export required CSV files and data dictionary."""
        paths = {}

        # 1. scenario_summary.csv
        p_summary = self.tables_dir / "scenario_summary.csv"
        summary_df.to_csv(p_summary, index=False)
        paths["scenario_summary"] = str(p_summary)

        # 2. objects_current.csv
        p_objects = self.tables_dir / "objects_current.csv"
        objects_df.to_csv(p_objects, index=False)
        paths["objects_current"] = str(p_objects)

        # 3. ego_current.csv
        p_ego = self.tables_dir / "ego_current.csv"
        ego_df.to_csv(p_ego, index=False)
        paths["ego_current"] = str(p_ego)

        # 4. waymo_excel_preview.csv (500-1000 human-friendly rows)
        p_preview = self.tables_dir / "waymo_excel_preview.csv"
        preview_df = self._build_excel_preview(summary_df, objects_df, ego_df, max_rows=preview_max_rows)
        preview_df.to_csv(p_preview, index=False)
        paths["waymo_excel_preview"] = str(p_preview)

        # 5. waymo_data_dictionary.csv
        p_dict = self.tables_dir / "waymo_data_dictionary.csv"
        dict_df = pd.DataFrame(
            DATA_DICTIONARY_ENTRIES,
            columns=["table_name", "column_name", "data_type", "unit_or_format", "description", "example_value"],
        )
        dict_df.to_csv(p_dict, index=False)
        paths["waymo_data_dictionary"] = str(p_dict)

        return paths

    def _build_excel_preview(
        self,
        summary_df: pd.DataFrame,
        objects_df: pd.DataFrame,
        ego_df: pd.DataFrame,
        max_rows: int = 800,
    ) -> pd.DataFrame:
        """Construct a merged, human-readable tabular preview for spreadsheet inspection."""
        # Join objects with ego speed and scenario track counts for context
        valid_objs = objects_df[objects_df["valid"]].copy()
        if len(valid_objs) == 0:
            valid_objs = objects_df.copy()

        merged = valid_objs.merge(
            ego_df[["scenario_id", "ego_speed", "nearby_vehicle_count", "nearby_pedestrian_count"]],
            on="scenario_id",
            how="left",
        )

        # Round numeric values for clean spreadsheet display
        numeric_cols = ["x", "y", "z", "velocity_x", "velocity_y", "speed", "heading", "length", "width", "height", "ego_speed"]
        for col in numeric_cols:
            if col in merged.columns:
                merged[col] = merged[col].round(3)

        preview = merged.head(max_rows)
        return preview

    def generate_figures(
        self,
        summary_df: pd.DataFrame,
        objects_df: pd.DataFrame,
        scenarios: Optional[List[scenario_pb2.Scenario]] = None,
    ) -> dict:
        """Generate exploratory visual plots:

        - outputs/figures/object_distribution.png
        - outputs/figures/speed_distribution.png
        - outputs/figures/objects_per_scenario.png
        - outputs/figures/topdown_scenario.png
        """
        paths = {}

        # 1. object_distribution.png
        p_obj_dist = self.figures_dir / "object_distribution.png"
        fig, ax = plt.subplots(figsize=(8, 5))
        valid_objs = objects_df[objects_df["valid"]] if "valid" in objects_df.columns else objects_df
        type_counts = valid_objs["object_type"].value_counts()
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        type_counts.plot(kind="bar", ax=ax, color=colors[:len(type_counts)], edgecolor="black")
        ax.set_title("Distribution of Observed Agent Types at Current Timestamp", fontsize=13, fontweight="bold")
        ax.set_xlabel("Object Classification", fontsize=11)
        ax.set_ylabel("Total Count", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        fig.savefig(p_obj_dist, dpi=200)
        plt.close(fig)
        paths["object_distribution"] = str(p_obj_dist)

        # 2. speed_distribution.png
        p_speed_dist = self.figures_dir / "speed_distribution.png"
        fig, ax = plt.subplots(figsize=(8, 5))
        speeds = valid_objs["speed"].dropna()
        ax.hist(speeds, bins=40, color="#2b5c8f", edgecolor="black", alpha=0.8)
        mean_speed = speeds.mean()
        p95_speed = np.percentile(speeds, 95) if len(speeds) > 0 else 0.0
        ax.axvline(mean_speed, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_speed:.2f} m/s")
        ax.axvline(p95_speed, color="orange", linestyle=":", linewidth=1.5, label=f"95th %ile: {p95_speed:.2f} m/s")
        ax.set_title("Agent Speed Distribution (m/s) at Current Timestamp", fontsize=13, fontweight="bold")
        ax.set_xlabel("Speed (m/s)", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        fig.savefig(p_speed_dist, dpi=200)
        plt.close(fig)
        paths["speed_distribution"] = str(p_speed_dist)

        # 3. objects_per_scenario.png
        p_per_scen = self.figures_dir / "objects_per_scenario.png"
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(summary_df["num_tracks"], bins=25, color="#388e3c", edgecolor="black", alpha=0.8)
        ax.set_title("Distribution of Total Tracks Per Scenario", fontsize=13, fontweight="bold")
        ax.set_xlabel("Track Count per Scenario", fontsize=11)
        ax.set_ylabel("Number of Scenarios", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        fig.savefig(p_per_scen, dpi=200)
        plt.close(fig)
        paths["objects_per_scenario"] = str(p_per_scen)

        # 4. topdown_scenario.png
        p_topdown = self.figures_dir / "topdown_scenario.png"
        self._plot_topdown_scenario(objects_df, scenarios=scenarios, out_path=p_topdown)
        paths["topdown_scenario"] = str(p_topdown)

        return paths

    def _plot_topdown_scenario(
        self,
        objects_df: pd.DataFrame,
        scenarios: Optional[List[scenario_pb2.Scenario]] = None,
        out_path: Optional[Path] = None,
    ) -> None:
        """Render bird's-eye-view (BEV) top-down plot distinguishing EGO, vehicles, pedestrians, cyclists."""
        if out_path is None:
            out_path = self.figures_dir / "topdown_scenario.png"

        # Select the scenario with the most diverse objects
        first_sc_id = objects_df["scenario_id"].iloc[0] if not objects_df.empty else None
        sc_subset = objects_df[(objects_df["scenario_id"] == first_sc_id) & objects_df["valid"]].copy()

        fig, ax = plt.subplots(figsize=(9, 9))

        # Center around ego if present
        ego_row = sc_subset[sc_subset["is_ego"]]
        if not ego_row.empty:
            cx, cy = ego_row["x"].iloc[0], ego_row["y"].iloc[0]
            sc_subset["rel_x"] = sc_subset["x"] - cx
            sc_subset["rel_y"] = sc_subset["y"] - cy
            xlabel = "Relative X (meters from Ego)"
            ylabel = "Relative Y (meters from Ego)"
        else:
            sc_subset["rel_x"] = sc_subset["x"]
            sc_subset["rel_y"] = sc_subset["y"]
            xlabel = "Global X (meters)"
            ylabel = "Global Y (meters)"

        # Scatter plot categories
        types_style = {
            "VEHICLE": {"color": "#1f77b4", "marker": "s", "size": 60, "label": "Vehicles"},
            "PEDESTRIAN": {"color": "#ff7f0e", "marker": "o", "size": 50, "label": "Pedestrians"},
            "CYCLIST": {"color": "#2ca02c", "marker": "^", "size": 70, "label": "Cyclists"},
            "OTHER": {"color": "#7f7f7f", "marker": "x", "size": 40, "label": "Other"},
        }

        # Plot non-ego agents
        non_ego = sc_subset[~sc_subset["is_ego"]]
        for t_name, style in types_style.items():
            sub = non_ego[non_ego["object_type"] == t_name]
            if not sub.empty:
                ax.scatter(
                    sub["rel_x"],
                    sub["rel_y"],
                    c=style["color"],
                    marker=style["marker"],
                    s=style["size"],
                    alpha=0.85,
                    edgecolors="black",
                    label=style["label"],
                )

        # Plot EGO prominently
        ego_plot = sc_subset[sc_subset["is_ego"]]
        if not ego_plot.empty:
            ax.scatter(
                ego_plot["rel_x"],
                ego_plot["rel_y"],
                c="red",
                marker="*",
                s=250,
                edgecolors="black",
                linewidth=1.5,
                label="SDC / EGO",
                zorder=10,
            )

        ax.set_title(f"Top-Down Bird's-Eye View: Scenario {first_sc_id}", fontsize=13, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_aspect("equal", adjustable="datalim")

        plt.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
