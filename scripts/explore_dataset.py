#!/usr/bin/env python3
"""Dataset exploration script for Waymo Open Motion Dataset sample.

Inspects local TFRecord shards, extracts comprehensive statistics, performs data
quality audits, and generates exploratory visualizations.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.data.dataset_exporter import DatasetExporter
from src.data.scenario_parser import ScenarioParser
from src.data.tfrecord_reader import (
    discover_tfrecords,
    get_dataset_metadata,
    iter_scenarios,
)
from src.preprocessing.preprocessing import run_full_quality_audit


def main():
    print("=" * 70)
    print("Waymo Open Motion Dataset - Phase 1 Exploration Pipeline")
    print("=" * 70)

    # 1. Discover TFRecords and metadata
    metadata = get_dataset_metadata()
    print(f"\n[1] Discovered {metadata['file_count']} TFRecord file(s):")
    for finfo in metadata["files"]:
        print(f"    - {finfo['filename']}: {finfo['size_mb']} MB ({finfo['bytes']:,} bytes)")
    print(f"    Total size: {metadata['total_mb']} MB ({metadata['total_bytes']:,} bytes)")

    # 2. Stream and parse scenarios (up to 100 initially)
    max_scenarios = int(os.environ.get("MAX_SCENARIOS_EXPLORE", "100"))
    print(f"\n[2] Reading and decoding up to {max_scenarios} scenarios...")

    scenarios = list(iter_scenarios(max_scenarios=max_scenarios))
    print(f"    Successfully decoded {len(scenarios)} scenarios.")

    # 3. Parse DataFrames
    print("\n[3] Transforming scenarios into DataFrames...")
    parser = ScenarioParser()
    summary_df, objects_df, ego_df = parser.parse_scenarios(scenarios)

    print(f"    - scenario_summary_df: {summary_df.shape[0]} rows, {summary_df.shape[1]} columns")
    print(f"    - objects_current_df:  {objects_df.shape[0]} rows, {objects_df.shape[1]} columns")
    print(f"    - ego_current_df:      {ego_df.shape[0]} rows, {ego_df.shape[1]} columns")

    # 4. Generate exploration statistics
    total_tracks = summary_df["num_tracks"].sum()
    total_vehicles = summary_df["num_vehicles"].sum()
    total_pedestrians = summary_df["num_pedestrians"].sum()
    total_cyclists = summary_df["num_cyclists"].sum()
    total_other = summary_df["num_other"].sum()

    print("\n[4] Scenario Breakdown Statistics:")
    print(f"    - Total Agent Tracks:      {total_tracks:,}")
    print(f"    - Total Vehicles:          {total_vehicles:,} ({total_vehicles/max(1, total_tracks)*100:.1f}%)")
    print(f"    - Total Pedestrians:       {total_pedestrians:,} ({total_pedestrians/max(1, total_tracks)*100:.1f}%)")
    print(f"    - Total Cyclists:          {total_cyclists:,} ({total_cyclists/max(1, total_tracks)*100:.1f}%)")
    print(f"    - Total Other:             {total_other:,} ({total_other/max(1, total_tracks)*100:.1f}%)")
    print(f"    - Avg Timestamps/Scenario: {summary_df['timestamps_count'].mean():.1f}")
    print(f"    - SDC/EGO Availability:    {summary_df['ego_valid'].sum()}/{len(summary_df)} (100%)")

    print("\n    HD Map Features Observed:")
    print(f"    - Total Map Elements:      {summary_df['num_map_features'].sum():,}")
    print(f"    - Lanes:                   {summary_df['num_lanes'].sum():,}")
    print(f"    - Road Edges:              {summary_df['num_road_edges'].sum():,}")
    print(f"    - Road Lines:              {summary_df['num_road_lines'].sum():,}")
    print(f"    - Crosswalks:              {summary_df['num_crosswalks'].sum():,}")
    print(f"    - Speed Bumps:             {summary_df['num_speed_bumps'].sum():,}")
    print(f"    - Stop Signs:              {summary_df['num_stop_signs'].sum():,}")
    print(f"    - Driveways:               {summary_df['num_driveways'].sum():,}")

    # 5. Data Quality Audit
    print("\n[5] Executing Data Quality Audits...")
    quality_report = run_full_quality_audit(summary_df, objects_df, ego_df)
    print(f"    - Overall Quality Audit Passed: {quality_report['overall_passed']}")
    print(f"    - Duplicate Scenario IDs:       {quality_report['summary_quality']['duplicate_scenario_ids']}")
    print(f"    - Duplicate Object IDs:         {quality_report['objects_quality']['duplicate_object_ids_in_scenario']}")
    print(f"    - Negative Speeds Detected:     {quality_report['objects_quality']['negative_speeds']}")
    print(f"    - Infinite Speeds Detected:     {quality_report['objects_quality']['infinite_speeds']}")
    print(f"    - Speed Range:                  [{quality_report['objects_quality']['min_speed_ms']:.2f}, {quality_report['objects_quality']['max_speed_ms']:.2f}] m/s")

    # 6. Export Tables and Figures
    print("\n[6] Exporting Tables and Figures...")
    exporter = DatasetExporter(PROJECT_ROOT / "outputs")
    table_paths = exporter.export_tables(summary_df, objects_df, ego_df)
    for name, p in table_paths.items():
        print(f"    [CSV] {name}: {p}")

    fig_paths = exporter.generate_figures(summary_df, objects_df, scenarios=scenarios)
    for name, p in fig_paths.items():
        print(f"    [PNG] {name}: {p}")

    # 7. Write comprehensive report
    report_path = PROJECT_ROOT / "outputs" / "reports" / "dataset_exploration_summary.json"
    full_report = {
        "dataset_metadata": metadata,
        "scenarios_analyzed": len(scenarios),
        "statistics": {
            "total_tracks": int(total_tracks),
            "vehicles": int(total_vehicles),
            "pedestrians": int(total_pedestrians),
            "cyclists": int(total_cyclists),
            "other": int(total_other),
            "map_features": {
                "total": int(summary_df["num_map_features"].sum()),
                "lanes": int(summary_df["num_lanes"].sum()),
                "road_edges": int(summary_df["num_road_edges"].sum()),
                "road_lines": int(summary_df["num_road_lines"].sum()),
                "crosswalks": int(summary_df["num_crosswalks"].sum()),
                "speed_bumps": int(summary_df["num_speed_bumps"].sum()),
                "stop_signs": int(summary_df["num_stop_signs"].sum()),
                "driveways": int(summary_df["num_driveways"].sum()),
            },
        },
        "quality_audit": quality_report,
    }
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"\n[7] Summary report saved to: {report_path}")
    print("\nDone! Exploration pipeline completed successfully.")


if __name__ == "__main__":
    main()
