#!/usr/bin/env python3
"""Dedicated CSV export script for Waymo tabular datasets.

Outputs:
- outputs/tables/scenario_summary.csv
- outputs/tables/objects_current.csv
- outputs/tables/ego_current.csv
- outputs/tables/waymo_excel_preview.csv (500-1000 rows)
- outputs/tables/waymo_data_dictionary.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.data.dataset_exporter import DatasetExporter
from src.data.scenario_parser import ScenarioParser
from src.data.tfrecord_reader import iter_scenarios


def main():
    parser = argparse.ArgumentParser(description="Export Waymo Scenario DataFrames to CSVs.")
    parser.add_argument("--max-scenarios", type=int, default=100, help="Max scenarios to parse (default: 100)")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory path")
    parser.add_argument("--preview-rows", type=int, default=800, help="Row count for Excel preview")
    args = parser.parse_args()

    out_path = Path(args.output_dir).resolve()
    print(f"Streaming and exporting up to {args.max_scenarios} scenarios to {out_path}...")

    scenarios = list(iter_scenarios(max_scenarios=args.max_scenarios))
    print(f"Decoded {len(scenarios)} scenarios.")

    s_parser = ScenarioParser()
    summary_df, objects_df, ego_df = s_parser.parse_scenarios(scenarios)

    exporter = DatasetExporter(output_dir=out_path)
    paths = exporter.export_tables(
        summary_df,
        objects_df,
        ego_df,
        preview_max_rows=args.preview_rows,
    )

    print("\nExported CSV Tables:")
    for name, p in paths.items():
        print(f"  - {name}: {p}")


if __name__ == "__main__":
    main()
