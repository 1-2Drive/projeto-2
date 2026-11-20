"""Tests for CSV table exports and figure generation."""

from pathlib import Path
import pandas as pd
import pytest
from src.data.dataset_exporter import DatasetExporter
from src.data.scenario_parser import ScenarioParser


def test_export_tables(tmp_path, sample_scenarios):
    parser = ScenarioParser()
    summary_df, objects_df, ego_df = parser.parse_scenarios(sample_scenarios)

    exporter = DatasetExporter(output_dir=tmp_path)
    paths = exporter.export_tables(summary_df, objects_df, ego_df, preview_max_rows=50)

    expected_keys = {
        "scenario_summary",
        "objects_current",
        "ego_current",
        "waymo_excel_preview",
        "waymo_data_dictionary",
    }
    assert set(paths.keys()) == expected_keys

    for k, p in paths.items():
        assert Path(p).exists()
        loaded = pd.read_csv(p)
        assert len(loaded) > 0


def test_generate_figures(tmp_path, sample_scenarios):
    parser = ScenarioParser()
    summary_df, objects_df, ego_df = parser.parse_scenarios(sample_scenarios)

    exporter = DatasetExporter(output_dir=tmp_path)
    fig_paths = exporter.generate_figures(summary_df, objects_df, scenarios=sample_scenarios)

    expected_figs = {
        "object_distribution",
        "speed_distribution",
        "objects_per_scenario",
        "topdown_scenario",
    }
    assert set(fig_paths.keys()) == expected_figs

    for k, p in fig_paths.items():
        assert Path(p).exists()
        assert Path(p).stat().st_size > 1000  # valid image file
