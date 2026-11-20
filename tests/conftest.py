"""Pytest fixtures and configuration."""

import os
import sys
from pathlib import Path

import pytest

# Ensure project root and src are on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.data.tfrecord_reader import discover_tfrecords, iter_scenarios


@pytest.fixture(scope="session")
def sample_scenarios():
    """Load a small batch of real scenarios for integration tests."""
    return list(iter_scenarios(max_scenarios=3))
