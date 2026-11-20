"""TFRecord reader for Waymo Open Motion Dataset scenarios.

Incremental, memory-efficient reading using TensorFlow TFRecordDataset and Waymo
Scenario protobuf definitions.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2


def get_default_data_dir() -> Path:
    """Resolve data directory from environment or standard candidate paths.

    Order of precedence:
    1. Environment variable `WAYMO_DATA_DIR`
    2. Local project directory `data/raw`
    3. User downloads folder `~/Downloads/archive`
    """
    env_dir = os.environ.get("WAYMO_DATA_DIR")
    if env_dir:
        env_path = Path(env_dir).expanduser().resolve()
        if env_path.exists():
            return env_path

    local_raw = Path("data/raw").resolve()
    if local_raw.exists() and any(local_raw.glob("*.tfrecord*")):
        return local_raw

    candidate = Path("~/Downloads/archive").expanduser().resolve()
    if candidate.exists() and any(candidate.glob("*.tfrecord*")):
        return candidate

    return local_raw


def discover_tfrecords(
    data_dir: Optional[Union[str, Path]] = None,
    pattern: str = "*.tfrecord*",
) -> List[Path]:
    """Discover TFRecord files matching the pattern within the data directory.

    Args:
        data_dir: Directory containing TFRecord files. If None, resolves automatically.
        pattern: Glob pattern to identify TFRecord files.

    Returns:
        Sorted list of Path instances pointing to discovered TFRecord files.

    Raises:
        FileNotFoundError: If the resolved directory does not exist or has no matching files.
    """
    target_dir = Path(data_dir).expanduser().resolve() if data_dir else get_default_data_dir()

    if not target_dir.exists():
        raise FileNotFoundError(f"Waymo data directory does not exist: {target_dir}")

    files = sorted(target_dir.glob(pattern))
    # Filter out hidden or non-file entries
    files = [f for f in files if f.is_file() and not f.name.startswith(".")]

    if not files:
        raise FileNotFoundError(
            f"No TFRecord files matching '{pattern}' found in {target_dir}"
        )

    return files


def get_dataset_metadata(
    data_dir: Optional[Union[str, Path]] = None,
) -> dict:
    """Retrieve metadata about local dataset files.

    Returns:
        dict containing:
            - directory: resolved directory Path
            - files: list of (filename, size_in_bytes, size_mb)
            - file_count: total number of files
            - total_bytes: combined size in bytes
            - total_mb: combined size in megabytes
    """
    files = discover_tfrecords(data_dir)
    file_info = []
    total_bytes = 0

    for f in files:
        size = f.stat().st_size
        total_bytes += size
        file_info.append({
            "path": str(f),
            "filename": f.name,
            "bytes": size,
            "size_mb": round(size / (1024 * 1024), 2),
        })

    return {
        "directory": str(files[0].parent),
        "file_count": len(files),
        "files": file_info,
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / (1024 * 1024), 2),
    }


def decode_scenario(raw_bytes: bytes) -> scenario_pb2.Scenario:
    """Decode a single raw protobuf byte string into a Waymo Scenario object.

    Args:
        raw_bytes: Binary serialized protobuf data.

    Returns:
        scenario_pb2.Scenario instance.
    """
    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(raw_bytes)
    return scenario


def iter_scenarios(
    tfrecord_files: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
    max_scenarios: Optional[int] = None,
) -> Generator[scenario_pb2.Scenario, None, None]:
    """Incrementally stream and decode Scenario objects across TFRecord files.

    Streams scenarios one by one without loading the complete dataset into memory.

    Args:
        tfrecord_files: Single file path, list of paths, or None to auto-discover.
        max_scenarios: Maximum scenarios to yield (None for all).

    Yields:
        scenario_pb2.Scenario objects.
    """
    if tfrecord_files is None:
        files = discover_tfrecords()
    elif isinstance(tfrecord_files, (str, Path)):
        files = [Path(tfrecord_files).expanduser().resolve()]
    else:
        files = [Path(f).expanduser().resolve() for f in tfrecord_files]

    yielded_count = 0

    for file_path in files:
        if not file_path.exists():
            continue

        raw_dataset = tf.data.TFRecordDataset(str(file_path))
        for raw_record in raw_dataset:
            scenario = decode_scenario(raw_record.numpy())
            yield scenario
            yielded_count += 1
            if max_scenarios is not None and yielded_count >= max_scenarios:
                return


def read_scenarios_from_file(
    file_path: Union[str, Path],
    max_scenarios: Optional[int] = None,
) -> List[scenario_pb2.Scenario]:
    """Read scenarios from a single TFRecord file into a list.

    Note: Use iter_scenarios for streaming to conserve memory on large datasets.
    """
    return list(iter_scenarios(file_path, max_scenarios=max_scenarios))
