"""Input/output utilities for reproducible experiments."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return its `Path`."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def timestamp_string() -> str:
    """Return a filesystem-safe timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_run_artifact_dirs(
    results_root: str | Path,
    logs_root: str | Path,
    figures_root: str | Path,
    run_name: str | None = None,
) -> dict[str, Path | str]:
    """Create timestamped directories for one experiment run.

    Args:
        results_root: Base results directory.
        logs_root: Base logs directory.
        figures_root: Base figures directory.
        run_name: Optional custom run name, otherwise timestamp.

    Returns:
        Dictionary with keys: run_id, result_dir, log_dir, figure_dir.
    """
    run_id = run_name or timestamp_string()
    result_dir = ensure_dir(Path(results_root) / run_id)
    log_dir = ensure_dir(Path(logs_root) / run_id)
    figure_dir = ensure_dir(Path(figures_root) / run_id)
    return {
        "run_id": run_id,
        "result_dir": result_dir,
        "log_dir": log_dir,
        "figure_dir": figure_dir,
    }


def save_json(data: dict[str, Any] | list[Any], path: str | Path) -> None:
    """Write JSON data with indentation."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def load_json(path: str | Path) -> Any:
    """Read JSON from disk."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """Save a dataframe to CSV without index."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)


def copy_file(src: str | Path, dst: str | Path) -> None:
    """Copy a file creating parent directories as needed."""
    destination = Path(dst)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, destination)
