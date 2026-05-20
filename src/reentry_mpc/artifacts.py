"""Artifact writers for CSV, JSON, figures, and blog logs."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "reentry_mpc_matplotlib"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_MPL_CACHE_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def ensure_output_dirs(output_dir: str | Path) -> dict[str, Path]:
    """Create and return the standard artifact directories."""

    root = Path(output_dir)
    dirs = {
        "root": root,
        "figures": root / "figures",
        "metrics": root / "metrics",
        "logs": root / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def write_metrics(
    trajectory: pd.DataFrame, summary: pd.DataFrame, artifact_dirs: dict[str, Path]
) -> dict[str, Path]:
    """Write reproducible numeric artifacts."""

    trajectory_path = artifact_dirs["metrics"] / "smoke_trajectory.csv"
    summary_csv_path = artifact_dirs["metrics"] / "smoke_summary.csv"
    summary_json_path = artifact_dirs["metrics"] / "smoke_summary.json"

    trajectory.to_csv(trajectory_path, index=False)
    summary.to_csv(summary_csv_path, index=False)
    summary.to_json(summary_json_path, orient="records", indent=2)

    return {
        "trajectory_csv": trajectory_path,
        "summary_csv": summary_csv_path,
        "summary_json": summary_json_path,
    }


def write_figures(
    trajectory: pd.DataFrame, artifact_dirs: dict[str, Path]
) -> dict[str, Path]:
    """Write deterministic smoke-test figures."""

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for controller, frame in trajectory.groupby("controller", sort=True):
        axes[0].plot(frame["time_s"], frame["angle_rad"], label=controller)
        axes[1].plot(frame["time_s"], frame["control"], label=controller)

    axes[0].plot(
        trajectory["time_s"].unique(),
        trajectory.groupby("time_s")["target_angle_rad"].first(),
        color="black",
        linestyle="--",
        linewidth=1,
        label="target",
    )
    axes[0].set_ylabel("Angle (rad)")
    axes[1].set_ylabel("Control torque")
    axes[1].set_xlabel("Time (s)")
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    fig.tight_layout()

    figure_path = artifact_dirs["figures"] / "smoke_attitude_tracking.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return {"tracking_figure": figure_path}


def append_blog_log(
    artifact_dirs: dict[str, Path],
    *,
    phase: str,
    seed: int,
    metrics: pd.DataFrame,
) -> Path:
    """Append a compact machine-readable blog log entry."""

    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "seed": seed,
        "metrics": metrics.to_dict(orient="records"),
        "notes": (
            "Generated deterministic smoke artifacts for baseline vs "
            "learning-augmented controller comparison."
        ),
    }
    log_path = artifact_dirs["logs"] / "blog_log.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return log_path
