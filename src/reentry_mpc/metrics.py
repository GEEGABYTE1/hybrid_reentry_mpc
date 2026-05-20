"""Metrics for comparing baseline and learning-augmented controllers."""

from __future__ import annotations

import pandas as pd


def summarize_trajectory(trajectory: pd.DataFrame) -> pd.DataFrame:
    """Summarize each controller trajectory into blog-ready metrics."""

    rows: list[dict[str, float | str]] = []
    for controller, frame in trajectory.groupby("controller", sort=True):
        error = frame["tracking_error_rad"].abs()
        control = frame["control"].abs()
        rows.append(
            {
                "controller": controller,
                "mean_abs_error_rad": float(error.mean()),
                "max_abs_error_rad": float(error.max()),
                "final_abs_error_rad": float(error.iloc[-1]),
                "mean_abs_control": float(control.mean()),
                "max_abs_control": float(control.max()),
            }
        )
    return pd.DataFrame(rows)
