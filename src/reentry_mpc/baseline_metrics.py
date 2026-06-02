from __future__ import annotations

import numpy as np
import pandas as pd


def rms(values: pd.Series) -> float:
    array = values.to_numpy(dtype=float)
    return float(np.sqrt(np.mean(array**2)))


def summarize_closed_loop(
    trajectory: pd.DataFrame,
    *,
    controller: str,
    success_thresholds: dict[str, float],
) -> dict[str, float | int | str]:
    alpha_error = trajectory["alpha_error_rad"].abs()
    q_error = trajectory["q_error_rad"]
    control = trajectory["delta_flap_rad"]
    corridor_violations = int(trajectory["corridor_violation"].sum())
    metrics: dict[str, float | int | str] = {
        "controller": controller,
        "rms_alpha_error_rad": rms(trajectory["alpha_error_rad"]),
        "max_alpha_error_rad": float(alpha_error.max()),
        "rms_pitch_rate_error_radps": rms(q_error),
        "control_effort_abs_rad_s": float(
            np.trapezoid(np.abs(control), trajectory["time_s"])
        ),
        "flap_saturation_fraction": float(trajectory["flap_saturated"].mean()),
        "flap_rate_saturation_fraction": float(
            trajectory["flap_rate_saturated"].mean()
        ),
        "corridor_violation_count": corridor_violations,
    }
    success = (
        metrics["rms_alpha_error_rad"] <= success_thresholds["rms_alpha_error_rad"]
        and metrics["max_alpha_error_rad"] <= success_thresholds["max_alpha_error_rad"]
        and corridor_violations <= int(success_thresholds["corridor_violation_count"])
    )
    metrics["success_label"] = "success" if success else "failure"
    return metrics


def summarize_all(
    rollouts: dict[str, pd.DataFrame], success_thresholds: dict[str, float]
) -> pd.DataFrame:
    rows = [
        summarize_closed_loop(
            trajectory,
            controller=controller,
            success_thresholds=success_thresholds,
        )
        for controller, trajectory in rollouts.items()
    ]
    return pd.DataFrame(rows)
