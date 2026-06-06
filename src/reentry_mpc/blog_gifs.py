from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

from reentry_mpc.artifacts import plt


def generate_blog_gifs(
    output_dir: str | Path = "outputs/blog_gifs",
    *,
    phase2_reference_csv: str | Path = "outputs/phase2_reference/reference_profile.csv",
    rollout_csv: str | Path = "outputs/phase7_scenario_mpc/phase7_rollouts.csv",
    fps: int = 12,
    max_frames: int = 96,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(phase2_reference_csv)
    rollouts = pd.read_csv(rollout_csv)
    profile_path = output_path / "reentry_profile_dynamic_pressure.gif"
    corridor_path = output_path / "alpha_corridor_replay.gif"
    write_reentry_profile_gif(reference, profile_path, fps=fps, max_frames=max_frames)
    representative = select_representative_rollout(rollouts)
    write_alpha_corridor_replay_gif(
        representative, corridor_path, fps=fps, max_frames=max_frames
    )
    return {
        "reentry_profile_gif": profile_path,
        "alpha_corridor_replay_gif": corridor_path,
    }


def select_representative_rollout(rollouts: pd.DataFrame) -> pd.DataFrame:
    """Select a deterministic rollout with large alpha error for blog replay."""

    grouped = (
        rollouts.groupby(["tier", "scenario_id", "controller"], as_index=False)[
            "alpha_error_rad"
        ]
        .apply(lambda values: float(values.abs().max()))
        .rename(columns={"alpha_error_rad": "max_alpha_error_rad"})
        .sort_values(
            ["max_alpha_error_rad", "tier", "scenario_id"],
            ascending=[False, True, True],
        )
    )
    row = grouped.iloc[0]
    selected = rollouts[
        (rollouts["tier"] == row["tier"])
        & (rollouts["scenario_id"] == int(row["scenario_id"]))
        & (rollouts["controller"] == row["controller"])
    ].copy()
    return selected.sort_values("time_s").reset_index(drop=True)


def write_reentry_profile_gif(
    reference: pd.DataFrame,
    path: str | Path,
    *,
    fps: int = 12,
    max_frames: int = 96,
) -> Path:
    frames = _frame_indices(len(reference), max_frames)
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.6))
    ax_profile, ax_qbar, ax_alpha, ax_status = axes.ravel()

    altitude_km = reference["altitude_m"] / 1000.0
    qbar_kpa = reference["dynamic_pressure_pa"] / 1000.0
    ax_profile.plot(reference["mach"], altitude_km, color="#1f77b4", lw=2)
    profile_marker = ax_profile.scatter([], [], s=58, color="#d62728", zorder=3)
    ax_profile.set_xlabel("Mach")
    ax_profile.set_ylabel("Altitude (km)")
    ax_profile.set_title("Scheduled Entry Profile")
    ax_profile.invert_xaxis()

    ax_qbar.plot(reference["time_s"], qbar_kpa, color="#2ca02c", lw=2)
    qbar_marker = ax_qbar.axvline(reference["time_s"].iloc[0], color="#d62728", lw=2)
    ax_qbar.set_xlabel("Time (s)")
    ax_qbar.set_ylabel("Dynamic pressure (kPa)")
    ax_qbar.set_title("Dynamic Pressure Builds With Descent")

    ax_alpha.fill_between(
        reference["time_s"],
        reference["alpha_min_rad"],
        reference["alpha_max_rad"],
        color="0.82",
        label="corridor",
    )
    ax_alpha.plot(reference["time_s"], reference["alpha_ref_rad"], color="black", lw=2)
    alpha_marker = ax_alpha.scatter([], [], s=54, color="#d62728", zorder=3)
    ax_alpha.set_xlabel("Time (s)")
    ax_alpha.set_ylabel("Alpha (rad)")
    ax_alpha.set_title("Reference Angle-of-Attack Corridor")

    ax_status.axis("off")
    status = ax_status.text(0.02, 0.82, "", fontsize=13, va="top", family="monospace")

    def update(frame_idx: int) -> tuple:
        idx = int(frame_idx)
        row = reference.iloc[idx]
        profile_marker.set_offsets([[row["mach"], row["altitude_m"] / 1000.0]])
        qbar_marker.set_xdata([row["time_s"], row["time_s"]])
        alpha_marker.set_offsets([[row["time_s"], row["alpha_ref_rad"]]])
        status.set_text(
            (
                "t = {time:6.1f} s\n"
                "alt = {alt:6.1f} km\n"
                "Mach = {mach:5.2f}\n"
                "qbar = {qbar:5.2f} kPa"
            ).format(
                time=row["time_s"],
                alt=row["altitude_m"] / 1000.0,
                mach=row["mach"],
                qbar=row["dynamic_pressure_pa"] / 1000.0,
            )
        )
        return profile_marker, qbar_marker, alpha_marker, status

    animation = FuncAnimation(fig, update, frames=frames, blit=False)
    fig.tight_layout()
    animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return Path(path)


def write_alpha_corridor_replay_gif(
    rollout: pd.DataFrame,
    path: str | Path,
    *,
    fps: int = 12,
    max_frames: int = 96,
) -> Path:
    frames = _frame_indices(len(rollout), max_frames)
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.2), sharex=True)
    ax_alpha, ax_flap = axes
    ax_alpha.fill_between(
        rollout["time_s"],
        rollout["alpha_min_rad"],
        rollout["alpha_max_rad"],
        color="0.84",
        label="alpha corridor",
    )
    (alpha_line,) = ax_alpha.plot([], [], color="#1f77b4", lw=2, label="alpha")
    (ref_line,) = ax_alpha.plot([], [], color="black", ls="--", lw=1.8, label="ref")
    alpha_marker = ax_alpha.scatter([], [], s=48, color="#d62728", zorder=3)
    ax_alpha.set_ylabel("Alpha (rad)")
    title = "{controller} | {tier} scenario {scenario}".format(
        controller=rollout["controller"].iloc[0],
        tier=rollout["tier"].iloc[0],
        scenario=int(rollout["scenario_id"].iloc[0]),
    )
    ax_alpha.set_title(f"Angle-of-Attack Corridor Replay: {title}")
    ax_alpha.legend(loc="best")

    (flap_line,) = ax_flap.plot([], [], color="#9467bd", lw=2)
    flap_marker = ax_flap.scatter([], [], s=48, color="#d62728", zorder=3)
    ax_flap.axhline(0.0, color="0.35", lw=1)
    ax_flap.set_xlabel("Time (s)")
    ax_flap.set_ylabel("Flap (rad)")

    def update(frame_idx: int) -> tuple:
        idx = int(frame_idx)
        frame = rollout.iloc[: idx + 1]
        row = rollout.iloc[idx]
        alpha_line.set_data(frame["time_s"], frame["alpha_rad"])
        ref_line.set_data(frame["time_s"], frame["alpha_ref_rad"])
        alpha_marker.set_offsets([[row["time_s"], row["alpha_rad"]]])
        flap_line.set_data(frame["time_s"], frame["delta_flap_rad"])
        flap_marker.set_offsets([[row["time_s"], row["delta_flap_rad"]]])
        ax_alpha.set_xlim(rollout["time_s"].min(), rollout["time_s"].max())
        ax_alpha.set_ylim(
            min(rollout["alpha_min_rad"].min(), rollout["alpha_rad"].min()) - 0.03,
            max(rollout["alpha_max_rad"].max(), rollout["alpha_rad"].max()) + 0.03,
        )
        ax_flap.set_ylim(
            rollout["delta_flap_rad"].min() - 0.05,
            rollout["delta_flap_rad"].max() + 0.05,
        )
        return alpha_line, ref_line, alpha_marker, flap_line, flap_marker

    animation = FuncAnimation(fig, update, frames=frames, blit=False)
    fig.tight_layout()
    animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return Path(path)


def _frame_indices(length: int, max_frames: int) -> np.ndarray:
    frame_count = min(max(int(max_frames), 2), length)
    return np.unique(np.linspace(0, length - 1, frame_count, dtype=int))
