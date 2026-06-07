from pathlib import Path

import pandas as pd

from reentry_mpc.blog_gifs import (
    select_representative_rollout,
    write_alpha_corridor_replay_gif,
    write_reentry_profile_gif,
)


def test_select_representative_rollout_uses_largest_alpha_error() -> None:
    rollouts = pd.DataFrame(
        {
            "tier": ["a", "a", "b", "b"],
            "scenario_id": [0, 0, 1, 1],
            "controller": ["c", "c", "c", "c"],
            "time_s": [0.0, 1.0, 0.0, 1.0],
            "alpha_error_rad": [0.1, 0.2, 0.1, 0.4],
        }
    )

    selected = select_representative_rollout(rollouts)

    assert set(selected["tier"]) == {"b"}
    assert set(selected["scenario_id"]) == {1}


def test_blog_gif_writers_create_files(tmp_path: Path) -> None:
    reference = pd.read_csv("outputs/phase2_reference/reference_profile.csv").head(12)
    rollout = pd.read_csv("outputs/phase7_scenario_mpc/phase7_rollouts.csv").head(12)
    profile_path = tmp_path / "profile.gif"
    replay_path = tmp_path / "replay.gif"

    write_reentry_profile_gif(reference, profile_path, fps=4, max_frames=6)
    write_alpha_corridor_replay_gif(rollout, replay_path, fps=4, max_frames=6)

    assert profile_path.exists()
    assert profile_path.stat().st_size > 0
    assert replay_path.exists()
    assert replay_path.stat().st_size > 0
