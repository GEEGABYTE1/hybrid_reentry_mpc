from pathlib import Path

from reentry_mpc.pipeline import run_smoke_experiment


def test_smoke_pipeline_writes_reproducible_artifacts(tmp_path: Path) -> None:
    artifacts = run_smoke_experiment(
        config_path="configs/smoke.yaml",
        output_dir=tmp_path,
    )

    assert artifacts["trajectory_csv"].exists()
    assert artifacts["summary_csv"].exists()
    assert artifacts["summary_json"].exists()
    assert artifacts["tracking_figure"].exists()
    assert artifacts["blog_log"].exists()

    summary = artifacts["summary"]
    assert set(summary["controller"]) == {"baseline_pd", "learning_augmented_pd"}
    assert (summary["final_abs_error_rad"] >= 0).all()
