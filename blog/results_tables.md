# Results Tables

## T-001: Smoke Summary Metrics

Source:

- CSV: `outputs/metrics/smoke_summary.csv`
- JSON: `outputs/metrics/smoke_summary.json`
- Config: `configs/smoke.yaml`

Columns:

| Column | Meaning |
|---|---|
| `controller` | Controller variant name. |
| `mean_abs_error_rad` | Mean absolute attitude tracking error. |
| `max_abs_error_rad` | Maximum absolute attitude tracking error. |
| `final_abs_error_rad` | Final absolute attitude tracking error. |
| `mean_abs_control` | Mean absolute control command. |
| `max_abs_control` | Maximum absolute control command. |

Blog use: summarize only after rerunning the smoke command and confirming values.
