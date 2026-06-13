# Phase 3 Baseline Metrics Summary

PID and gain-scheduled LQR were evaluated on the shared Phase 2 reference/corridor profile.

| controller | rms_alpha_error_rad | max_alpha_error_rad | rms_pitch_rate_error_radps | control_effort_abs_rad_s | flap_saturation_fraction | flap_rate_saturation_fraction | corridor_violation_count | success_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pid | 0.199993 | 0.276024 | 0.00793694 | 38.7139 | 0.286604 | 0 | 316 | failure |
| gain_scheduled_lqr | 0.220976 | 0.282199 | 0.00137512 | 40.3582 | 0.635514 | 0.610592 | 316 | failure |

Success labels use the thresholds in `configs/phase3_baselines.yaml`.
