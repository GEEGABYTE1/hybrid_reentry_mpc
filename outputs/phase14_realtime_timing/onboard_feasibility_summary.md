# Phase 14 Real-Time Feasibility Summary

Budgets:
- 10 Hz: 100 ms
- 20 Hz: 50 ms
- 50 Hz: 20 ms

Configurations meeting p95 total-loop budgets:
- 10 Hz: 43
- 20 Hz: 26
- 50 Hz: 18

Fastest configurations by p95 total-loop time:

| controller | horizon_steps | control_frequency_hz | p95_total_loop_time_ms | solver_failure_rate | success_rate |
| --- | --- | --- | --- | --- | --- |
| gain_scheduled_lqr | 15 | 50 | 0.37275 | 0 | 0 |
| gain_scheduled_lqr | 10 | 20 | 0.373217 | 0 | 0 |
| pid | 10 | 50 | 0.377975 | 0 | 0.05 |
| gain_scheduled_lqr | 15 | 10 | 0.38305 | 0 | 0 |
| gain_scheduled_lqr | 5 | 20 | 0.383408 | 0 | 0 |
| pid | 15 | 50 | 0.3886 | 0 | 0.05 |
| gain_scheduled_lqr | 5 | 10 | 0.415066 | 0 | 0 |
| gain_scheduled_lqr | 10 | 50 | 0.416792 | 0 | 0 |

Warm-start note: warm starts are not implemented in this benchmark yet; all rows use `warm_start=false` and `warm_start_implemented=false`.