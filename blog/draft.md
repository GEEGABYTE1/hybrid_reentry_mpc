# Learning-Augmented MPC for Reentry Attitude Control

Atmospheric reentry attitude control sits in an awkward part of the control-design map: the vehicle is fast, the environment changes quickly, and small attitude errors can create large downstream consequences. MPC is attractive because it can reason about constraints and future behavior, but a nominal model can still miss structured effects in the real trajectory.

This project explores a learning-augmented approach: keep the controller anchored in a model-based control loop, then add learned structure where repeatable model mismatch appears.

The current repository is intentionally at scaffold stage. The first runnable artifact is a deterministic single-axis smoke simulation that compares a baseline controller with a learning-augmented variant. It is designed to test the research workflow: config in, metrics and figures out, claim registry updated before prose hardens.

## Current Result Placeholder

After running the smoke command, summarize:

- Tracking error from `outputs/metrics/smoke_summary.csv`.
- Control effort from `outputs/metrics/smoke_summary.csv`.
- Figure F-001 from `plots_manifest.md`.
- Caveats from `limitations.md`.

## Drafting Rule

Any numerical statement in this draft should point back to `claims_register.md`, a metrics file, and a generated figure or table when applicable.
