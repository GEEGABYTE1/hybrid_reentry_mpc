# Headline Success Summary

| Story Stage | Tier | Controller | Strict | Controlled | Note |
|---|---|---|---:|---:|---|
| Phase 12: residual-learning MPC | moderate | `nominal_nmpc` | 3/30 | not logged | Residual correction did not improve strict success. |
| Phase 12: residual-learning MPC | moderate | `residual_corrected_nmpc` | 3/30 | not logged | Residual correction did not improve strict success. |
| Phase 20: oracle imitation | moderate | `ridge_safety` | 15/30 | 24/30 | Ridge oracle imitation scaled to the full benchmark. |
| Phase 22: actuator-aware slack-MPC | moderate | `online_slack_mpc_centered` | 15/30 | 24/30 | Online slack-first MPC matched the later audited ceiling. |
| Phase 23: audited feasibility ceiling | moderate | `actuator_truth_consistent_oracle` | 15/30 | 24/30 | Non-causal audited ceiling; not an online controller. |
| Phase 24: hybrid imitation + MPC | moderate | `hybrid_blended_slack_mpc` | 15/30 | 24/30 | Learning proposes a warm start/prior; MPC decides. |
| Phase 5: nominal benchmark | moderate | `gain_scheduled_lqr` | 0/30 | not logged | Original fixed Monte Carlo benchmark. |
| Phase 5: nominal benchmark | moderate | `nominal_nmpc` | 3/30 | not logged | Original fixed Monte Carlo benchmark. |
| Phase 5: nominal benchmark | moderate | `pid` | 1/30 | not logged | Original fixed Monte Carlo benchmark. |
| Phase 12: residual-learning MPC | stress | `nominal_nmpc` | 3/30 | not logged | Residual correction did not improve strict success. |
| Phase 12: residual-learning MPC | stress | `residual_corrected_nmpc` | 3/30 | not logged | Residual correction did not improve strict success. |
| Phase 20: oracle imitation | stress | `ridge_safety` | 11/30 | 17/30 | Ridge oracle imitation scaled to the full benchmark. |
| Phase 22: actuator-aware slack-MPC | stress | `online_slack_mpc_centered` | 11/30 | 17/30 | Online slack-first MPC matched the later audited ceiling. |
| Phase 23: audited feasibility ceiling | stress | `actuator_truth_consistent_oracle` | 11/30 | 17/30 | Non-causal audited ceiling; not an online controller. |
| Phase 24: hybrid imitation + MPC | stress | `hybrid_blended_slack_mpc` | 11/30 | 17/30 | Learning proposes a warm start/prior; MPC decides. |
| Phase 5: nominal benchmark | stress | `gain_scheduled_lqr` | 0/30 | not logged | Original fixed Monte Carlo benchmark. |
| Phase 5: nominal benchmark | stress | `nominal_nmpc` | 3/30 | not logged | Original fixed Monte Carlo benchmark. |
| Phase 5: nominal benchmark | stress | `pid` | 2/30 | not logged | Original fixed Monte Carlo benchmark. |
