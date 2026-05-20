# Outline

## Working Title

Learning-Augmented MPC for Reentry Attitude Control

## Thesis

Learning can improve an MPC controller when it is treated as an uncertainty-aware augmentation rather than a replacement for the physics and constraints that make MPC useful.

## Structure

1. Problem framing: why reentry attitude control is unforgiving.
2. Baseline control setup: dynamics, constraints, and tracking objectives.
3. Where nominal MPC struggles: model mismatch and disturbance structure.
4. Learning augmentation: residual prediction, bias correction, or adaptive cost shaping.
5. Reproducible experiment loop: configs, seeds, metrics, figures, and logs.
6. Smoke result: baseline vs learning-augmented toy model.
7. What would be required for a serious flight-dynamics claim.
8. Takeaways and next experiments.

## Evidence Needed

- Baseline trajectory metrics.
- Learning-augmented trajectory metrics.
- Control-effort comparison.
- Robustness sweep over seeds and disturbance scales.
- Explicit limitations for toy vs high-fidelity model results.
