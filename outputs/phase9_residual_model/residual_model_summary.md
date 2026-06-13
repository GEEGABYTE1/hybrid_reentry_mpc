# Residual Model Summary

The Phase 9 MLP residual model predicts `x_dot_true - x_dot_nominal`
from scheduled state/control/aerodynamic features.

| Model | Test MSE | Test MAE | Normalized RMSE |
|---|---:|---:|---:|
| MLP residual | 4.474398e-09 | 2.994068e-05 | 0.9583 |
| Zero residual | 5.058987e-09 | 2.979640e-05 | 1.0190 |

Improves over zero residual by MSE: `True`.

MSE improvement fraction: `0.1156`.

Interpretation: a positive improvement means the network learned
repeatable structure in the sampled model mismatch. It does not mean
the residual model is safe to close the loop around without further
rollout testing.
