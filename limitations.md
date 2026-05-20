# Limitations

- The initial smoke simulation is a deterministic toy model, not a validated six-degree-of-freedom reentry simulator.
- The current "learning-augmented" controller uses a simple disturbance-bias term rather than a trained residual model.
- No hard state or input constraints are solved through a true MPC optimizer yet.
- Aerothermal, actuator, sensor, navigation, and flexible-body effects are out of scope for the scaffold.
- Blog claims must distinguish repository mechanics from control-theoretic or flight-dynamics conclusions.

## Limitation Tracking Template

| ID | Limitation | Affected Claim/Figure | Mitigation |
|---|---|---|---|
| L-001 | Toy single-axis dynamics | C-001, C-002 | Label results as smoke-test artifacts until richer models are added. |
