# Fault Injection Limitations

- Fault cases are deterministic stress probes, not statistically calibrated failure probabilities.
- The stuck-flap and actuator-delay models are simplified actuator-level approximations.
- Safe-mode LQR fallback is implemented after repeated solver failures, but most current failures are corridor/dynamics failures rather than solver failures.
- Constraint tightening is triggered from measured alpha residual error; it is not a formal fault detector.

Failed fault/controller rows: 12 of 12.