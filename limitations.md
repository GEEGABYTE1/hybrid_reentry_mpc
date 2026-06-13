# Limitations

- The initial smoke simulation is a deterministic toy model, not a validated six-degree-of-freedom reentry simulator.
- The current "learning-augmented" controller uses a simple disturbance-bias term rather than a trained residual model.
- Phase 4 solves hard actuator constraints through a true MPC optimizer, but state corridor constraints are softened rather than guaranteed hard constraints.
- Aerothermal, actuator, sensor, navigation, and flexible-body effects are out of scope for the scaffold.
- Blog claims must distinguish repository mechanics from control-theoretic or flight-dynamics conclusions.
- Phase 1 schedules altitude and velocity rather than propagating full translational reentry dynamics.
- The Phase 1 standard atmosphere is a compact approximation and does not model high-altitude composition, real-gas effects, winds, or vehicle heating.
- The Phase 1 aerodynamic pitching-moment model is coefficient-level and illustrative; coefficients are not identified from a specific vehicle dataset.
- Phase 3 PID and gain-scheduled LQR are baseline controllers only; their failure on the current corridor does not prove that PID or LQR cannot work on better-tuned or different formulations.
- Phase 3 LQR uses local finite-difference linearizations and nearest-neighbor gain scheduling, not a full time-varying constrained optimal controller.
- Phase 4 is nominal NMPC, not robust MPC; it optimizes against the current reduced-order model without uncertainty or learned residuals.
- Phase 4 softens alpha and pitch-rate corridor constraints with logged violations to keep the NLP solvable on the current aggressive corridor; flap angle and flap-rate limits remain hard.
- Phase 4's success label uses a small `0.001 rad` corridor-counting tolerance and a feasible nonzero initial pitch-rate condition; it should not be read as proof of broad reentry feasibility.
- Phase 5 is an illustrative two-tier Monte Carlo benchmark with 30 moderate scenarios and 30 stress scenarios. It estimates behavior under configured uncertainty ranges but does not provide statistical certification or flight robustness.
- Phase 5 keeps controllers nominal. NMPC does not know the sampled perturbed plant, so results describe nominal-controller degradation under mismatch rather than robust MPC.
- The Phase 5 stress tier intentionally pushes the reduced-order model into harsher uncertainty. Its unstable-response labels are useful for motivating robust/learning-augmented control, but they should not be interpreted as a validated physical instability boundary.
- Phase 6 is only a first robust-MPC-style experiment. Constraint tightening adds conservative planning margins, but it does not model uncertainty dynamics, optimize over sampled scenarios, or learn residual prediction error.
- Phase 6 does not improve binary success rate over nominal NMPC on the current benchmark, so it should be presented as a useful negative/partial result rather than a solved robust-control phase.
- Phase 7 scenario NMPC is evaluated on a 12-scenario-per-tier subset for runtime reasons. Its comparison table filters prior results to the same subset, but it should not be described as a full 30-scenario-per-tier replacement for Phase 5/6.
- Phase 7 increases optimizer complexity and currently introduces solver-failure labels, especially in the stress tier. This is a controller-reliability limitation, not just a tracking limitation.
- Blog GIFs are explanatory animations generated from existing CSV artifacts. They are useful for reader intuition, but they are not photorealistic entry visuals, not six-degree-of-freedom animations, and not new validation evidence.
- Phase 8 residual data is generated from the project's perturbed reduced-order model, not from high-fidelity CFD, wind tunnel data, flight telemetry, or a validated 6DOF simulator.
- Phase 9 trains an open-loop derivative residual predictor only. It is not yet embedded inside MPC, and it does not prove closed-loop tracking, robustness, stability, or constraint-satisfaction improvement.
- Phase 10 uses a local equivalent-moment correction from the learned residual model, not a full residual dynamics model embedded throughout the CasADi horizon. Its negative result should motivate better integration, not be treated as a proof that residual learning cannot help.
- Phase 11 embeds a polynomial residual surrogate throughout the horizon, but the surrogate is still trained on synthetic reduced-order mismatch and can be miscalibrated out of distribution. The Phase 11 negative result should not be read as a general failure of learning-augmented MPC.
- Phase 12 uses a horizon-scheduled PyTorch residual approximation: the neural model predicts fixed `q_dot` corrections before each CasADi solve, but those corrections are not symbolic functions of the optimizer's decision-state/control trajectory. This keeps the benchmark practical and auditable, but it is not full differentiable neural MPC.
- Phase 12 does not improve binary success rate over nominal NMPC on the current full paired benchmark. Its small RMS-alpha improvements for the tightened variant should not be described as a robustness breakthrough because alpha-corridor violation remains the dominant failure label.
- Phase 13 is diagnostic only. Its corridor-expansion metric measures how far failed trajectories miss the existing corridor; it does not validate changing the corridor, and it does not prove whether failures are physically infeasible versus poorly controlled.
- Phase 14 timing is measured on the local Python/CasADi implementation, not on flight hardware or generated embedded code. Warm starts are not implemented, and the benchmark times representative solve calls rather than full hardware-in-the-loop execution.
- Phase 15 fault injection uses deterministic simplified fault models. The fallback logic is logged, but the first policy does not recover strict success under the tested faults. This should be presented as a fault-tolerance gap, not a completed fault-tolerant controller.
- Phase 16 is a controller-side success-recovery harness, not a changed benchmark. The staged 12-scenario-per-tier result does not improve binary success, and it should not be presented as a solved controller-tuning phase.
- Phase 17 introduces controlled recovery as a secondary metric. It must not be confused with strict Phase 5 success, which remains zero in the saved Phase 17 probe.
- Phase 18 is a non-causal slack oracle. It estimates a feasibility ceiling but does not represent deployable MPC performance.
- Phase 19 is a first oracle-imitation policy trained on synthetic reduced-order oracle trajectories. Its six-scenario-per-tier result is preliminary, and matching the moderate oracle ceiling does not prove robust success on the full benchmark.
- Phase 20 scales oracle imitation to the full configured benchmark, but the oracle is still non-causal and reduced-order. Matching the aggregate moderate oracle count does not mean every oracle-feasible scenario is solved.
- Phase 21 replays non-causal oracle commands as a diagnostic. It does not prove no online controller can recover the missed cases; it only shows that open-loop oracle-command transfer through the uncertain plant does not recover strict success for the selected Phase 20 misses.
- Phase 22 uses an actuator-aware slack-first MPC objective, but still relies on the reduced-order synthetic plant and nominal prediction model. Matching Phase 20 does not prove robust reentry control or close the remaining strict corridor gap.
- Phase 23 audits the feasibility ceiling using a truth-consistent reduced-order oracle, but it is still non-causal and not a formal reachability proof for a validated vehicle.
- Phase 24 uses a simple ridge oracle-imitation policy as warm start/prior for MPC. It demonstrates a clean learning-augmented architecture but does not improve strict success beyond the audited ceiling.
- Phase 25 packages results and creates GIFs from saved reduced-order rollout data. The animations are explanatory media, not high-fidelity reentry visuals or validation evidence.

## Limitation Tracking Template

| ID | Limitation | Affected Claim/Figure | Mitigation |
|---|---|---|---|
| L-001 | Toy single-axis dynamics | C-001, C-002 | Label results as smoke-test artifacts until richer models are added. |
| L-002 | Scheduled translational variables | F-002, F-003, F-004 | Treat Phase 1 as an attitude-subsystem simulator, not a full reentry trajectory solver. |
| L-003 | Illustrative aero coefficients | F-002, F-004 | Replace with traceable coefficients or uncertainty ranges before making vehicle-specific claims. |
| L-004 | Unconstrained/simple baseline controllers | C-003, C-004, F-009, F-010 | Use results to motivate constrained MPC, not to make universal claims about PID or LQR. |
| L-005 | Nominal soft-constrained NMPC with tolerance-counted corridor success | C-005, C-006, F-011, F-012, F-013, F-014 | Treat Phase 4 as the first predictive-control baseline, not a final constraint-satisfaction or robustness result. |
| L-006 | Small illustrative Monte Carlo benchmark | C-007, C-008, C-009, F-015, F-016, F-017, F-018, F-019 | Use Phase 5 to discuss failure modes and robustness pressure, not to claim certified controller robustness. |
| L-007 | Constraint tightening is not enough in the current benchmark | C-010, C-011, F-020, F-021, F-022, F-023, F-024 | Treat Phase 6 as the first robust-control probe and use it to motivate scenario MPC or learned residual correction. |
| L-008 | Scenario NMPC subset size and solver reliability | C-012, C-013, F-025, F-026, F-027, F-028, F-029 | Treat Phase 7 as a first scenario-MPC probe; improve warm-starting and scenario selection before making stronger robustness claims. |
| L-009 | Blog animations are explanatory media, not new validation | A-001, A-002 | Generate animations from saved CSV artifacts and keep captions explicit about their illustrative role. |
| L-010 | Residual-learning data is synthetic reduced-order mismatch | C-014, F-030, F-031, F-032, F-033, F-034, F-035, F-036 | Use the dataset/model to motivate learning augmentation, not to claim vehicle-valid residual identification. |
| L-011 | Residual model is not yet closed-loop control evidence | C-014, F-034, F-035, F-036 | Evaluate learned residuals inside MPC rollouts before making control-performance claims. |
| L-012 | First learned-residual MPC integration is local and limited | C-015, F-037, F-038, F-039, F-040, F-041, F-042 | Treat Phase 10 as a first integration probe; future work should embed residual corrections consistently across the horizon. |
| L-013 | Horizon-embedded residual surrogate does not solve the current corridor benchmark | C-016, F-043, F-044, F-045, F-046, F-047, F-048, F-049 | Treat Phase 11 as a negative calibration/feasibility signal; diagnose corridor feasibility and residual scaling before claiming learning-augmented control success. |
| L-014 | Horizon-scheduled PyTorch residual correction is not full symbolic neural MPC and does not improve success labels | C-017, F-050, F-051, F-052, F-053, F-054 | Present Phase 12 as a fair consolidated benchmark and use the result to motivate feasibility/tuning diagnosis rather than stronger learning claims. |
| L-015 | Feasibility diagnostics measure corridor misses but do not prove true reachability | C-018, F-055, F-056, F-057, F-058 | Treat Phase 13 as failure analysis; a true feasibility phase would solve reachability or optimization problems per scenario. |
| L-016 | Real-time timing is local Python/CasADi timing without warm starts or embedded code generation | C-019, F-059, F-060, F-061 | Present Phase 14 as a practical software benchmark and motivation for warm starts/code generation, not as onboard certification. |
| L-017 | Fault injection models and fallbacks are first-pass deterministic probes | C-020, F-062, F-063, F-064 | Treat Phase 15 as evidence of where fallback still breaks; future work needs formal fault detection, recovery policy design, and stochastic fault campaigns. |
| L-018 | Phase 16 staged sweep is not the full 30-scenario benchmark and does not recover success | C-021, F-065, F-066, F-067, F-068, F-069, F-070 | Treat Phase 16 as evidence that simple faster-update/corridor-weight tuning is insufficient. Use diagnostics to motivate feasibility-aware reference/corridor design or a different constrained-control strategy. |
| L-019 | Phase 17 controlled recovery is a bounded-miss diagnostic, not strict success | C-022, F-071, F-072, F-073, F-074 | Keep strict and controlled metrics separate. Scale the probe and add a formal feasibility oracle before claiming robust success. |
| L-020 | Phase 18 oracle feasibility is optimistic and non-causal | C-023, F-075, F-076, F-077 | Use the oracle to guide controller design and estimate ceilings, not to claim online success. |
| L-021 | Phase 19 oracle imitation is preliminary and trained on synthetic oracle data | C-024, F-078, F-079, F-080 | Treat Phase 19 as an online imitation probe. Scale to the full benchmark and compare against paired NMPC variants before making stronger claims. |
| L-022 | Phase 20 full benchmark still uses a non-causal reduced-order oracle ceiling | C-025, F-081, F-082, F-083, F-084, F-085 | Present Phase 20 as a scaling and diagnosis artifact. Use missed-feasible diagnostics before claiming controller robustness. |
| L-023 | Phase 21 oracle-command replay is diagnostic and non-causal | C-026, F-086, F-087, F-088 | Treat Phase 21 as evidence about transfer mismatch and imitation limits, not as proof that no closed-loop hybrid controller can recover the missed cases. |
| L-024 | Phase 22 slack-MPC matches Phase 20 but does not improve strict success | C-027, F-089, F-090, F-091, F-092, F-093, F-094 | Treat Phase 22 as a closed-loop slack-objective diagnostic. The next mitigation is an actuator-consistent feasibility oracle or hybrid warm-start/safety-filter design. |
| L-025 | Phase 23 audited ceiling is non-causal and reduced-order | C-028, F-095, F-096, F-097, F-098 | Use Phase 23 to explain the benchmark plateau, not to claim formal reachability or vehicle-valid feasibility. |
| L-026 | Phase 24 hybrid learning does not improve strict success | C-029, F-099, F-100, F-101, F-102, F-103, F-104 | Present Phase 24 as architecture/timing evidence: learning proposes and MPC decides, but the audited strict ceiling remains unchanged. |
| L-027 | Phase 25 GIFs are explanatory media, not validation evidence | C-030, A-003, A-004, A-005 | Describe the controller-comparison GIF as a stylized data-driven animation, not an actual flight rendering, CFD/aerothermal simulation, or proof of controller validity. |
