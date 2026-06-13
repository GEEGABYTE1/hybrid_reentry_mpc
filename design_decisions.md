# Design Decisions

## DD-001: Use a src-layout Python Package

Decision: source code lives under `src/reentry_mpc`.

Rationale: src-layout packaging catches import-path mistakes earlier and keeps tests closer to installed-package behavior.

## DD-002: Artifact-First Research Workflow

Decision: experiments must write CSV/JSON metrics, figures, and a compact blog log entry.

Rationale: the blog is easier to audit when every result is connected to a file path and configuration.

## DD-003: Deterministic Smoke Experiment

Decision: the initial smoke experiment uses a seeded single-axis toy attitude model and two PD-style controllers.

Rationale: the first milestone should validate repository mechanics without depending on a full vehicle model, optimizer stack, or learned model training loop.

## DD-004: Keep Generated Outputs in a Dedicated Tree

Decision: generated artifacts live under `outputs/metrics`, `outputs/figures`, and `outputs/logs`.

Rationale: this keeps reproducibility evidence discoverable and separates generated data from source code and prose.

## DD-005: Phase 1 Uses Scheduled Translational Variables

Decision: the Phase 1 simulator integrates only longitudinal attitude states `alpha_rad`, `q_radps`, and `theta_rad`, while altitude, velocity, Mach, density, and dynamic pressure are scheduled from a reference trajectory.

Rationale: this keeps the first simulator small enough to test thoroughly while exposing the reentry-specific dependence of attitude dynamics on atmospheric and Mach schedules.

## DD-006: Fixed-Step RK4 for Phase 1

Decision: Phase 1 uses fixed-step RK4 integration.

Rationale: RK4 is deterministic, easy to audit, and accurate enough for this reduced-order open-loop artifact without introducing adaptive-step solver variability.

## DD-007: Sign Convention for Flap Pitching Moment

Decision: positive flap deflection uses a negative `cm_delta`, so a positive flap command creates a nose-down pitching-moment increment in the default model.

Rationale: explicit sign conventions make unit tests and later controller design less ambiguous.

## DD-008: Phase 3 Uses Shared Rollout and Metrics for Baselines

Decision: PID and gain-scheduled LQR use the same rollout runner, actuator limiting, corridor checks, and metrics.

Rationale: baseline comparisons should differ by controller logic, not by evaluation plumbing.

## DD-009: Phase 3 LQR Uses Local Finite-Difference Linearization

Decision: the LQR baseline linearizes the reduced-order nonlinear model at configured schedule points and solves a small discrete Riccati problem without adding SciPy.

Rationale: this keeps the dependency footprint small while creating a real gain-scheduled linear baseline for comparison against future MPC.

## DD-010: Phase 4 Uses CasADi Opti for Nominal NMPC

Decision: Phase 4 builds a per-step CasADi `Opti` problem over a downsampled Phase 2 reference profile.

Rationale: this keeps the first NMPC implementation readable and close to the math, while generating solver logs and artifacts without introducing a larger solver framework.

## DD-011: Phase 4 Softens State Corridor Constraints

Decision: alpha and pitch-rate corridor constraints use nonnegative slack with a large penalty, while flap angle and flap-rate bounds remain hard.

Rationale: the existing reduced-order plant and corridor can be infeasible with hard state constraints; soft state constraints let the run complete while logging predicted and realized violations honestly.

## DD-012: Phase 4 Uses a Small Corridor-Counting Tolerance

Decision: Phase 4 counts realized corridor violations with a `0.001 rad` tolerance while preserving the raw state, corridor, and solver-log artifacts.

Rationale: after fixing the reduced alpha dynamics and using a feasible initial pitch-rate condition, the remaining alpha corridor misses were sub-milliradian numerical boundary effects. The tolerance prevents those from flipping the success label while keeping the raw trajectory available for audit.

## DD-013: Reduced Alpha Dynamics Use Pitch Rate

Decision: the reduced-order plant, finite-difference linearization, and NMPC prediction model use `alpha_dot = q - 0.22 * alpha`.

Rationale: `alpha_dot` has units of rad/s, so the direct kinematic attitude term must be pitch rate `q`, not pitch acceleration `q_dot`. This also aligns the implementation with the project theory notes that approximate `alpha_dot ~= q - gamma_dot`.

## DD-014: Phase 5 Uses Paired Nominal-Controller Monte Carlo Scenarios

Decision: Phase 5 samples one uncertainty scenario per seed and runs PID, gain-scheduled LQR, and nominal NMPC on that same sampled scenario. Controllers remain nominal; the rollout plant receives the sampled perturbations.

Rationale: paired scenarios make controller comparisons fair, while nominal-controller testing measures degradation under mismatch rather than giving any controller oracle knowledge of the perturbed plant.

## DD-015: Phase 5 Uses Moderate and Stress Uncertainty Tiers

Decision: Phase 5 now defines named uncertainty tiers in `configs/phase5_monte_carlo.yaml`. The moderate tier preserves the original 30-scenario benchmark; the stress tier adds 30 wider-range scenarios for density, aerodynamic scales, actuator lag/delay, sensor noise, initial errors, and disturbance moment.

Rationale: a single uncertainty level can hide whether failures are mild corridor misses or deeper robustness breakdowns. The tiered benchmark gives the blog two distinct stories: moderate mismatch exposes low nominal-controller success rates, while stress mismatch exposes unstable-response labels that motivate robust or learning-augmented MPC.

## DD-016: Phase 6 Starts With Constraint-Tightened NMPC

Decision: Phase 6 implements a constraint-tightened NMPC variant before adding scenario MPC or learned residual models. The controller plans with tighter alpha and pitch-rate bounds, but the rollout metrics are evaluated against the original Phase 5 corridor.

Rationale: constraint tightening is the simplest robust-MPC idea to test first. It keeps the learning path clear: add margin, reuse the same uncertainty benchmark, and check whether the extra conservatism changes success rates or failure modes.

## DD-017: Phase 6 Preserves the Phase 5 Benchmark

Decision: Phase 6 reuses the Phase 5 sampled scenario definitions, failure thresholds, and baseline summary rather than changing the benchmark.

Rationale: a robust-control phase should improve on the existing uncertainty task. Changing the task at the same time as changing the controller would make the comparison harder to trust.

## DD-018: Phase 7 Uses Shared-Control Scenario NMPC

Decision: Phase 7 optimizes one shared flap sequence across multiple design futures inside each NMPC solve.

Rationale: this is the next robust-MPC step after static constraint tightening. A shared control sequence forces the optimizer to choose commands that are acceptable across several possible plants, rather than fitting one nominal prediction.

## DD-019: Phase 7 Uses a 12-Scenario-Per-Tier First Artifact Run

Decision: Phase 7 evaluates the first 12 moderate scenarios and first 12 stress scenarios, then filters the Phase 5/6 comparison table to the same overlapping tier/scenario keys.

Rationale: scenario NMPC is substantially more expensive than nominal NMPC because each solve propagates multiple design futures. A smaller first benchmark keeps iteration practical while preserving paired comparison discipline.

## DD-020: Phase 7 Treats `Solved_To_Acceptable_Level` as Successful Convergence

Decision: Phase 7 does not label IPOPT `Solved_To_Acceptable_Level` as a solver failure.

Rationale: scenario NMPC has a larger nonlinear program than nominal NMPC. IPOPT's acceptable convergence status is still a usable solution for this research scaffold, while true `RuntimeError` solve failures remain logged and labeled.

## DD-021: Blog GIFs Are Generated From Saved Artifacts

Decision: blog-facing GIFs are generated from `outputs/phase2_reference/reference_profile.csv` and `outputs/phase7_scenario_mpc/phase7_rollouts.csv`.

Rationale: animations should make the technical blog more readable without becoming separate, unauditable media. Reusing saved CSV artifacts keeps the visuals reproducible and tied to the same data used by plots, tables, and claims.

## DD-022: Residual Learning Uses Additive Phase 8/9 Numbering

Decision: the requested residual dataset and PyTorch model phases are implemented as Phase 8 and Phase 9 in this repository.

Rationale: Phase 6 and Phase 7 already exist as constraint-tightened NMPC and scenario NMPC. Additive numbering preserves backward compatibility with existing scripts, outputs, tests, and claims while still implementing the requested residual-learning work.

## DD-023: Residual Targets Are Derivative Corrections

Decision: residual targets are generated as `x_dot_true - x_dot_nominal` at sampled state/control/schedule points.

Rationale: derivative residuals are the cleanest supervised target for later learning-augmented MPC. They teach a model the local correction to the dynamics rather than mixing dynamics error with controller behavior, rollout integration error, or failure labels.

## DD-024: Phase 10 Uses Local Equivalent-Moment Residual Correction

Decision: Phase 10 does not embed the PyTorch residual model directly inside the CasADi symbolic graph. Instead, it predicts a local `q_dot` residual at each NMPC update and converts that correction into an equivalent `cm0` bias for the current solve.

Rationale: this preserves the existing NMPC solver structure and keeps the first learned-residual control experiment auditable. It is not the final architecture for learning-augmented MPC, but it tests whether a simple learned model correction helps before adding a more complex differentiable or exported residual model.

## DD-025: Phase 11 Uses a CasADi-Compatible Polynomial Residual Surrogate

Decision: Phase 11 fits a deterministic ridge-regression polynomial surrogate for `residual_q_dot` from Phase 8 data and embeds that surrogate directly in the NMPC horizon.

Rationale: PyTorch models are not directly compatible with the current CasADi `Opti` graph. A small polynomial surrogate keeps the horizon correction symbolic, reproducible, and easy to inspect while preserving the fixed Phase 5/7 benchmark.

## DD-026: Phase 12 Uses Horizon-Scheduled PyTorch Residual Biases

Decision: Phase 12 uses the trained Phase 9 PyTorch residual model before each NMPC solve to predict one fixed `residual_q_dot` value per horizon interval. The CasADi optimization then treats those values as scheduled dynamics biases.

Rationale: direct PyTorch-inside-CasADi integration is out of scope for this pass. Scheduling the residuals preserves a clean benchmark, logs neural inference time, and lets the learned model influence every prediction interval without making the NLP depend on a non-symbolic neural network.

## DD-027: Phase 12 Uses the Phase 4/5 Ten-Step NMPC Horizon

Decision: Phase 12 uses `horizon_steps: 10`, matching the nominal NMPC horizon used by Phase 4 and the Phase 5 nominal Monte Carlo controller.

Rationale: an initial six-step artifact run completed but did not exactly match the Phase 5 nominal baseline. The saved Phase 12 artifacts were regenerated with the ten-step horizon so the nominal, residual-corrected, and residual-corrected tightened variants are compared under the intended baseline configuration.

## DD-028: Phase 13 Measures Corridor Miss Size Before Changing Controllers

Decision: Phase 13 is an analysis-only pass over Phase 12 artifacts. It computes needed alpha/q corridor expansion, first violation time, violation duration, and actuator saturation fractions without changing controller commands, thresholds, or failure labels.

Rationale: the project needs to learn why success is low before adding another controller variant. Measuring miss size and timing separates small boundary grazes from large/early violations and keeps the next controller phase grounded in evidence.

## DD-029: Real-Time Timing Is Implemented as Additive Phase 14

Decision: the user-requested real-time feasibility and timing analysis is implemented as Phase 14 rather than overwriting the existing Phase 10 learned-residual NMPC artifacts.

Rationale: Phase 10 already has a meaning in this repository. Additive numbering preserves script/output compatibility while still delivering the requested timing milestone.

## DD-030: Phase 14 Times Representative Control Calls

Decision: Phase 14 times representative PID/LQR command calls and NMPC solve calls across horizon length and control frequency, rather than rerunning full Monte Carlo rollouts for every timing configuration.

Rationale: the goal is onboard timing feasibility, not another controller-performance benchmark. Reusing Phase 5/12 success rates for the Pareto plot keeps performance evidence tied to existing paired Monte Carlo artifacts while keeping the timing benchmark fast and reproducible.

## DD-031: Phase 15 Uses Deterministic Fault Probes Before Stochastic Fault Campaigns

Decision: Phase 15 defines one deterministic scenario per fault type and compares residual NMPC with and without fallback logic.

Rationale: the project first needs inspectable case studies showing how faults enter the closed loop and which fallback hooks fire. A stochastic fault campaign would be premature before the fault logging and fallback state machine are visible and tested.

## DD-032: Phase 15 Logs Fallback Hooks Even When They Do Not Rescue Success

Decision: Phase 15 records fallback actions on every rollout row and keeps previous-feasible/LQR safe-mode fallbacks in the implementation even when the current artifact run does not trigger them.

Rationale: a fault-tolerance phase must distinguish "logic exists but did not activate" from "logic activated and helped." The current result is a useful negative baseline: residual-error tightening activates, while solver-failure fallbacks do not because the dominant failures are plant/corridor failures.

## DD-033: Phase 16 Keeps the Benchmark Fixed While Changing Controller Planning

Decision: Phase 16 reuses Phase 5 scenario sampling and Phase 5 failure-label logic while testing only controller-side planning changes: faster update periods, objective weight scaling, and inward planning buffers.

Rationale: the project needs to improve success honestly. If the alpha corridor or tolerance changes, success rates are no longer directly comparable to Phase 5/12. Planning buffers are allowed because they make the controller conservative internally while evaluation still uses the original corridor.

## DD-034: Phase 16 Supports Quick Artifact Runs Without Treating Them as Claims

Decision: Phase 16 includes config-level `max_time_s` and scenario-count controls so tests and visible learning runs can complete quickly, but the docs mark quick results as preliminary.

Rationale: full paired NMPC sweeps are expensive in Python/CasADi. A quick run is useful for validating artifacts and diagnostics, but claims must wait for the staged 12-scenario and full paired runs.

## DD-035: Phase 17 Separates Strict Success From Controlled Recovery

Decision: Phase 17 keeps the original Phase 5 strict success labels and adds a secondary `controlled_recovery` metric for bounded corridor miss without instability or solver failure.

Rationale: the user explicitly wanted both metrics in the blog. Strict success answers whether the controller satisfies the original zero-violation requirement. Controlled recovery answers whether the controller keeps the response bounded even when the corridor is grazed.

## DD-036: Phase 17 Adds Applied Flap to the Prediction State

Decision: Phase 17 uses an actuator-aware prediction state `[alpha, q, theta, delta_applied]` with a first-order effective actuator time constant based on lag and delay.

Rationale: previous NMPC variants optimized as if raw flap commands affected the plant immediately, while the Monte Carlo plant includes lag and delay. This mismatch is a plausible contributor to early and medium-time alpha corridor exits.

## DD-037: Phase 18 Uses a Non-Causal Slack Oracle as a Ceiling Diagnostic

Decision: Phase 18 solves one full-horizon nonlinear program per scenario to minimize alpha/q slack under actuator limits.

Rationale: after multiple online controllers failed to recover strict success, the project needs to know whether strict corridor satisfaction is achievable in principle. A non-causal oracle is not deployable, but it separates controller weakness from task infeasibility.

## DD-038: Phase 19 Imitates the Oracle With an Interpretable Ridge Policy

Decision: Phase 19 uses a deterministic ridge-regression policy over corridor-centered features before trying a neural or MPC-hybrid imitation controller.

Rationale: the project needs to learn whether the slack oracle's command structure is useful online. A small linear policy makes the first imitation step reproducible, debuggable, and easy to explain in the blog. Features that were not present in the Phase 18 oracle trajectory artifact are excluded from the default policy to avoid a train/deploy mismatch.

## DD-039: Phase 20 Reports Scenario-Level Ceiling Gaps

Decision: Phase 20 records both missed oracle-feasible scenarios and online successes beyond the oracle strict classification.

Rationale: aggregate success counts can hide scenario swaps. A policy can match the oracle feasible count while failing a specific oracle-feasible case and succeeding elsewhere. The blog needs the scenario-level gap to avoid overclaiming.

## DD-040: Phase 21 Uses Oracle-Command Replay as a Transfer Diagnostic

Decision: Phase 21 replays the Phase 20 oracle raw flap command sequence through the uncertain Monte Carlo plant for missed oracle-feasible cases.

Rationale: this isolates whether the remaining Phase 20 misses are mainly caused by ridge-policy command imitation error. Keeping the same actuator dynamics, uncertainty scenario, corridor, tolerance, and failure labels makes the comparison fair while preserving the diagnostic nature of the non-causal oracle.

## DD-041: Phase 22 Uses Slack-First Online MPC Before Another Learner

Decision: Phase 22 implements actuator-aware slack-MPC before adding a larger imitation model or DAgger loop.

Rationale: Phase 21 showed that replaying oracle commands does not recover missed strict successes. The next most informative experiment is therefore to put the oracle's slack-first objective into online feedback while preserving the fixed benchmark. This tests objective transfer before adding another learning layer.

## DD-042: Phase 23 Audits the Oracle With Scenario-Specific Truth Dynamics

Decision: Phase 23 rebuilds the slack oracle with density scaling, aero coefficient scaling, disturbance moment, and actuator lag/delay approximation from each sampled Monte Carlo scenario.

Rationale: Phase 22 matched but did not beat Phase 20. Before adding another controller, the project needs to know whether the previous oracle ceiling was optimistic. A truth-consistent oracle separates controller gap from benchmark feasibility pressure.

## DD-043: Phase 24 Keeps Learning Subordinate to MPC

Decision: Phase 24 uses the learned oracle-imitation policy only as a warm start and optional command prior for slack-MPC.

Rationale: Phase 23 showed the strict-success ceiling is already matched by Phase 22. A learned policy should therefore improve architecture, initialization, and timing diagnostics without bypassing the safety-critical constrained optimizer.

## DD-044: Package Blog Artifacts From Saved Evidence

Decision: Phase 25 reads existing phase outputs instead of rerunning every expensive benchmark. It creates headline tables, manifests, and GIFs from saved CSV/JSON artifacts.

Rationale: the final blog needs a fast, reproducible media pack while preserving the detailed source commands for full benchmark regeneration. GIFs are generated from data, but they remain explanatory artifacts rather than validation evidence.
