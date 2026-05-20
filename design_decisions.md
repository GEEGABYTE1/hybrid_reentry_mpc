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
