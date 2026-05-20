# Learning-Augmented MPC for Reentry Attitude Control

Research-engineering repository for a technical blog project on learning-augmented model predictive control (MPC) for atmospheric reentry attitude control.

The repo is organized around reproducible artifacts: every experiment phase should save numeric metrics, generated figures, and a compact blog log entry before any prose claim is promoted into the draft.

## Repository Layout

```text
.
├── blog/                 # Outline, draft, captions, tables, publishing checklist
├── configs/              # Versioned experiment configs
├── outputs/              # Generated metrics, figures, and logs
├── scripts/              # Command-line experiment wrappers
├── src/reentry_mpc/      # Python package
└── tests/                # Pytest checks for artifact generation
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m reentry_mpc.cli --config configs/smoke.yaml --output-dir outputs
pytest
ruff check .
black --check .
```

The smoke command writes:

- `outputs/metrics/smoke_trajectory.csv`
- `outputs/metrics/smoke_summary.csv`
- `outputs/metrics/smoke_summary.json`
- `outputs/figures/smoke_attitude_tracking.png`
- `outputs/logs/blog_log.jsonl`

## Reproducibility Contract

Every phase must record:

- Config path and deterministic seed.
- CSV and JSON metrics.
- Generated figure files with stable names.
- A short `outputs/logs/blog_log.jsonl` entry.
- Any blog-facing claim in `claims_register.md`.
- Any caveat or known non-goal in `limitations.md`.

## Current Smoke Scope

The initial code is a deterministic single-axis toy attitude simulation. It is not a validated reentry vehicle model. Its purpose is to exercise the artifact workflow and give the blog project a reliable scaffold for future dynamics, MPC optimization, and learned residual models.
