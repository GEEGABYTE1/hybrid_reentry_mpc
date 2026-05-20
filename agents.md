# Agents Guide

## Mission

Build a careful, reproducible technical blog repository for "Learning-Augmented MPC for Reentry Attitude Control." Treat code, metrics, figures, and prose as linked artifacts.

## Ground Rules

- Prefer small, reviewable experiment phases.
- Keep random seeds explicit in configs.
- Do not add a claim to `blog/draft.md` unless it is traceable to `claims_register.md`.
- Every run that supports the blog must emit CSV/JSON metrics, generated figures when applicable, and a blog log entry.
- Preserve generated outputs only when they are intentionally part of the record.
- Keep the smoke test fast enough for routine CI.

## Quality Bar

- `pytest` passes.
- `ruff check .` passes.
- `black --check .` passes.
- Figures have captions in `blog/figure_captions.md`.
- Tables have source paths in `blog/results_tables.md`.

## Suggested Phase Flow

1. Define or update a config in `configs/`.
2. Run the experiment script.
3. Inspect generated metrics and figures.
4. Append or verify the log entry.
5. Register supported claims and limitations.
6. Update draft prose only after the evidence exists.
