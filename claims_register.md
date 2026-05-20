# Claims Register

Use this file to keep blog claims tied to evidence. Do not promote unsupported claims into `blog/draft.md`.

| Claim ID | Claim | Evidence Artifact | Status | Notes |
|---|---|---|---|---|
| C-001 | The repository can generate deterministic smoke artifacts for a baseline and learning-augmented attitude controller. | `outputs/metrics/smoke_summary.json`, `outputs/figures/smoke_attitude_tracking.png` | Scaffolded | Supported after smoke command is run locally. |
| C-002 | Learning augmentation reduces tracking error in the toy smoke model. | `outputs/metrics/smoke_summary.csv` | Provisional | Requires checking generated metrics; toy-model-only claim. |

## Status Values

- `Planned`: not yet supported by artifacts.
- `Provisional`: supported by preliminary artifacts only.
- `Supported`: backed by saved metrics, figures, and a log entry.
- `Retired`: removed or contradicted by later evidence.
