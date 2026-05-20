# Publishing Checklist

## Reproducibility

- [ ] All blog figures are regenerated from committed configs.
- [ ] CSV and JSON metrics exist for each result.
- [ ] `outputs/logs/blog_log.jsonl` has an entry for each experiment phase.
- [ ] Seeds are listed in configs and cited where relevant.

## Evidence

- [ ] Every technical claim appears in `claims_register.md`.
- [ ] Every limitation appears in `limitations.md`.
- [ ] Every figure appears in `plots_manifest.md`.
- [ ] Every results table appears in `blog/results_tables.md`.

## Engineering

- [ ] `pytest` passes.
- [ ] `ruff check .` passes.
- [ ] `black --check .` passes.
- [ ] Smoke CLI runs from a clean checkout.

## Editorial

- [ ] The draft distinguishes toy-model findings from validated reentry-control conclusions.
- [ ] Figure captions are complete and caveated.
- [ ] The opening thesis matches the actual evidence.
- [ ] The conclusion lists concrete next experiments.
