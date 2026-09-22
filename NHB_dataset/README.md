# NarrativeHarmBench Dataset

This package is a cleaned and GitHub-ready organization of the benchmark data from the provided project archive.

## Public-release files

- `benchmark/NarrativeHarmBench.csv` — final retained benchmark
- `metadata/events.csv` — cleaned event metadata with stable `event_uid`
- `metadata/framing_taxonomy.csv` — adversarial framing taxonomy

## Reproducibility files

- `source/all_generated_prompts.csv` — all 11,700 generated prompt records before quality filtering
- `audit/prompt_quality_audit.csv` — cleaned prompt-quality audit scores and failure flags
- `review/` — items that should be checked before the final public release

## Cleaning applied

- standardized column names;
- trimmed leading/trailing whitespace and non-breaking spaces;
- removed the completely empty `Opinion-based prompts` field;
- removed raw duplicated audit JSON from the clean audit table because the values already exist in dedicated score/failure columns;
- added stable `prompt_id` and `event_uid`;
- preserved the source identifier as `original_event_id`;
- removed spreadsheet-specific comments/notes from the release format by exporting clean CSV files;
- preserved all retained rows, including exact duplicates, so the benchmark count was not silently changed.

## Final benchmark

The source rule `quality_flag_ge_3 = 1` yields **4,991 retained prompt records**.
