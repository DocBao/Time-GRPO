# Data README

This directory contains the data released with the Time-GRPO project.

## Naming convention

The naming pattern is:

```text
<dataset>_<granularity>_granularity_<role>.jsonl
```

The granularity value is the target number of levels:

- `5_granularity`: 5-granularity setting
- `10_granularity`: 10-granularity setting

Dataset prefixes are used as follows:

- `dtels_bench`: DTELS-Bench data; the name does not include CCKS.
- `ccks2025`: CCKS2025 data; the name explicitly includes CCKS2025.

## File inventory

### DTELS-Bench

#### 5-granularity

- `dtels_bench_5_granularity_input.jsonl`
- `dtels_bench_5_granularity_output.jsonl`
- `dtels_bench_5_granularity_answer_processed.jsonl`

`dtels_bench_5_granularity_answer_processed.jsonl` is the complete processed 5-granularity answer file.

#### 10-granularity

- `dtels_bench_10_granularity_input.jsonl`
- `dtels_bench_10_granularity_output.jsonl`
- `dtels_bench_10_granularity_filtered_gold_reference.jsonl`

The `filtered_gold_reference` file is a corrected/filtered version for experimental use. Records with identified problems in the original DTELS-Bench data were removed. Users should preserve this filename and description when citing or mirroring the release.

### CCKS2025

#### 5-granularity

- `ccks2025_5_granularity_input.jsonl`
- `ccks2025_5_granularity_output.jsonl`

#### 10-granularity

- `ccks2025_10_granularity_input.jsonl`
- `ccks2025_10_granularity_output.jsonl`

## Format

Each file is JSON Lines (JSONL): one JSON object per line. The records contain timeline-related fields such as a title, identifier, timeline, dates, summaries, and—depending on the file—stage, label, atom, or reference information.

Because the files may contain multilingual text, readers should open them as UTF-8. The visible text may look corrupted if a tool decodes UTF-8 as another encoding.

## Source and redistribution

| Dataset | Source | Redistribution status |
|---|---|---|
| DTELS-Bench | DTELS-Bench benchmark and project processing | Check the upstream terms and the project release terms before redistribution. |
| CCKS2025 | CCKS2025 data and project processing | Check the original competition/data terms before redistribution. |

This repository does not override upstream licenses, competition rules, or third-party copyright. In particular, the presence of a file here should not be interpreted as permission to redistribute the underlying third-party source material beyond the applicable terms.

## Known processing note

The 10-granularity DTELS-Bench gold-reference release excludes records identified as problematic in the original data. The filtered file is intended to make the experimental setting reproducible while avoiding those known erroneous records.

## Version

Current release: initial dataset release, July 2026.

Checksums and a more detailed dataset card will be added after the final upload contents are frozen.
