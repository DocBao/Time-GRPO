# Time-GRPO

Official data and supplementary-material repository for:

**Timeline Granularity Transformation: Aligning Narrative Logic with Temporal-Semantic Evolution**

Accepted at the **SIAM International Conference on Data Mining (SDM 2026)**.

> Dataset files and documentation are available in this repository. Training and evaluation code are planned for release in August 2026.

## Overview

This repository contains the data used for experiments on timeline granularity transformation. The task transforms a fine-grained chronological timeline into a target timeline with an exact number of granularity levels while preserving temporal-semantic evolution.

The repository currently contains data for two sources:

- **DTELS-Bench**
- **CCKS2025**

Here, `5` and `10` in a filename denote the target **5-granularity** and **10-granularity** settings, respectively. They do not denote dataset versions.

## Data files

All JSONL files are stored under [`data/`](data/).

| Dataset | Setting | Files |
|---|---:|---|
| DTELS-Bench | 5-granularity | `dtels_bench_5_granularity_input.jsonl`, `dtels_bench_5_granularity_output.jsonl`, `dtels_bench_5_granularity_answer_processed.jsonl` |
| DTELS-Bench | 10-granularity | `dtels_bench_10_granularity_input.jsonl`, `dtels_bench_10_granularity_output.jsonl`, `dtels_bench_10_granularity_filtered_gold_reference.jsonl` |
| CCKS2025 | 5-granularity | `ccks2025_5_granularity_input.jsonl`, `ccks2025_5_granularity_output.jsonl` |
| CCKS2025 | 10-granularity | `ccks2025_10_granularity_input.jsonl`, `ccks2025_10_granularity_output.jsonl` |

The filenames intentionally include `ccks2025` for the CCKS2025 data. DTELS-Bench filenames do not include `CCKS`.

## Important dataset notes

- `dtels_bench_5_granularity_answer_processed.jsonl` is the complete processed DTELS-Bench file for the 5-granularity setting.
- `dtels_bench_10_granularity_filtered_gold_reference.jsonl` is the filtered DTELS-Bench gold-reference file for the 10-granularity setting. Some problematic records were removed because the original DTELS-Bench data contained errors.
- The filtered 10-granularity file should therefore be treated as a corrected/filtered release, not as an unchanged copy of the original benchmark.
- Please read [`data/README.md`](data/README.md) before using or redistributing the files.

## Repository status

| Component | Status |
|---|---|
| Dataset files | Available |
| Dataset documentation | Available |
| Paper/supplementary materials | Included in repository directory |
| Training code | Planned for August 2026 |
| Evaluation code | Planned for August 2026 |

## Citation

Please cite the paper when using this repository. The complete bibliographic record and DOI will be added when the official proceedings metadata becomes available.

```bibtex
@inproceedings{timegrpo2026,
  title     = {Timeline Granularity Transformation: Aligning Narrative Logic with Temporal-Semantic Evolution},
  booktitle = {Proceedings of the 2026 SIAM International Conference on Data Mining},
  year      = {2026}
}
```

## License and responsible use

The data files may be subject to different licenses and access terms. No license for third-party data is granted by this repository. See [`data/README.md`](data/README.md) for source and redistribution information.

The code license will be added with the planned August 2026 code release.

## Contact

For questions or to report a data issue, please open a GitHub issue in this repository.
