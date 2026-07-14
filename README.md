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
| DTELS-Bench | 5-granularity | `dtels_bench_5_granularity_input.jsonl`, `dtels_bench_5_granularity_output.jsonl` |
| DTELS-Bench | 10-granularity | `dtels_bench_10_granularity_input.jsonl`, `dtels_bench_10_granularity_output.jsonl` |
| CCKS2025 | 5-granularity | `ccks2025_5_granularity_input.jsonl`, `ccks2025_5_granularity_output.jsonl` |
| CCKS2025 | 10-granularity | `ccks2025_10_granularity_input.jsonl`, `ccks2025_10_granularity_output.jsonl` |

The filenames intentionally include `ccks2025` for the CCKS2025 data. DTELS-Bench filenames do not include `CCKS`.

## Important dataset notes

- The current DTELS-Bench release contains 373 paired records for the 5-granularity setting and 362 paired records for the 10-granularity setting.
- The 10-granularity records were filtered to remove problematic samples found in the original DTELS-Bench data.
- `input` files contain the source timelines; `output` files contain the corresponding target timelines with exactly 5 or 10 nodes.
- Please read [`data/README.md`](data/README.md) before using or redistributing the files.

## Repository status

| Component | Status |
|---|---|
| Dataset files | Available: DTELS-Bench and CCKS2025 |
| Dataset documentation | Available |
| Training code | Available: Time-GRPO and RS-SFT |
| Evaluation code | Available: model evaluation and CCKS2025 demo |
| Paper/supplementary materials | Included in repository directory |
| Base models and checkpoints | Not included; provide paths locally |

## Models and hardware

The reported experimental setup uses:

- **Generation model:** Llama-3.1-8B-Instruct.
- **Fine-tuning:** LoRA/PEFT applied to the causal language model; the training scripts support gradient checkpointing and optional 4-bit or 8-bit quantization.
- **Embedding model:** `m3e-base`, used for semantic similarity and reward-search / STAR-related computations.
- **Precision:** bfloat16 for the reported GPU experiments, with float16 fallback where bfloat16 is unavailable.
- **Hardware:** four NVIDIA A800 GPUs for the multi-GPU experiments.

Model weights, embedding weights, LoRA adapters, and experiment outputs are intentionally excluded from this repository. The migrated scripts contain some legacy local defaults such as `Qwen3-8B`; reproduce the paper setup by explicitly passing the Llama-3.1-8B-Instruct and `m3e-base` paths.

## Code structure

```text
SDM2026/
├── data/                         # Released JSONL datasets
├── scripts/
│   ├── time_grpo.py              # Time-GRPO training
│   ├── rs_sft_train.py           # Reward-Search SFT baseline
│   ├── evaluate_model.py         # Acc / ROUGE / STAR evaluation
│   └── convert_data.py           # JSONL-to-training-format conversion
├── src/
│   └── time_value.py             # Temporal distribution reward utilities
├── evaluation/ccks2025_demo/    # CCKS2025 evaluation implementation
├── configs/                      # Legacy experiment configuration
└── requirements.txt
```

The repository does not include base models, embedding models, LoRA checkpoints, or experiment outputs. Provide local model paths and use the matching `input`/`output` files when running training or evaluation. The migrated scripts retain some original `data_us/` defaults; the released dataset files should be passed explicitly where supported.

## Quick start

From the repository root, install the dependencies and inspect a script's options:

```bash
pip install -r requirements.txt
python scripts/rs_sft_train.py --help
python scripts/evaluate_model.py --help
```

Example Time-GRPO training command for the DTELS-Bench 10-granularity setting:

```bash
python scripts/time_grpo.py \
  --model_name_or_path /path/to/base-model \
  --questions_path /path/to/verified-input.jsonl \
  --answers_path /path/to/verified-reference.jsonl \
  --top_k 10 \
  --output_dir outputs/time-grpo-k10
```

The evaluation script supports five-fold evaluation and multi-GPU sharding. Set `--peft_model_path` when evaluating a LoRA checkpoint and provide an embedding model with `--embed_model_path`.

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
