# Experiment scripts

Run these commands from the repository root so that `data/` and `src/` resolve correctly.

## Time-GRPO

`time_grpo.py` is the main GRPO training entry point. It uses the temporal reward utilities in `src/time_value.py` and supports `--top_k 5` or `--top_k 10`.

```bash
python scripts/time_grpo.py \
  --model_name_or_path /path/to/base-model \
  --questions_path /path/to/verified-input.jsonl \
  --answers_path /path/to/verified-reference.jsonl \
  --top_k 10 \
  --output_dir outputs/time-grpo-k10
```

## Reward-Search SFT

`rs_sft_train.py` implements the reward-search SFT baseline. Pass `--questions_path` and `--answers_path` explicitly; the original experiment defaults refer to files under `data_us/`, which are not included in this release.

## Model evaluation

`evaluate_model.py` computes model-based evaluation results including accuracy, ROUGE, and STAR-related scores. For a LoRA model, provide both `--model_name_or_path` and `--peft_model_path`.

## Data conversion

`convert_data.py` converts a matched input/output pair into a PPO-style JSON training file. Its original default paths refer to `data_us/`; update the paths in the script before using it with the released files.

## Reproducibility notes

- The scripts retain the experimental hyperparameters used in the original working directory where possible.
- Base models, embedding models, GPU count, checkpoint paths, and output directories are machine-specific and must be supplied by the user.
- The old experimental directory is preserved separately; this repository contains a cleaned release copy only.
