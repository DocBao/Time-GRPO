import argparse
import collections
import datetime
import itertools
import json
import logging
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jieba
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from rouge_chinese import Rouge
from sentence_transformers import SentenceTransformer
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

# 保持与 Time-GRPO 训练代码一致的时间分布奖励实现
from src import time_value


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
rouge = Rouge()


@dataclass
class SearchResult:
    indices: List[int]
    total_reward: float
    reward_breakdown: Dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reward-Search SFT (RS-SFT) baseline converted from Time-GRPO"
    )
    parser.add_argument("--output_dir", type=str, default="./outputs/rs-sft")
    # /media/hello/9616eaa1-d1a4-49ca-ba88-0d251ec109d7/bcl/model/model_output/rs-sft
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='model/Qwen3-8B'
        # "/media/hello/9616eaa1-d1a4-49ca-ba88-0d251ec109d7/bcl/model/Llama-3.1-8B-Instruct",
    )
    # model/Qwen3-8B
    parser.add_argument(
        "--embedding_model_path",
        type=str,
        default="/home/hello/Desktop/DTELS-Bench/model/m3e-base",
    )
    parser.add_argument("--questions_path", type=str, default="data_us/filter_input.jsonl")
    parser.add_argument("--answers_path", type=str, default="data_us/filter_output.jsonl")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--quantization", type=str, default="None", choices=["None", "4bit", "8bit"])
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=8196)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--id_field", type=str, default="id")
    parser.add_argument("--top_k", type=int, default=5, choices=[5, 10])
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--test_fold_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search_device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--search_batch_size", type=int, default=32)
    parser.add_argument(
        "--search_pool_size",
        type=int,
        default=24,
        help="reward-search 候选池大小上限；K=10 建议设为 20-24",
    )
    parser.add_argument("--beam_width", type=int, default=32)
    parser.add_argument("--exhaustive_max_combinations", type=int, default=50000)
    parser.add_argument("--local_refine_rounds", type=int, default=2)
    parser.add_argument(
        "--pseudo_label_cache",
        type=str,
        default="",
        help="不填则默认保存在 output_dir/pseudo_labels_C_top{K}_{split}.jsonl",
    )
    parser.add_argument("--overwrite_pseudo_cache", action="store_true")
    parser.add_argument(
        "--eval_target_source",
        type=str,
        default="gold",
        choices=["gold", "pseudo"],
        help="训练中 eval loss 使用 gold 还是 pseudo target",
    )
    parser.add_argument("--report_to", nargs="*", default=["tensorboard"])
    return parser.parse_args()


class RewardSearchEngine:
    """离线 reward-search：使用与 Time-GRPO 相同的 reward 结构搜索 pseudo targets。"""

    def __init__(self, embed_model: SentenceTransformer, args: argparse.Namespace):
        self.embed_model = embed_model
        self.args = args
        self.search_device = args.search_device

    @staticmethod
    def clean_timeline(timeline: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
        cleaned: List[Dict[str, str]] = []
        for event in timeline:
            time_str = str(event.get("time", "")).strip()
            summary = str(event.get("summary", "")).strip()
            if time_str and summary:
                cleaned.append({"time": time_str, "summary": summary})
        return cleaned

    @staticmethod
    def event_key(event: Dict[str, str]) -> Tuple[str, str]:
        return event.get("time", "").strip(), event.get("summary", "").strip()

    def compute_global_t_min_tensor(self, full_timeline: Sequence[Dict[str, str]]) -> torch.Tensor:
        global_t_min_val = 0.0
        for event in full_timeline:
            time_str = event.get("time", "")
            if re.match(r"^\d{4}-\d{2}-\d{2}$", time_str):
                try:
                    dt = datetime.datetime.strptime(time_str, "%Y-%m-%d")
                    global_t_min_val = dt.timestamp()
                    break
                except ValueError:
                    continue
        return torch.tensor(global_t_min_val, dtype=torch.float32)

    def encode_summaries(self, summaries: Sequence[str]) -> torch.Tensor:
        if not summaries:
            return torch.empty((0, 1), device=self.search_device)
        embeddings = self.embed_model.encode(
            list(summaries),
            convert_to_tensor=True,
            device=self.search_device,
            batch_size=self.args.search_batch_size,
            show_progress_bar=False,
        )
        if not torch.is_tensor(embeddings):
            embeddings = torch.tensor(embeddings, device=self.search_device)
        return embeddings

    @staticmethod
    def calculate_ngram_entropy(text: str, n: int = 2) -> float:
        if not text:
            return 0.0
        tokens = list(jieba.cut(text))
        if len(tokens) < n:
            return 0.0
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        ngram_counts = collections.Counter(ngrams)
        total_ngrams = len(ngrams)
        probabilities = [count / total_ngrams for count in ngram_counts.values()]
        entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
        return float(entropy)

    def date_and_text_reward(
        self,
        selected_events: Sequence[Dict[str, str]],
        reference_events: Sequence[Dict[str, str]],
    ) -> Tuple[float, float]:
        reference_events_state = [
            {"time": e["time"], "summary": e["summary"], "matched": False}
            for e in reference_events
        ]

        date_scores: List[float] = []
        selected_summaries: List[str] = []
        for sel_event in selected_events:
            sel_time = sel_event.get("time", "").strip()
            sel_sum = sel_event.get("summary", "").strip()
            if sel_sum:
                selected_summaries.append(sel_sum)

            max_score = 0.0
            matched_ref_index = -1
            for idx, ref_event in enumerate(reference_events_state):
                if ref_event["matched"] or not sel_time or not sel_sum:
                    continue
                ref_time = ref_event["time"]
                ref_sum = ref_event["summary"]

                current_date_score = 0.0
                if sel_time == ref_time:
                    current_date_score = 1.0
                elif sel_time.replace("-", "") == ref_time.replace("-", ""):
                    current_date_score = 0.9
                elif sel_time in ref_time or ref_time in sel_time:
                    current_date_score = 0.7

                if current_date_score > 0 and sel_sum == ref_sum and current_date_score > max_score:
                    max_score = current_date_score
                    matched_ref_index = idx

            if matched_ref_index != -1:
                reference_events_state[matched_ref_index]["matched"] = True
            date_scores.append(max_score)

        date_reward = sum(date_scores) / len(date_scores) if date_scores else 0.0

        total_candidate_count = len(selected_summaries)
        summary_count = collections.Counter(selected_summaries)
        repeat_count = sum(count - 1 for count in summary_count.values() if count > 1)
        penalty_factor = 1.0 if total_candidate_count == 0 else max(0.0, 1.0 - (repeat_count / total_candidate_count))

        unique_selected_summaries = list(summary_count.keys())
        selected_full_text = " ".join(unique_selected_summaries)
        reference_full_text = " ".join([e["summary"] for e in reference_events if e.get("summary")])

        entropy_value = self.calculate_ngram_entropy(text=selected_full_text, n=2)
        base_rouge_score = 0.0
        if selected_full_text and reference_full_text:
            selected_segmented = " ".join(list(jieba.cut(selected_full_text)))
            reference_segmented = " ".join(list(jieba.cut(reference_full_text)))
            base_rouge_score = rouge.get_scores(selected_segmented, reference_segmented)[0]["rouge-1"]["f"]

        normalized_entropy = min(entropy_value / 5.0, 1.0)
        text_reward = base_rouge_score * penalty_factor * 0.8 + normalized_entropy * 0.2
        return float(date_reward), float(text_reward)

    def star_reward(
        self,
        selected_indices: Sequence[int],
        full_timeline: Sequence[Dict[str, str]],
        full_embeddings: torch.Tensor,
        global_t_min_tensor: torch.Tensor,
    ) -> float:
        if len(selected_indices) < 2:
            return 0.0

        timestamps: List[float] = []
        for idx in selected_indices:
            time_str = full_timeline[idx]["time"]
            try:
                ts = time_value.str2ts_tensor(time_str, global_t_min_tensor).item()
            except Exception:
                ts = global_t_min_tensor.item()
            timestamps.append(ts)

        ts_tensor = torch.tensor(timestamps, dtype=torch.float32, device=full_embeddings.device)
        intervals = torch.relu(ts_tensor[1:] - ts_tensor[:-1])
        log_intervals = torch.log(intervals + 1.0)
        time_dist_sum = torch.sum(log_intervals)
        if float(time_dist_sum.item()) < 1e-8:
            return 0.0
        time_dist = log_intervals / time_dist_sum

        emb = full_embeddings[list(selected_indices)]
        cos_sim = F.cosine_similarity(emb[:-1], emb[1:], dim=-1)
        sem_distance = 1.0 - cos_sim
        sem_dist_sum = torch.sum(sem_distance)
        if float(sem_dist_sum.item()) < 1e-8:
            return 0.0
        sem_dist = sem_distance / sem_dist_sum

        alignment_error = torch.sum(torch.abs(time_dist - sem_dist))
        star_reward = 1.0 - torch.pow(0.5 * alignment_error, 0.5)
        star_reward = torch.clamp(star_reward, 0.0, 1.0)
        return float(star_reward.detach().cpu().item())

    def time_reward(
        self,
        selected_events: Sequence[Dict[str, str]],
        reference_topk: Sequence[Dict[str, str]],
        full_timeline: Sequence[Dict[str, str]],
    ) -> float:
        if not selected_events:
            return 0.0
        selected_times = [e["time"] for e in selected_events]
        ref_times = [e["time"] for e in reference_topk]
        full_times = [e["time"] for e in full_timeline]
        try:
            timeline_state = time_value.detect_uniformity_no_deduplicate(full_times)
            return float(
                time_value.adaptive_time_reward_no_deduplicate(
                    selected_times,
                    ref_times,
                    full_times,
                    timeline_state,
                )
            )
        except Exception as exc:
            logger.warning("time_reward 计算失败，返回 0.0: %s", exc)
            return 0.0

    def total_reward(
        self,
        selected_indices: Sequence[int],
        full_timeline: Sequence[Dict[str, str]],
        reference_topk: Sequence[Dict[str, str]],
        full_embeddings: torch.Tensor,
        global_t_min_tensor: torch.Tensor,
    ) -> Tuple[float, Dict[str, float]]:
        selected_events = [full_timeline[i] for i in selected_indices]
        date_reward, text_reward = self.date_and_text_reward(selected_events, reference_topk)
        time_reward = self.time_reward(selected_events, reference_topk, full_timeline)
        star_reward = self.star_reward(selected_indices, full_timeline, full_embeddings, global_t_min_tensor)

        # 在离线搜索中，所有候选都满足 K 个事件且输出 JSON 模板固定，因此 format reward 为常数 1。
        format_reward = 1.0
        total_reward = (
            format_reward * 0.05
            + date_reward * 0.425
            + text_reward * 0.18
            + time_reward * 0.175
            + star_reward * 0.175
        )
        breakdown = {
            "format": float(format_reward),
            "date": float(date_reward),
            "text": float(text_reward),
            "time": float(time_reward),
            "star": float(star_reward),
            "total": float(total_reward),
        }
        return float(total_reward), breakdown

    def build_candidate_pool(
        self,
        full_timeline: Sequence[Dict[str, str]],
        reference_topk: Sequence[Dict[str, str]],
        full_embeddings: torch.Tensor,
    ) -> List[int]:
        n_events = len(full_timeline)
        if n_events <= self.args.search_pool_size:
            return list(range(n_events))

        ref_keys = {self.event_key(e) for e in reference_topk}
        ref_embeddings = self.encode_summaries([e["summary"] for e in reference_topk])

        scores: List[Tuple[float, int]] = []
        for idx, event in enumerate(full_timeline):
            key = self.event_key(event)
            exact_match = 1.0 if key in ref_keys else 0.0

            time_match = 0.0
            for ref_event in reference_topk:
                sel_time = event["time"]
                ref_time = ref_event["time"]
                if sel_time == ref_time:
                    time_match = max(time_match, 1.0)
                elif sel_time.replace("-", "") == ref_time.replace("-", ""):
                    time_match = max(time_match, 0.9)
                elif sel_time in ref_time or ref_time in sel_time:
                    time_match = max(time_match, 0.7)

            sem_sim = 0.0
            if len(ref_embeddings) > 0:
                ref_sim = F.cosine_similarity(full_embeddings[idx].unsqueeze(0), ref_embeddings, dim=-1)
                sem_sim = float((torch.max(ref_sim).item() + 1.0) / 2.0)

            normalized_position = idx / max(1, n_events - 1)
            center_bonus = 1.0 - abs(normalized_position - 0.5) * 2.0
            score = 0.55 * exact_match + 0.20 * time_match + 0.20 * sem_sim + 0.05 * center_bonus
            scores.append((score, idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        selected_indices = [idx for _, idx in scores[: self.args.search_pool_size]]

        # 额外加入边界与分位点锚点，避免候选池只集中在局部高分区域。
        anchor_indices = {0, n_events - 1}
        if n_events >= self.args.top_k:
            for q in range(self.args.top_k):
                anchor_indices.add(round(q * (n_events - 1) / max(1, self.args.top_k - 1)))

        selected_indices.extend(anchor_indices)
        return sorted(set(i for i in selected_indices if 0 <= i < n_events))

    def exhaustive_search(
        self,
        candidate_indices: Sequence[int],
        full_timeline: Sequence[Dict[str, str]],
        reference_topk: Sequence[Dict[str, str]],
        full_embeddings: torch.Tensor,
        global_t_min_tensor: torch.Tensor,
    ) -> SearchResult:
        best_score = -1e9
        best_indices: List[int] = list(candidate_indices[: self.args.top_k])
        best_breakdown: Dict[str, float] = {}

        for combo in itertools.combinations(candidate_indices, self.args.top_k):
            score, breakdown = self.total_reward(
                combo,
                full_timeline,
                reference_topk,
                full_embeddings,
                global_t_min_tensor,
            )
            if score > best_score:
                best_score = score
                best_indices = list(combo)
                best_breakdown = breakdown

        return SearchResult(indices=best_indices, total_reward=best_score, reward_breakdown=best_breakdown)

    def beam_search(
        self,
        candidate_indices: Sequence[int],
        full_timeline: Sequence[Dict[str, str]],
        reference_topk: Sequence[Dict[str, str]],
        full_embeddings: torch.Tensor,
        global_t_min_tensor: torch.Tensor,
    ) -> SearchResult:
        beam: List[Tuple[List[int], int, float]] = [([], -1, 0.0)]

        for _ in range(self.args.top_k):
            new_beam: List[Tuple[List[int], int, float]] = []
            for selected, last_pos, _ in beam:
                min_remaining = self.args.top_k - len(selected) - 1
                max_pos_exclusive = len(candidate_indices) - min_remaining
                for pos in range(last_pos + 1, max_pos_exclusive):
                    new_selected = selected + [candidate_indices[pos]]
                    partial_score, _ = self.total_reward(
                        new_selected,
                        full_timeline,
                        reference_topk,
                        full_embeddings,
                        global_t_min_tensor,
                    )
                    new_beam.append((new_selected, pos, partial_score))

            if not new_beam:
                break
            new_beam.sort(key=lambda x: x[2], reverse=True)
            beam = new_beam[: self.args.beam_width]

        best_indices = list(candidate_indices[: self.args.top_k])
        best_score = -1e9
        best_breakdown: Dict[str, float] = {}
        for selected, _, _ in beam:
            if len(selected) != self.args.top_k:
                continue
            score, breakdown = self.total_reward(
                selected,
                full_timeline,
                reference_topk,
                full_embeddings,
                global_t_min_tensor,
            )
            if score > best_score:
                best_indices = selected
                best_score = score
                best_breakdown = breakdown

        return SearchResult(indices=best_indices, total_reward=best_score, reward_breakdown=best_breakdown)

    def local_refine(
        self,
        initial_indices: Sequence[int],
        candidate_indices: Sequence[int],
        full_timeline: Sequence[Dict[str, str]],
        reference_topk: Sequence[Dict[str, str]],
        full_embeddings: torch.Tensor,
        global_t_min_tensor: torch.Tensor,
    ) -> SearchResult:
        current = sorted(initial_indices)
        current_score, current_breakdown = self.total_reward(
            current,
            full_timeline,
            reference_topk,
            full_embeddings,
            global_t_min_tensor,
        )

        for _ in range(self.args.local_refine_rounds):
            improved = False
            selected_set = set(current)
            unselected = [idx for idx in candidate_indices if idx not in selected_set]
            for pos in range(len(current)):
                for cand in unselected:
                    proposal = current[:]
                    proposal[pos] = cand
                    if len(set(proposal)) != self.args.top_k:
                        continue
                    proposal = sorted(proposal)
                    score, breakdown = self.total_reward(
                        proposal,
                        full_timeline,
                        reference_topk,
                        full_embeddings,
                        global_t_min_tensor,
                    )
                    if score > current_score:
                        current = proposal
                        current_score = score
                        current_breakdown = breakdown
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

        return SearchResult(indices=current, total_reward=current_score, reward_breakdown=current_breakdown)

    def search(self, example: Dict[str, Any]) -> SearchResult:
        full_timeline = self.clean_timeline(example["full_timeline"])
        reference_topk = self.clean_timeline(example["target_timeline"])
        if not full_timeline:
            return SearchResult(indices=[], total_reward=0.0, reward_breakdown={})

        if len(full_timeline) < self.args.top_k:
            logger.warning(
                "样本 %s 的 full_timeline 长度 %d < top_k=%d，直接返回全部事件。",
                example.get("id", "unknown"),
                len(full_timeline),
                self.args.top_k,
            )
            full_embeddings = self.encode_summaries([e["summary"] for e in full_timeline])
            global_t_min_tensor = self.compute_global_t_min_tensor(full_timeline)
            score, breakdown = self.total_reward(
                list(range(len(full_timeline))),
                full_timeline,
                reference_topk,
                full_embeddings,
                global_t_min_tensor,
            )
            return SearchResult(indices=list(range(len(full_timeline))), total_reward=score, reward_breakdown=breakdown)

        full_embeddings = self.encode_summaries([e["summary"] for e in full_timeline])
        global_t_min_tensor = self.compute_global_t_min_tensor(full_timeline)
        candidate_indices = self.build_candidate_pool(full_timeline, reference_topk, full_embeddings)

        combination_count = math.comb(len(candidate_indices), self.args.top_k)
        if combination_count <= self.args.exhaustive_max_combinations:
            result = self.exhaustive_search(
                candidate_indices,
                full_timeline,
                reference_topk,
                full_embeddings,
                global_t_min_tensor,
            )
        else:
            result = self.beam_search(
                candidate_indices,
                full_timeline,
                reference_topk,
                full_embeddings,
                global_t_min_tensor,
            )

        result = self.local_refine(
            result.indices,
            candidate_indices,
            full_timeline,
            reference_topk,
            full_embeddings,
            global_t_min_tensor,
        )
        return result


class SupervisedDataCollator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(feature["input_ids"], dtype=torch.long) for feature in features]
        attention_mask = [torch.tensor(feature["attention_mask"], dtype=torch.long) for feature in features]
        labels = [torch.tensor(feature["labels"], dtype=torch.long) for feature in features]

        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded,
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(file_path: str, id_field: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                if id_field in item:
                    data.append(item)
                else:
                    logger.warning("JSONL 文件 %s 第 %d 行缺少 ID 字段 %s，跳过", file_path, line_num, id_field)
            except json.JSONDecodeError as exc:
                logger.warning("JSONL 文件 %s 第 %d 行解析失败: %s，跳过", file_path, line_num, exc)
    return data


def load_custom_dataset(args: argparse.Namespace) -> Dataset:
    questions_data = read_jsonl(args.questions_path, args.id_field)
    answers_data = read_jsonl(args.answers_path, args.id_field)
    logger.info("成功读取输入数据 %d 条，答案数据 %d 条", len(questions_data), len(answers_data))

    answers_dict = {item[args.id_field]: item for item in answers_data if args.id_field in item}
    examples: List[Dict[str, Any]] = []
    for item in questions_data:
        data_id = item[args.id_field]
        if data_id not in answers_dict:
            continue
        examples.append(
            {
                args.id_field: data_id,
                "title": item.get("title", f"无标题_{data_id}"),
                "full_timeline": item.get("timeline", []),
                "target_timeline": answers_dict[data_id].get("timeline", []),
            }
        )

    logger.info(
        "最终构建训练样本 %d 条（跳过缺失答案样本 %d 条）",
        len(examples),
        len(questions_data) - len(examples),
    )
    return Dataset.from_list(examples)


def split_dataset(full_dataset: Dataset, args: argparse.Namespace) -> Tuple[Dataset, Dataset]:
    total_size = len(full_dataset)
    num_folds = args.num_folds
    test_fold_idx = args.test_fold_idx

    fold_size = total_size // num_folds
    remaining = total_size % num_folds
    folds: List[Tuple[int, int]] = []
    start = 0
    for i in range(num_folds):
        current_fold_size = fold_size + (1 if i < remaining else 0)
        end = start + current_fold_size
        folds.append((start, end))
        start = end

    test_start, test_end = folds[test_fold_idx]
    eval_dataset = full_dataset.select(range(test_start, test_end))
    train_indices = [
        idx
        for i, (fold_start, fold_end) in enumerate(folds)
        if i != test_fold_idx
        for idx in range(fold_start, fold_end)
    ]
    train_dataset = full_dataset.select(train_indices)
    logger.info("训练集 %d 条，验证集 %d 条", len(train_dataset), len(eval_dataset))
    return train_dataset, eval_dataset


def build_instruction(top_k: int) -> str:
    return f"""你是一个专业的新闻编辑助手，你的任务是从给定的新闻话题时间线中，挑选出最重要的{top_k}个关键事件摘要，帮助读者快速掌握该事件的核心发展脉络。
请遵循以下规则：
1. 仅从提供的 `timeline` 列表中选择事件，不得编造或修改原文摘要内容。
2. 优先选择具有标志性、转折性、公众广泛关注或引发重大后续影响的事件。
3. 输出按照<answer>JSON</answer>格式，JSON包含 `title` 和 `top_k_timeline` 字段，其中 `top_k_timeline` 是一个包含{top_k}个对象的列表，每个对象包含 `time` 和 `summary` 字段。
4. 保持时间顺序（按 `time` 升序排列）。"""


def format_timeline_for_display(timeline_data: Sequence[Dict[str, Any]]) -> str:
    return "\n\n".join(
        [
            f"{i + 1}. 时间: {event.get('time', '未知时间')}\n   摘要: {event.get('summary', '')}\n"
            for i, event in enumerate(timeline_data)
        ]
    )


def format_timeline_json(timeline_data: Sequence[Dict[str, Any]], title: str) -> str:
    simplified_timeline = [
        {"time": str(event.get("time", "")), "summary": str(event.get("summary", ""))}
        for event in timeline_data
    ]
    payload = {"title": title, "top_k_timeline": simplified_timeline}
    json_content = json.dumps(payload, ensure_ascii=False, indent=2)
    return f"<answer>\n{json_content}\n</answer>"


def make_conversation_record(example: Dict[str, Any], target_timeline: Sequence[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
    full_timeline_str = format_timeline_for_display(example["full_timeline"])
    instruction = build_instruction(top_k)
    user_content = f"\n{instruction}\n\n标题: {example['title']}\n\ntimeline:\n{full_timeline_str}"
    response = format_timeline_json(target_timeline, example["title"])
    return {
        "id": example.get("id", example.get("sample_id", "unknown")),
        "prompt": [
            {"role": "system", "content": "/no_think你是一个专业的新闻编辑助手，请按要求只输出 JSON 格式的答案。/no_think"},
            {"role": "user", "content": user_content},
        ],
        "response": response,
        "title": example["title"],
    }


def render_prompt(tokenizer: AutoTokenizer, prompt_messages: Sequence[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    rendered = []
    for msg in prompt_messages:
        rendered.append(f"{msg['role'].upper()}: {msg['content']}")
    rendered.append("ASSISTANT:")
    return "\n\n".join(rendered)


def tokenize_supervised_example(example: Dict[str, Any], tokenizer: AutoTokenizer, args: argparse.Namespace) -> Dict[str, List[int]]:
    prompt_text = render_prompt(tokenizer, example["prompt"])
    response_text = example["response"] + (tokenizer.eos_token or "")

    max_prompt_len = max(1, args.max_seq_length - args.max_completion_length)
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_prompt_len,
    )["input_ids"]
    response_ids = tokenizer(
        response_text,
        add_special_tokens=False,
        truncation=True,
        max_length=args.max_completion_length,
    )["input_ids"]

    input_ids = prompt_ids + response_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + response_ids
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def get_lora_config() -> LoraConfig:
    return LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_model_and_tokenizer(args: argparse.Namespace) -> Tuple[torch.nn.Module, AutoTokenizer]:
    if args.quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif args.quantization == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        bnb_config = None

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("已设置 pad_token 为 eos_token: %s", tokenizer.pad_token)

    target_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(args.dtype, torch.bfloat16)
    if target_dtype == torch.bfloat16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        logger.warning("当前设备不支持 bfloat16，自动切换为 float16")
        target_dtype = torch.float16

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": target_dtype,
    }
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
    if torch.cuda.is_available():
        model_kwargs["attn_implementation"] = "flash_attention_2"

    # model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=target_dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )

    if bnb_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model = get_peft_model(model, get_lora_config())
    model.config.use_cache = False
    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    return model, tokenizer


def resolve_cache_path(args: argparse.Namespace, split_name: str) -> str:
    if args.pseudo_label_cache:
        root, ext = os.path.splitext(args.pseudo_label_cache)
        if ext.lower() == ".jsonl":
            return f"{root}_{split_name}.jsonl"
        return os.path.join(args.pseudo_label_cache, f"pseudo_labels_top{args.top_k}_{split_name}.jsonl")
    return os.path.join(args.output_dir, f"pseudo_labels_top{args.top_k}_{split_name}.jsonl")


def generate_or_load_pseudo_labels(
    dataset: Dataset,
    engine: RewardSearchEngine,
    args: argparse.Namespace,
    split_name: str,
) -> Dict[Any, Dict[str, Any]]:
    cache_path = resolve_cache_path(args, split_name)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path) and not args.overwrite_pseudo_cache:
        logger.info("从缓存加载 %s pseudo labels: %s", split_name, cache_path)
        records: Dict[Any, Dict[str, Any]] = {}
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                records[item[args.id_field]] = item
        return records

    logger.info("开始为 %s 集生成 pseudo labels，共 %d 条样本", split_name, len(dataset))
    records: Dict[Any, Dict[str, Any]] = {}
    reward_values: List[float] = []
    exact_match_count = 0

    with open(cache_path, "w", encoding="utf-8") as writer:
        for idx, example in enumerate(dataset):
            result = engine.search(example)
            full_timeline = engine.clean_timeline(example["full_timeline"])
            pseudo_timeline = [full_timeline[i] for i in result.indices]
            pseudo_keys = [engine.event_key(e) for e in pseudo_timeline]
            gold_keys = [engine.event_key(e) for e in engine.clean_timeline(example["target_timeline"])]
            if pseudo_keys == gold_keys:
                exact_match_count += 1

            record = {
                args.id_field: example[args.id_field],
                "title": example["title"],
                "pseudo_timeline": pseudo_timeline,
                "pseudo_reward": result.total_reward,
                "reward_breakdown": result.reward_breakdown,
            }
            records[example[args.id_field]] = record
            reward_values.append(result.total_reward)
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (idx + 1) % 50 == 0 or idx + 1 == len(dataset):
                logger.info(
                    "[%s] 已生成 %d/%d，当前平均 pseudo reward=%.4f",
                    split_name,
                    idx + 1,
                    len(dataset),
                    sum(reward_values) / max(1, len(reward_values)),
                )

    logger.info(
        "%s pseudo labels 完成：平均 reward=%.4f，exact-match-to-gold=%.2f%%，缓存已写入 %s",
        split_name,
        sum(reward_values) / max(1, len(reward_values)),
        100.0 * exact_match_count / max(1, len(dataset)),
        cache_path,
    )
    return records


def attach_pseudo_timeline(dataset: Dataset, pseudo_records: Dict[Any, Dict[str, Any]], args: argparse.Namespace) -> Dataset:
    merged: List[Dict[str, Any]] = []
    for example in dataset:
        record = pseudo_records.get(example[args.id_field])
        if record is None:
            raise KeyError(f"未找到样本 {example[args.id_field]} 的 pseudo label")
        updated = dict(example)
        updated["pseudo_timeline"] = record["pseudo_timeline"]
        updated["pseudo_reward"] = record["pseudo_reward"]
        updated["pseudo_breakdown"] = record["reward_breakdown"]
        merged.append(updated)
    return Dataset.from_list(merged)


def build_tokenized_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
    target_source: str,
) -> Dataset:
    tokenized_records: List[Dict[str, Any]] = []
    for example in dataset:
        if target_source == "pseudo":
            target_timeline = example["pseudo_timeline"]
        elif target_source == "gold":
            target_timeline = example["target_timeline"]
        else:
            raise ValueError(f"未知 target_source: {target_source}")

        conversation = make_conversation_record(example, target_timeline, args.top_k)
        tokenized = tokenize_supervised_example(conversation, tokenizer, args)
        tokenized_records.append(tokenized)
    return Dataset.from_list(tokenized_records)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)

    if args.search_pool_size <= args.top_k:
        logger.warning("search_pool_size=%d 不大于 top_k=%d，自动调整为 %d", args.search_pool_size, args.top_k, args.top_k + 4)
        args.search_pool_size = args.top_k + 4
    elif args.top_k == 10 and args.search_pool_size < 20:
        logger.warning("top_k=10 时 search_pool_size=%d 偏小，建议使用 20-24。", args.search_pool_size)

    full_dataset = load_custom_dataset(args)
    train_dataset, eval_dataset = split_dataset(full_dataset, args)

    # 先离线生成 pseudo labels，避免与大模型训练抢占显存。
    logger.info("加载 embedding 模型用于离线 reward-search: %s", args.embedding_model_path)
    embed_model = SentenceTransformer(args.embedding_model_path, device=args.search_device)
    search_engine = RewardSearchEngine(embed_model, args)

    train_pseudo = generate_or_load_pseudo_labels(train_dataset, search_engine, args, "train")
    train_dataset = attach_pseudo_timeline(train_dataset, train_pseudo, args)

    if args.eval_target_source == "pseudo":
        eval_pseudo = generate_or_load_pseudo_labels(eval_dataset, search_engine, args, "eval")
        eval_dataset = attach_pseudo_timeline(eval_dataset, eval_pseudo, args)

    del embed_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model, tokenizer = load_model_and_tokenizer(args)
    train_tokenized = build_tokenized_dataset(train_dataset, tokenizer, args, target_source="pseudo")
    eval_target = "pseudo" if args.eval_target_source == "pseudo" else "gold"
    eval_tokenized = build_tokenized_dataset(eval_dataset, tokenizer, args, target_source=eval_target)

    collator = SupervisedDataCollator(tokenizer)

    bf16 = args.dtype == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = args.dtype == "float16"
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        # evaluation_strategy="steps" if len(eval_tokenized) > 0 else "no",
        eval_steps=args.save_steps,
        bf16=bf16,
        fp16=fp16,
        report_to=args.report_to,
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=0,
        push_to_hub=args.push_to_hub,
        optim="adamw_torch",
        # optim="paged_adamw_8bit",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    logger.info("开始 RS-SFT 训练：train=%d, eval=%d", len(train_tokenized), len(eval_tokenized))
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("RS-SFT 模型已保存至: %s", args.output_dir)


if __name__ == "__main__":
    main()
