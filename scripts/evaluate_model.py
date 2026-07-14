import argparse
import json
import logging
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import jieba
import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger("eval")


logger = build_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast batched multi-GPU evaluation for Acc / ROUGE / STAR")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--peft_model_path", type=str, default=None)
    parser.add_argument("--embed_model_path", type=str, default=None)
    parser.add_argument("--questions_path", type=str, default="data_us/filter_input.jsonl")
    parser.add_argument("--answers_path", type=str, default="data_us/filter_output.jsonl")
    parser.add_argument("--output_jsonl_path", type=str, default="data/evaluation_results_comprehensive.jsonl")
    parser.add_argument("--num_key_events", type=int, default=5, choices=[5, 10])
    parser.add_argument("--id_field", type=str, default="id")
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--test_fold_idx", type=int, default=0)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--embed_device", type=str, default="same", choices=["same", "cpu", "cuda"])
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--merge_only", action="store_true", help="Only merge shard result files and report final metrics")
    return parser.parse_args()


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name == "bfloat16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        logger.warning("当前 GPU 不支持 bfloat16，自动回退到 float16")
        return torch.float16
    return mapping[dtype_name]


def date_to_float(date_str: str) -> float:
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(date_str.strip(), fmt).timestamp()
        except Exception:
            pass
    return 0.0


# 加速 jieba，避免首次切词的冷启动影响计时
jieba.initialize()


def chinese_tokenize(text: str) -> str:
    return " ".join(jieba.cut(text))


def get_instruction(n: int) -> str:
    return f"""
    你是一个专业的新闻编辑助手，你的任务是从给定的新闻话题时间线中，**挑选出最重要的{n}个关键事件摘要**，帮助读者快速掌握该事件的核心发展脉络。
请遵循以下规则：
1. 仅从提供的 `timeline` 列表中选择事件，**不得编造或修改原文摘要内容**。
2. 优先选择具有**标志性、转折性、公众广泛关注或引发重大后续影响**的事件。
3. 输出按照<answer>JSON</answer>格式，JSON包含 `title` 和 `top_k_timeline` 字段，其中 `top_k_timeline` 是一个包含{n}个对象的列表，每个对象包含 `time` 和 `summary` 字段。
4. 保持时间顺序（按 `time` 升序排列）。
输出格式：
<answer>
{{
  "title": "新闻标题",
  "top_k_timeline": [
    {{"time": "时间1", "summary": "摘要1"}},
    {{"time": "时间2", "summary": "摘要2"}}
  ]
}}
</answer>
"""


def format_input(example: Dict[str, Any], num_key_events: int, id_field: str) -> Dict[str, Any]:
    tl = "\n".join(
        [f"{i + 1}. 时间：{e.get('time', '')} 摘要：{e.get('summary', '')}" for i, e in enumerate(example["full_timeline"])]
    )
    prompt = [
        {"role": "system", "content": "你是一个专业的新闻编辑助手，请按要求只输出JSON格式的答案"},
        {"role": "user", "content": f"{get_instruction(num_key_events)}\n\n标题: {example['title']}\ntimeline:\n{tl}"},
    ]
    ref = example["top_timeline"]
    ref_text = chinese_tokenize(" ".join([r.get("summary", "") for r in ref[:num_key_events]]))
    return {
        "prompt": prompt,
        "ref": ref,
        "ref_text": ref_text,
        "id": example[id_field],
        "title": example["title"],
    }


def extract_json(text: str, n: int) -> List[Dict[str, str]]:
    try:
        answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if answer_match:
            json_obj = json.loads(answer_match.group(1).strip())
        else:
            json_obj = json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0))
        return json_obj.get(f"top_{n}_timeline", json_obj.get("top_k_timeline", []))
    except Exception:
        return []


def calculate_star_score(preds: Sequence[Dict[str, str]], embed_model: SentenceTransformer, embed_device: str) -> float:
    if len(preds) < 2:
        return 0.0

    timestamps = [date_to_float(p.get("time", "")) for p in preds]
    summaries = [p.get("summary", "") for p in preds]

    if len(summaries) < 2:
        return 0.0

    ts_tensor = torch.tensor(timestamps, dtype=torch.float32, device=embed_device)
    intervals = torch.relu(ts_tensor[1:] - ts_tensor[:-1])
    log_intervals = torch.log(intervals + 1.0)
    time_dist_sum = torch.sum(log_intervals)
    if time_dist_sum < 1e-8:
        return 0.0
    time_dist = log_intervals / time_dist_sum

    with torch.inference_mode():
        embeddings = embed_model.encode(summaries, convert_to_tensor=True, device=embed_device)
        cos_sim = F.cosine_similarity(embeddings[:-1], embeddings[1:])
        sem_distance = 1.0 - cos_sim
        sem_dist_sum = torch.sum(sem_distance)
        if sem_dist_sum < 1e-8:
            return 0.0
        sem_dist = sem_distance / sem_dist_sum

    alignment_error = torch.sum(torch.abs(time_dist - sem_dist))
    star_reward = 0.5 * alignment_error
    star_reward = 1.0 - torch.pow(star_reward, 0.5)
    return float(torch.clamp(star_reward, 0.0, 1.0).cpu().item())


def calculate_metrics(pred_timeline: Sequence[Dict[str, str]], ref_timeline: Sequence[Dict[str, str]], ref_text_tok: str,
                      scorer: rouge_scorer.RougeScorer, embed_model: SentenceTransformer, embed_device: str) -> Dict[str, float]:
    correct = 0
    ref_pairs = {(r.get("time"), r.get("summary")) for r in ref_timeline}
    for p in pred_timeline:
        if (p.get("time"), p.get("summary")) in ref_pairs:
            correct += 1
    accuracy = correct / len(pred_timeline) if pred_timeline else 0.0

    pred_text = chinese_tokenize(" ".join([p.get("summary", "") for p in pred_timeline]))
    rouge_scores = scorer.score(ref_text_tok, pred_text)
    star = calculate_star_score(pred_timeline, embed_model, embed_device)

    return {
        "accuracy": accuracy,
        "rouge1": rouge_scores["rouge1"].fmeasure,
        "rouge2": rouge_scores["rouge2"].fmeasure,
        "star_score": star,
        "match_count": correct,
    }


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_eval_examples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    q_data = {x[args.id_field]: x for x in read_jsonl(args.questions_path)}
    a_data = {x[args.id_field]: x for x in read_jsonl(args.answers_path)}

    matched = [
        format_input(
            {
                args.id_field: k,
                "title": v.get("title", ""),
                "full_timeline": v.get("timeline", []),
                "top_timeline": a_data[k].get("timeline", []),
            },
            num_key_events=args.num_key_events,
            id_field=args.id_field,
        )
        for k, v in q_data.items()
        if k in a_data
    ]

    # 与训练脚本一致的连续切片式 5-fold
    total_size = len(matched)
    fold_size = total_size // args.num_folds
    remaining = total_size % args.num_folds
    folds: List[tuple[int, int]] = []
    start = 0
    for i in range(args.num_folds):
        current_fold_size = fold_size + (1 if i < remaining else 0)
        end = start + current_fold_size
        folds.append((start, end))
        start = end

    test_start, test_end = folds[args.test_fold_idx]
    eval_list = matched[test_start:test_end]

    if args.num_shards > 1:
        eval_list = eval_list[args.shard_id::args.num_shards]

    logger.info(
        "总样本=%d, 折=%d/%d, 当前 shard=%d/%d, 本次评测样本=%d",
        total_size,
        args.test_fold_idx,
        args.num_folds,
        args.shard_id,
        args.num_shards,
        len(eval_list),
    )
    return eval_list


def get_output_path(base_path: str, num_shards: int, shard_id: int) -> str:
    if num_shards <= 1:
        return base_path
    p = Path(base_path)
    stem = p.stem
    suffix = p.suffix or ".jsonl"
    return str(p.with_name(f"{stem}.shard{shard_id}{suffix}"))


def load_model_and_tokenizer(args: argparse.Namespace, device: str):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = get_torch_dtype(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
        device_map={"": device},
    )
    if args.peft_model_path:
        model = PeftModel.from_pretrained(model, args.peft_model_path)
        model = model.merge_and_unload()
        model.to(device)
    model.eval()
    return model, tokenizer


def get_embed_device(args: argparse.Namespace, model_device: str) -> str:
    if args.embed_device == "same":
        return model_device
    if args.embed_device == "cuda":
        return model_device
    return "cpu"


def evaluate_batch(model, tokenizer, batch_examples: Sequence[Dict[str, Any]], scorer, embed_model, model_device: str,
                   embed_device: str, args: argparse.Namespace) -> List[Dict[str, Any]]:
    input_texts = [
        tokenizer.apply_chat_template(ex["prompt"], tokenize=False, add_generation_prompt=True)
        for ex in batch_examples
    ]
    inputs = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_seq_length,
    )
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = outputs[:, prompt_len:]
    gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

    batch_results: List[Dict[str, Any]] = []
    for ex, gen_text in zip(batch_examples, gen_texts):
        preds = extract_json(gen_text, args.num_key_events)[:args.num_key_events]
        refs = ex["ref"][:args.num_key_events]
        metrics = calculate_metrics(preds, refs, ex["ref_text"], scorer, embed_model, embed_device)
        batch_results.append({
            "id": ex["id"],
            "title": ex["title"],
            "metrics": metrics,
            "pred": preds,
            "ref": refs,
        })
    return batch_results


def summarize_results(results: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not results:
        return {"accuracy": 0.0, "rouge1": 0.0, "rouge2": 0.0, "star_score": 0.0}
    return {
        "accuracy": float(np.mean([r["metrics"]["accuracy"] for r in results])),
        "rouge1": float(np.mean([r["metrics"]["rouge1"] for r in results])),
        "rouge2": float(np.mean([r["metrics"]["rouge2"] for r in results])),
        "star_score": float(np.mean([r["metrics"]["star_score"] for r in results])),
    }


def save_results_incrementally(results: Sequence[Dict[str, Any]], path: str, mode: str = "a") -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def merge_shard_files(base_path: str, num_shards: int) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for shard_id in range(num_shards):
        shard_path = get_output_path(base_path, num_shards, shard_id)
        if not os.path.exists(shard_path):
            logger.warning("缺少 shard 文件: %s", shard_path)
            continue
        merged.extend(read_jsonl(shard_path))
    merged.sort(key=lambda x: str(x.get("id", "")))
    if merged:
        save_results_incrementally(merged, base_path, mode="w")
    return merged


def main() -> None:
    args = parse_args()

    if args.merge_only:
        merged = merge_shard_files(args.output_jsonl_path, args.num_shards)
        summary = summarize_results(merged)
        logger.info("合并后样本数: %d", len(merged))
        logger.info("Node Accuracy: %.4f", summary["accuracy"])
        logger.info("ROUGE-1:       %.4f", summary["rouge1"])
        logger.info("ROUGE-2:       %.4f", summary["rouge2"])
        logger.info("STAR Score:    %.4f", summary["star_score"])
        return

    if not args.model_name_or_path:
        raise ValueError("--model_name_or_path 不能为空")
    if not args.embed_model_path:
        raise ValueError("--embed_model_path 不能为空")

    if not torch.cuda.is_available():
        raise RuntimeError("当前脚本要求 CUDA。")
    if args.gpu_id >= torch.cuda.device_count():
        raise ValueError(f"gpu_id={args.gpu_id} 超过可见 GPU 数量 {torch.cuda.device_count()}")
    if args.shard_id >= args.num_shards:
        raise ValueError("shard_id 必须小于 num_shards")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = f"cuda:{args.gpu_id}"
    torch.cuda.set_device(args.gpu_id)

    logger.info(
        "Model=%s | PEFT=%s | K=%d | fold=%d | shard=%d/%d | batch=%d | device=%s",
        args.model_name_or_path,
        args.peft_model_path,
        args.num_key_events,
        args.test_fold_idx,
        args.shard_id,
        args.num_shards,
        args.batch_size,
        device,
    )

    eval_list = load_eval_examples(args)
    output_path = get_output_path(args.output_jsonl_path, args.num_shards, args.shard_id)
    if os.path.exists(output_path):
        os.remove(output_path)

    model, tokenizer = load_model_and_tokenizer(args, device=device)
    embed_device = get_embed_device(args, device)
    embed_model = SentenceTransformer(args.embed_model_path, device=embed_device)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    all_results: List[Dict[str, Any]] = []
    start_time = datetime.now()
    total = len(eval_list)

    for start in range(0, total, args.batch_size):
        batch_examples = eval_list[start:start + args.batch_size]
        batch_results = evaluate_batch(
            model=model,
            tokenizer=tokenizer,
            batch_examples=batch_examples,
            scorer=scorer,
            embed_model=embed_model,
            model_device=device,
            embed_device=embed_device,
            args=args,
        )
        all_results.extend(batch_results)

        if (len(all_results) % args.save_every == 0) or (start + len(batch_examples) >= total):
            save_results_incrementally(batch_results, output_path, mode="a")
        else:
            save_results_incrementally(batch_results, output_path, mode="a")

        elapsed = (datetime.now() - start_time).total_seconds()
        processed = start + len(batch_examples)
        speed = processed / max(elapsed, 1e-6)
        logger.info(
            "[%d/%d] shard=%d | speed=%.2f samples/s | last Acc=%.3f | last STAR=%.3f",
            processed,
            total,
            args.shard_id,
            speed,
            float(np.mean([x["metrics"]["accuracy"] for x in batch_results])) if batch_results else 0.0,
            float(np.mean([x["metrics"]["star_score"] for x in batch_results])) if batch_results else 0.0,
        )

    summary = summarize_results(all_results)
    logger.info("=" * 60)
    logger.info("Shard %d 评测完成，结果文件: %s", args.shard_id, output_path)
    logger.info("Node Accuracy: %.4f", summary["accuracy"])
    logger.info("ROUGE-1:       %.4f", summary["rouge1"])
    logger.info("ROUGE-2:       %.4f", summary["rouge2"])
    logger.info("STAR Score:    %.4f", summary["star_score"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
