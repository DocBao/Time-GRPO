import json
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_jsonl(file_path):
    """读取 JSONL 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                logger.warning(f"无法解析文件 {file_path} 第 {i + 1} 行，跳过。")
    return data


def convert_to_ppo_format(
        input_file,
        answer_file,
        output_file,
        id_field="id",
        top_k=5
):
    """
    读取输入和答案文件，通过 id 匹配，转换为 PPO 训练格式。
    """
    logger.info(f"正在读取输入文件: {input_file}")
    questions_data = read_jsonl(input_file)

    logger.info(f"正在读取答案文件: {answer_file}")
    answers_data = read_jsonl(answer_file)

    # 1. 构建答案索引字典 (Hash Map) 以便快速查找
    # 格式: { "id_123": { ...answer_item... }, ... }
    answers_map = {
        item[id_field]: item
        for item in answers_data
        if id_field in item
    }
    logger.info(f"加载了 {len(answers_map)} 条答案数据")

    ppo_data = []
    matched_count = 0
    missing_count = 0

    # 2. 遍历问题，匹配答案并构建 Prompt
    for item in questions_data:
        data_id = item.get(id_field)

        # 如果找不到对应的 ID 或 ID 不在答案中，则跳过
        if not data_id or data_id not in answers_map:
            missing_count += 1
            continue

        matched_count += 1
        answer_item = answers_map[data_id]

        # --- 构建 Prompt (与您 GRPO 代码中的逻辑保持一致) ---
        title = item.get('title', f"无标题_{data_id}")
        timeline_list = item.get('timeline', [])

        timeline_str = "\n\n".join([
            f"{i + 1}. 时间: {event.get('time', '未知时间')}\n   摘要: {event.get('summary', '')}"
            for i, event in enumerate(timeline_list)
        ])

        # 这里使用纯文本 Prompt。
        # 如果模型需要 Chat 格式 (<|im_start|>user...), run_ppo.py 中的 tokenizer 会处理，
        # 或者您可以在这里直接写好格式。这里我们只准备内容。
        instruction = (
            f"你是一个专业的新闻编辑助手，你的任务是从给定的新闻话题时间线中，"
            f"**挑选出最重要的{top_k}个关键事件摘要**，帮助读者快速掌握该事件的核心发展脉络。\n"
            f"请遵循以下规则：\n"
            f"1. 仅从提供的 `timeline` 列表中选择事件，**不得编造或修改原文摘要内容**。\n"
            f"2. 优先选择具有**标志性、转折性、公众广泛关注或引发重大后续影响**的事件。\n"
            f"3. 输出按照<answer>JSON</answer>格式，JSON包含 `title` 和 `top_k_timeline` 字段。\n"
            f"4. 保持时间顺序（按 `time` 升序排列）。\n\n"
            f"标题: {title}\n\n"
            f"timeline:\n{timeline_str}\n\n"
            f"请输出答案："
        )

        # --- 提取 Ground Truth (Reference) ---
        # 这里的 top5_timeline 就是我们在奖励函数中要用来做对比的“正确答案”
        # 注意：字段名可能在不同文件里不同，这里沿用您代码里的 key
        reference_timeline = answer_item.get('timeline', [])  # 或者是 'top5_timeline'，取决于您的 jsonl 结构

        # 如果答案为空，通常也应该跳过，否则奖励函数会出错
        if not reference_timeline:
            logger.warning(f"ID {data_id} 匹配成功但在答案文件中缺少 timeline 字段")
            continue

        ppo_data.append({
            "query": instruction,
            # 将 list 转为 json string 存储，防止 Dataset 加载时因为长度不一报错
            # 在 run_ppo.py 的奖励函数里再 json.loads 回来
            "reference_data": json.dumps(reference_timeline, ensure_ascii=False)
        })

    # 3. 保存结果
    logger.info(f"匹配完成: 成功 {matched_count} 条, 缺失 {missing_count} 条")
    logger.info(f"正在写入输出文件: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ppo_data, f, ensure_ascii=False, indent=2)

    logger.info("转换结束。")


if __name__ == "__main__":
    # 请根据实际路径修改
    convert_to_ppo_format(
        input_file="data_us/filter_input.jsonl",
        answer_file="data_us/filter_output.jsonl",
        output_file="data_us/ppo_dataset.json",
        id_field="id"
    )