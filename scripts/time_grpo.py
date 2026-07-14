import logging
import re
import torch
import json
from datasets import Dataset
from trl import (
    GRPOConfig,
    GRPOTrainer,
    TrlParser,
)
from peft import LoraConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from rouge_chinese import Rouge
import jieba
from src import time_value

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 命令行参数解析
parser = TrlParser()
parser.add_argument("--output_dir", type=str, default="./model_output/10_fold0", help="输出目录")
parser.add_argument("--model_name_or_path", type=str, default="model/Qwen3-8B", help="模型路径")
parser.add_argument("--dtype", type=str, default="bfloat16", help="数据类型")
parser.add_argument("--quantization", type=str, default="None", choices=[None, "4bit", "8bit"], help="量化方式")
parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="每个设备的训练批次大小")
parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="梯度累积步数")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
parser.add_argument("--logging_steps", type=int, default=10, help="日志打印步数")
parser.add_argument("--push_to_hub", action="store_true", help="是否推送到Hub")
parser.add_argument("--max_seq_length", type=int, default=8192, help="最大序列长度")
parser.add_argument("--id_field", type=str, default="id", help="数据集中用于匹配的ID字段名")
parser.add_argument("--top_k", type=int, default=10, choices=[5, 10], help="要选择的关键事件数量（例如 5 或 10）")
parser.add_argument("--questions_path", type=str, required=True, help="Input timeline JSONL")
parser.add_argument("--answers_path", type=str, required=True, help="Reference timeline JSONL")
# parser.add_argument("--dtype", type=str, default="float16", help="数据类型")
args = parser.parse_args()


################
# 模型加载 - 本地Qwen3-8B
################

rouge = Rouge()
def load_local_qwen_model(model_path):
    """加载本地Qwen3-8B模型，增加硬件兼容性处理"""
    # 配置量化
    bnb_config = None
    if args.quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    elif args.quantization == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # 加载tokenizer（修改内部变量名）
    local_tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        max_length=args.max_seq_length,
        enable_thinking=False,
        padding_side='left'  # 加上之前提到的左填充配置
    )
    if local_tokenizer.pad_token is None:
        local_tokenizer.pad_token = local_tokenizer.eos_token
        logger.info(f"已设置pad_token为eos_token: {local_tokenizer.pad_token}")

    # 验证配置是否生效
    logger.info(f"Tokenizer padding_side: {local_tokenizer.padding_side}")

    # 自动选择兼容的dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    target_dtype = dtype_map.get(args.dtype, torch.bfloat16)

    # 检查硬件是否支持bfloat16
    if target_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        logger.warning("当前设备不支持bfloat16，自动切换为float16")
        target_dtype = torch.float16

    # 加载模型
    local_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=target_dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )

    return local_model, local_tokenizer  # 返回修改后的内部变量


model, tokenizer = load_local_qwen_model(args.model_name_or_path)


class StepTracker:
    def __init__(self):
        self.global_step = 0


from transformers import TrainerCallback, TrainerState, TrainerControl


class StepUpdateCallback(TrainerCallback):
    def __init__(self, tracker):
        self.tracker = tracker

    def on_step_end(self, args, state, control, **kwargs):
        self.tracker.global_step = state.global_step

# 1. 创建全局追踪器

global_step_tracker = StepTracker()

################
# PEFT配置
################


def get_lora_config():
    """获取LoRA配置（确认Qwen3-8B模块兼容性）"""
    return LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_custom_dataset(questions_path, answers_path):
    """加载自定义数据集（通过ID匹配输入和答案），增加鲁棒性处理"""
    def read_jsonl(file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):  # 记录行号，方便定位错误
                try:
                    item = json.loads(line.strip())
                    # 检查是否包含ID字段
                    if args.id_field not in item:
                        logger.warning(f"JSONL文件{file_path}第{line_num}行：缺少ID字段（{args.id_field}），跳过该行")
                        continue
                    data.append(item)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSONL文件{file_path}第{line_num}行：解析失败: {e}，跳过该行")
        return data

    # 读取输入和答案数据
    questions_data = read_jsonl(questions_path)
    answers_data = read_jsonl(answers_path)
    logger.info(f"成功读取输入数据{len(questions_data)}条，答案数据{len(answers_data)}条")

    # 构建答案字典：key=ID，value=答案item（保留第一个重复ID）
    answers_dict = {}
    duplicate_ids = set()
    for item in answers_data:
        data_id = item[args.id_field]
        if data_id not in answers_dict:
            answers_dict[data_id] = item
        else:
            if data_id not in duplicate_ids:
                logger.warning(f"答案数据中检测到重复ID: {data_id}，保留第一个实例")
                duplicate_ids.add(data_id)

    # 构建训练数据集（通过ID匹配输入和答案）
    examples = []
    missing_answer_ids = set()
    for item in questions_data:
        data_id = item[args.id_field]
        title = item.get('title', f"无标题_{data_id}")  # 标题可选，无则用ID填充

        # 匹配对应答案
        if data_id in answers_dict:
            examples.append({
                args.id_field: data_id,  # 保留ID字段
                'title': title,
                'full_timeline': item.get('timeline', []),  # 输入的完整时间线
                'top5_timeline': answers_dict[data_id].get('timeline', [])  # 答案的Top5时间线
            })
        else:
            if data_id not in missing_answer_ids:
                logger.warning(f"输入数据ID {data_id} 未找到对应的答案，跳过该条数据")
                missing_answer_ids.add(data_id)

    logger.info(f"最终构建训练样本{len(examples)}条（跳过缺失答案的样本{len(missing_answer_ids)}条）")
    return Dataset.from_list(examples)


# 加载数据集（通过ID匹配）
full_dataset = load_custom_dataset(
    questions_path=args.questions_path,
    answers_path=args.answers_path
)

# 分割训练集和验证集（按ID独立分割，避免数据泄露）
num_folds = 5          # 固定为5折
test_fold_idx = 0      # 选择第1折作为测试集（0-4可选，对应5折）


# 计算每折的大小和索引范围
total_size = len(full_dataset)
fold_size = total_size // num_folds
remaining = total_size % num_folds  # 不能整除时，前remaining折各多1个样本

# 生成各折的索引切片
folds = []
start = 0
for i in range(num_folds):
    current_fold_size = fold_size + 1 if i < remaining else fold_size
    end = start + current_fold_size
    folds.append( (start, end) )  # 存储每折的起始和结束索引
    start = end

# 提取测试集（指定的那一折）
test_start, test_end = folds[test_fold_idx]
eval_dataset = full_dataset.select(range(test_start, test_end))

# 提取训练集（除测试折外的所有数据）
train_indices = []
for i in range(num_folds):
    if i != test_fold_idx:
        fold_start, fold_end = folds[i]
        train_indices.extend(range(fold_start, fold_end))
train_dataset = full_dataset.select(train_indices)
logger.info(f"训练集{len(train_dataset)}条，验证集{len(eval_dataset)}条")

# 提示词模板
INSTRUCTION = f""" 你是一个专业的新闻编辑助手，你的任务是从给定的新闻话题时间线中，**挑选出最重要的{args.top_k}个关键事件摘要**，帮助读者快速掌握该事件的核心发展脉络。
请遵循以下规则：
1. 仅从提供的 `timeline` 列表中选择事件，**不得编造或修改原文摘要内容**。
2. 优先选择具有**标志性、转折性、公众广泛关注或引发重大后续影响**的事件。
3. 输出按照<answer>JSON</answer>格式，JSON包含 `title` 和 `top_k_timeline` 字段（注意：字段名从 top_5_timeline 变为 top_k_timeline），其中 `top_k_timeline` 是一个包含{args.top_k}个对象的列表，每个对象包含 `time` 和 `summary` 字段。
4. 保持时间顺序（按 `time` 升序排列）。
输出格式：
<answer>
{{
  "title": "新闻标题",
  "top_k_timeline": [
    {{"time": "时间1", "summary": "摘要1"}},
    {{"time": "时间2", "summary": "摘要2"}},
    // ... 共{args.top_k}个
  ]
}}
</answer> """


def format_timeline_for_display(timeline_data):
    """格式化时间线用于模型输入"""
    formatted = []
    for i, event in enumerate(timeline_data):
        time = event.get('time', '未知时间')
        summary = event.get('summary', '')
        formatted.append(f"{i + 1}. 时间: {time}\n   摘要: {summary}\n")
    return "\n\n".join(formatted)


def format_timeline_for_output(timeline_data, title):
    """安全格式化输出JSON，避免转义问题"""
    simplified_timeline = []
    for event in timeline_data:
        simplified_timeline.append({
            "time": event.get('time', ''),
            "summary": event.get('summary', '')
        })
    return json.dumps({
        "title": title,
        "top_k_timeline": simplified_timeline # 键名修改为 top_k_timeline
    }, ensure_ascii=False, indent=2)


def make_conversation(example):
    """构建对话格式，修复标签错误"""
    full_timeline_str = format_timeline_for_display(example['full_timeline'])
    user_content = f"/no_think \n{INSTRUCTION}\n\n标题: {example['title']}\n\ntimeline:\n{full_timeline_str}/no_think"

    # 生成正确格式的期望输出（思考块+输出块）
    expected_output = format_timeline_for_output(example['top5_timeline'], example['title'])
    # 注意: 数据集加载部分 'top5_timeline' 字段可能需要根据实际情况修改
    # 假设 'filter_output.jsonl' 仍然是 Top5，如果需要 Top10 训练，数据源也要更换
    # 目前假设目标是：使用 Top5 数据集，但模型提示词和奖励函数支持 TopK (K=5 或 10)

    # 修复：添加明确的思维链标记
    expected_response = f"\n\n{expected_output}"

    # >>> 修改返回字段，使用 'top_k_timeline' 以匹配新的 JSON 结构 <<<
    return {
        "prompt": [
            {"role": "system", "content": "/no_think你是一个专业的新闻编辑助手。"},
            {"role": "user", "content": user_content},
        ],
        "response": expected_response,  # 这里应该是模型应该输出的完整内容
        "title": example['title'],
        "full_timeline": example['full_timeline'],
        "top_k_timeline": example['top5_timeline'] # 字段名修改以匹配INSTRUCTION，但数据内容仍然是 Top5
    }


# 应用数据转换
train_dataset = train_dataset.map(make_conversation)
eval_dataset = eval_dataset.map(make_conversation)


################
# 自定义奖励函数
################
def extract_answer_json(response):
    """从响应中提取answer标签内的JSON内容"""
    try:
        # 查找<answer>标签内的内容
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            json_str = answer_match.group(1).strip()
            # return json.loads(json_str) # 不直接返回，继续尝试解析

            # >>> 确保尝试解析时，检查的键是 top_k_timeline，而不是 top_5_timeline <<<
            parsed = json.loads(json_str)
            if 'top_k_timeline' in parsed:
                return parsed

        # 如果没有answer标签或解析失败，尝试直接解析整个响应中的JSON
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0).strip())
            # >>> 确保检查的键是 top_k_timeline <<<
            if 'top_k_timeline' in parsed:
                return parsed

    except Exception as e:
        print(f"JSON提取错误: {e}")
    return None


def timeline_selection_reward(response, reference_topk):  # 参数名改为 reference_topk
    """分别计算日期准确性和文本Rouge-1分数"""
    date_reward = 0.0
    text_reward = 0.0
    time_list_pre = []

    # 获取 Top K 数量
    K = args.top_k

    try:
        parsed = extract_answer_json(response)
        # >>> 检查键名为 'top_k_timeline' <<<
        if not parsed or 'top_k_timeline' not in parsed:
            return date_reward, text_reward, time_list_pre

        selected_events = parsed['top_k_timeline']  # 键名修改

        # >>> 检查数量是否等于 K <<<
        if len(selected_events) != K:
            # 如果数量不匹配 K，可以考虑返回低分或 0.0
            # 这里先按要求继续计算，但如果数量不匹配，后续奖励计算可能不准确
            pass

            # 准备参考数据
        reference_events = [
            {
                'time': event.get('time', '').strip(),
                'summary': event.get('summary', '').strip(),
                'matched': False  # 标记是否已被候选事件匹配
            }
            for event in reference_topk
            if event.get('time', '').strip() and event.get('summary', '').strip()  # 过滤无效参考事件
        ]

        # 准备候选数据（每个候选事件的时间和摘要）
        selected_events_clean = [
            {
                'time': event.get('time', '').strip(),
                'summary': event.get('summary', '').strip()
            }
            for event in selected_events
        ]
        time_list_pre = [item['time'] for item in selected_events_clean]
        # 1. 计算日期准确性（参考事件仅匹配一次，避免重复计分）
        date_scores = []
        for sel_event in selected_events_clean:
            sel_time = sel_event['time']
            sel_sum = sel_event['summary']
            max_score = 0.0
            matched_ref_index = -1

            # 遍历未匹配的参考事件寻找最优匹配
            for idx, ref_event in enumerate(reference_events):
                if ref_event['matched'] or not sel_time or not sel_sum:
                    continue

                ref_time = ref_event['time']
                ref_sum = ref_event['summary']
                current_date_score = 0.0

                # 日期匹配规则（梯度不变）
                if sel_time == ref_time:
                    current_date_score = 1.0
                elif sel_time.replace('-', '') == ref_time.replace('-', ''):
                    current_date_score = 0.9
                elif sel_time in ref_time or ref_time in sel_time:
                    current_date_score = 0.7

                # 更新最优匹配
                if current_date_score > 0 and sel_sum == ref_sum and current_date_score > max_score:
                    max_score = current_date_score
                    matched_ref_index = idx

            # 标记已匹配的参考事件
            if matched_ref_index != -1:
                reference_events[matched_ref_index]['matched'] = True

            date_scores.append(max_score)

        # 日期奖励平均分
        date_reward = sum(date_scores) / len(date_scores) if date_scores else 0.0

        # 2. 计算文本Rouge-1分数（含重复文本惩罚机制）
        selected_summaries = [event['summary'] for event in selected_events_clean if event['summary']]
        total_candidate_count = len(selected_summaries)  # 有效候选摘要总数（非空）

        # 步骤1：统计摘要重复情况
        summary_count = {}  # key: 摘要文本, value: 出现次数
        for summary in selected_summaries:
            summary_count[summary] = summary_count.get(summary, 0) + 1

        # 步骤2：计算重复惩罚系数（重复越多，惩罚越重）
        repeat_count = 0  # 重复的总次数（例：出现3次的摘要贡献2次重复）
        for count in summary_count.values():
            if count > 1:
                repeat_count += (count - 1)  # 仅统计超出1次的重复部分

        # 惩罚系数计算：范围[0.0, 1.0]，无重复→1.0（无惩罚），全重复→0.0（最大惩罚）
        if total_candidate_count == 0:
            penalty_factor = 1.0  # 无有效摘要时无惩罚
        else:
            penalty_factor = 1.0 - (repeat_count / total_candidate_count)
            penalty_factor = max(0.0, penalty_factor)  # 确保惩罚系数不小于0

        # 步骤3：去重后的摘要用于计算基础Rouge-1分数
        unique_selected_summaries = list(summary_count.keys())
        selected_full_text = " ".join(unique_selected_summaries)
        reference_full_text = " ".join([event['summary'] for event in reference_events if event['summary']])

        # 步骤4：计算基础Rouge-1分数，再乘以惩罚系数
        base_rouge_score = 0.0
        if selected_full_text and reference_full_text:
            selected_full_text = ' '.join(jieba.cut(selected_full_text))
            reference_full_text = ' '.join(jieba.cut(reference_full_text))
            base_rouge_score = rouge.get_scores(selected_full_text, reference_full_text)[0]['rouge-1']['f']

        # 应用惩罚：最终文本奖励 = 基础分数 × 惩罚系数
        text_reward = base_rouge_score * penalty_factor

        # 可选：打印调试信息
        # print(f"重复次数: {repeat_count}, 惩罚系数: {penalty_factor:.3f}, 基础Rouge-1: {base_rouge_score:.3f}, 最终文本奖励: {text_reward:.3f}")

    except Exception as e:
        print(f"时间线选择奖励计算错误: {e}")

    return date_reward, text_reward, time_list_pre


def answer_format_reward(response):
    """优化格式奖励：更严格的JSON结构检查+无标签场景处理"""
    reward = 0.0
    K = args.top_k # 获取 K
    pattern = r'<answer>(.*?)</answer>'
    try:
        # 优先检查answer标签
        answer_match = re.search(pattern, response, re.DOTALL)
        json_content = None

        if answer_match:
            reward+=0.1
            json_content = answer_match.group(1).strip()
        else:
            # 无标签时提取纯JSON（增强正则匹配，避免截断）
            json_match = re.search(r'\{[\s\S]*\}', response)  # 匹配完整大括号对
            if json_match:
                json_content = json_match.group(0).strip()

        if json_content:
            parsed = json.loads(json_content)
            if isinstance(parsed, dict) and "title" in parsed:
                title = parsed['title']
                if title != '新闻标题':
                    reward += 0.05
            # 检查核心结构：必须是dict，包含 top_k_timeline 且为列表
            # >>> 检查键名为 'top_k_timeline' <<<
            if isinstance(parsed, dict) and "top_k_timeline" in parsed:
                timeline = parsed["top_k_timeline"]  # 键名修改
                if isinstance(timeline, list):
                    # 检查每个事件的字段完整性
                    valid_events = [
                        e for e in timeline
                        if isinstance(e, dict) and "time" in e and "summary" in e
                    ]
                    # 奖励与有效事件数量挂钩（满 K 个且全有效得 0.8 * K/5）
                    # 奖励调整：基于 K，按比例调整奖励上限
                    reward = reward + min(len(valid_events) / K, 0.8)

                    # 额外奖励：严格符合 K 个事件
                    if len(timeline) == K and len(valid_events) == K:
                        reward += 0.05  # 鼓励精准性
                    reward = min(reward, 1.0)  # 上限1.0
                else:
                    reward = 0.3  # top_k_timeline不是列表
            else:
                reward = 0.2  # 缺少核心字段
    except json.JSONDecodeError:
        reward = 0.1  # 格式错误但有JSON结构
    except Exception as e:
        logger.warning(f"格式奖励解析错误: {e}")

    return reward


def combined_reward_func(prompts, completions, **kwargs):
    """组合奖励函数"""
    current_step = global_step_tracker.global_step

    # 定义权重阶段
    THRESHOLD_STEP = 500

    # 动态权重分配逻辑
    if current_step < THRESHOLD_STEP:
        format_weight = 0.1
        date_weight = 0.5
        text_weight = 0.2
        time_weight = 0.2
    else:
        format_weight = 0.05
        date_weight = 0.55
        text_weight = 0.2
        time_weight = 0.2
    rewards = []
    examples = kwargs.get('top_k_timeline', [])
    ref_timeline =  kwargs.get('full_timeline', [])[0]
    ref_timek_time = [item['time'] for item in examples[0]]  # 仍然使用 examples[0]
    time_list = [item['time'] for item in ref_timeline]
    dect = time_value.detect_uniformity_no_deduplicate(time_list)
    for i, completion in enumerate(completions):
        # print(f"\n=== 评估第 {i + 1} 个响应 ===")
        completion = completion[0]['content']
        # 初始化各维度分数
        format_reward = answer_format_reward(completion)
        date_reward = 0.0
        text_reward = 0.0
        time_reward = 0.0

        # 计算时间线选择质量
        if i < len(examples):
            date_reward, text_reward, time_list_pre = timeline_selection_reward(completion, examples[i])
            if time_list_pre == []:
                time_reward = 0.0
            else:
                # >>> 传入的参考 Top K 时间是 ref_timek_time <<<
                time_reward = time_value.adaptive_time_reward_no_deduplicate(time_list_pre, ref_timek_time, time_list,
                                                                           dect)
        else:
            print("警告: 没有找到参考时间线数据")

        # 组合奖励：格式(10%) + 节点准确性(40%) + 语义相似度(30%) + 时间间隔(20%)
        total_reward = (
                format_weight * format_reward +
                date_weight * date_reward +
                text_weight * text_reward +
                time_weight * time_reward
        )

        rewards.append(total_reward)
        print(f"总奖励: {total_reward:.3f} (格式:{format_reward:.3f}, 日期:{date_reward:.3f}, 时间覆盖:{time_reward:.3f}，文本:{text_reward:.3f})")
    max_index = rewards.index(max(rewards))
    ans = completions[max_index][0]['content']
    print(ans)

    return torch.tensor(rewards, dtype=torch.float32)



# 训练配置
training_args = GRPOConfig(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    lr_scheduler_type="cosine",
    max_prompt_length=8192,
    num_train_epochs=5,
    num_generations=8,
    max_completion_length=1024,
    learning_rate=args.learning_rate,
    logging_steps=args.logging_steps,
    push_to_hub=args.push_to_hub,
    shuffle_dataset=False,
    bf16=True,
    save_steps=100,  # 每100步保存一次模型
    save_total_limit=8
)

# 训练器



if __name__ == "__main__":
    step_callback = StepUpdateCallback(global_step_tracker)
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_lora_config(),
        reward_funcs=[combined_reward_func],
        callbacks=[step_callback]
    )
    trainer.train()
    # # 保存模型（包含PEFT权重）
    trainer.save_model(training_args.output_dir)
    logger.info(f"模型已保存至: {training_args.output_dir}")

