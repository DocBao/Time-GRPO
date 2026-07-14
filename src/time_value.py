import datetime
import numpy as np


def detect_uniformity_no_deduplicate(all_times: list[str]) -> str:
    """
    不做时间去重，检测全量事件的时间分布均匀性（保留相同时间点）
    输入：该条数据的全量原始时间字符串列表（含重复时间）
    输出：均匀性等级（"均匀"/"较均匀"/"不均匀"）
    """
    # 1. 预处理：排序+转时间戳（秒级），不做去重
    if len(all_times) < 2:
        return "不均匀"  # 事件数过少，默认按不均匀处理

    def str2ts(time_str):
        return datetime.datetime.strptime(time_str, "%Y-%m-%d").timestamp()

    # 排序（确保间隔计算顺序正确）
    sorted_times = sorted(all_times)
    sorted_ts = np.array([str2ts(t) for t in sorted_times])

    # 2. 计算核心指标（保留所有间隔，包括0）
    intervals = sorted_ts[1:] - sorted_ts[:-1]  # 相邻间隔（秒），允许为0
    total_intervals = len(intervals)

    # （1）间隔变异系数（CV）：避免均值为0
    interval_mean = np.mean(intervals)
    interval_std = np.std(intervals)
    cv = interval_std / (interval_mean + 1e-8)  # 加1e-8避免除零

    # （2）最大最小间隔比（R）：避免最小间隔为0
    interval_min = np.min(intervals)
    interval_max = np.max(intervals)
    R = (interval_max + 1e-8) / (interval_min + 1e-8)

    # （3）零间隔占比（ZR）：反映事件集中爆发程度
    zero_interval_count = np.sum(intervals == 0)
    ZR = zero_interval_count / total_intervals if total_intervals > 0 else 0.0

    # 3. 三指标投票（避免单一指标失真）
    def get_level(cv, R, ZR):
        scores = 0
        # 每个指标按“均匀”得2分，“较均匀”得1分，“不均匀”得0分
        scores += 2 if cv < 0.6 else (1 if cv <= 1.2 else 0)
        scores += 2 if R < 8 else (1 if R <= 15 else 0)
        scores += 2 if ZR < 0.2 else (1 if ZR <= 0.4 else 0)

        if scores >= 5:
            print('均匀')
            return "均匀"
        elif scores >= 3:
            print('较均匀')
            return "较均匀"
        else:
            return "不均匀"

    uniformity_level = get_level(cv, R, ZR)
    return uniformity_level


import torch
from typing import List, Dict
import datetime
import re

def str2ts_tensor(time_str: str, global_T_min: torch.Tensor = None) -> torch.Tensor:
    """
    容错时间解析：
    - 标准格式（YYYY-MM-DD）：正常转换为时间戳；
    - 非标准格式（如'时间1'、'未知'、'2022/03/08'）：返回默认时间戳（全局最小值global_T_min）；
    """
    # 先尝试匹配标准格式（YYYY-MM-DD，允许年份4位、月日2位，如2022-03-08）
    time_pattern = r"^\d{4}-\d{2}-\d{2}$"
    if re.match(time_pattern, time_str):
        try:
            dt = datetime.datetime.strptime(time_str, "%Y-%m-%d")
            return torch.tensor(dt.timestamp(), dtype=torch.float32)
        except:
            # 极端情况：格式匹配但解析失败（如2022-02-30），返回默认值
            pass

    # 非标准时间：返回全局最小值（若未传入则默认0，后续会被全局范围修正）
    return global_T_min if global_T_min is not None else torch.tensor(0.0, dtype=torch.float32)


# 工具函数：时间列表→排序后的时间戳张量（容错版）
def get_sorted_ts(time_list: List[str], global_T_min: torch.Tensor = None) -> torch.Tensor:
    """
    输入K个时间点的字符串列表（支持非标准时间），返回排序后的时间戳张量（shape[K]，保留重复）
    参数：global_T_min - 全量时间的最小值（用于填充无效时间）
    # >>> 修改点 1：移除硬编码的长度校验，改为支持动态 K <<<
    """
    K = len(time_list)
    if K == 0:
        return torch.tensor([], dtype=torch.float32)

    # 转换为时间戳（容错处理）
    ts_list = [str2ts_tensor(time_str, global_T_min) for time_str in time_list]
    ts_tensor = torch.stack(ts_list)
    sorted_ts, _ = torch.sort(ts_tensor)  # 可微分排序，保留重复值
    return sorted_ts


def adaptive_time_reward_no_deduplicate(
        model_output: List[str],  # 模型输出：K个时间点的字符串列表
        standard_answer: List[str],  # 标准答案：K个时间点的字符串列表
        all_times: List[str],  # 全量原始时间列表
        uniformity_level: str  # 离线检测的均匀性等级
) -> float:
    """
    不去重的自适应时间分布奖励函数（GRPO可微分+容错时间解析），支持动态 K
    输出：0-1之间的float分数（越高越好）
    """
    K = len(model_output)
    if K < 2 or len(standard_answer) != K:
        # 数量不足或不匹配，无法计算间隔，返回最低奖励
        return 0.0
    # -------------------------- 1. 预处理：计算全量时间的全局参数（用于容错填充） --------------------------
    # 先处理全量时间，得到全局最小值、最大值、跨度（容错解析）
    sorted_all_times = sorted(all_times)
    # 计算全量时间的全局最小值（用于填充无效时间）
    global_T_min = None
    if sorted_all_times:
        # 找到全量时间中第一个有效时间作为全局最小值
        for time_str in sorted_all_times:
            if re.match(r"^\d{4}-\d{2}-\d{2}$", time_str):
                try:
                    dt = datetime.datetime.strptime(time_str, "%Y-%m-%d")
                    global_T_min = torch.tensor(dt.timestamp(), dtype=torch.float32)
                    break
                except:
                    pass
        # 若全量时间均为无效时间，默认全局最小值为0
        if global_T_min is None:
            global_T_min = torch.tensor(0.0, dtype=torch.float32)

    # 计算全局最大值和跨度
    if sorted_all_times and global_T_min is not None:
        # 找到全量时间中最后一个有效时间作为全局最大值
        global_T_max = global_T_min
        for time_str in reversed(sorted_all_times):
            if re.match(r"^\d{4}-\d{2}-\d{2}$", time_str):
                try:
                    dt = datetime.datetime.strptime(time_str, "%Y-%m-%d")
                    global_T_max = torch.tensor(dt.timestamp(), dtype=torch.float32)
                    break
                except:
                    pass
        global_span = global_T_max - global_T_min + 1e-8  # 避免除零
    else:
        global_T_max = torch.tensor(0.0, dtype=torch.float32)
        global_span = 1e-8  # 避免除零

    # -------------------------- 2. 转换模型输出和标准答案（容错解析） --------------------------
    model_ts = get_sorted_ts(model_output, global_T_min)  # 模型输出时间戳（shape[K]）
    std_ts = get_sorted_ts(standard_answer, global_T_min)  # 标准答案时间戳（shape[K]）
    # -------------------------- 3. 子项1：对齐标准答案时间分布（R_std_align，60%权重） --------------------------
    # （1）跨度对齐：模型输出跨度与标准答案跨度的一致性
    model_span = model_ts[-1] - model_ts[0]
    std_span = std_ts[-1] - std_ts[0] + 1e-8
    span_ratio = model_span / std_span
    span_alignment = 1 - torch.abs(span_ratio - 1.0)  # 归一化到[0,1]

    # （2）间隔模式对齐：相邻间隔的比例一致性（允许间隔为0）
    model_intervals = model_ts[1:] - model_ts[:-1]  # 模型间隔（shape[K-1]）
    std_intervals = std_ts[1:] - std_ts[:-1]

    # 可微分余弦相似度（衡量间隔比例一致性）
    def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        dot_product = torch.dot(a, b)
        norm_a = torch.norm(a)
        norm_b = torch.norm(b)
        return dot_product / ((norm_a * norm_b) + 1e-8)

    interval_alignment = cos_sim(model_intervals, std_intervals)
    interval_alignment = (interval_alignment + 1.0) / 2.0  # 映射到[0,1]

    # 合并对齐奖励（[0,1]范围）
    R_std_align = 0.5 * span_alignment + 0.5 * interval_alignment

    # -------------------------- 4. 子项2：贴合全量事件分布（R_global_adapt，40%权重） --------------------------
    # （1）覆盖度：模型输出覆盖全量时间范围的比例
    coverage = model_span / global_span  # 归一化到[0,1]
    coverage = torch.clamp(coverage, 0.0, 1.0)  # 避免覆盖度超过1

    if uniformity_level == "均匀":
        print('均匀')
        # 全量均匀→奖励模型间隔均匀（允许0间隔，离散度低）
        model_interval_mean = torch.mean(model_intervals) + 1e-8
        model_interval_std = torch.std(model_intervals)
        model_cv = model_interval_std / model_interval_mean  # 变异系数
        uniformity_score = 1 - torch.clamp(model_cv / 0.6, 0.0, 1.0)  # 阈值适配不重复场景
        R_global_adapt = 0.5 * coverage + 0.5 * uniformity_score  # [0,1]范围
    else:
        # 全量不均匀→奖励覆盖度+零间隔占比适配（贴合集中爆发特性）
        # 模型输出的零间隔占比
        model_zero_ratio = torch.sum(model_intervals == 0) / len(model_intervals)
        # 全量事件的零间隔占比（容错计算）
        sorted_all_ts = torch.tensor([str2ts_tensor(t, global_T_min) for t in sorted_all_times])
        all_intervals = sorted_all_ts[1:] - sorted_all_ts[:-1] if len(sorted_all_ts) > 1 else torch.tensor([0.0])
        global_zero_ratio = torch.sum(all_intervals == 0) / len(all_intervals) if len(all_intervals) > 0 else 0.0

        # 零间隔占比越接近全量，得分越高
        zero_adapt_score = 1 - torch.abs(model_zero_ratio - global_zero_ratio)
        R_global_adapt = 0.6 * coverage + 0.4 * zero_adapt_score  # [0,1]范围

    # -------------------------- 5. 合并总奖励（0-1范围） --------------------------
    total_time_reward = 0.6 * R_std_align + 0.4 * R_global_adapt
    # 转换为float并确保在0-1之间
    return float(torch.clamp(total_time_reward, 0.0, 1.0).item())

# 测试（含相同时间点的示例）
if __name__ == "__main__":
    # 含相同时间点的全量时间列表（如2022-03-08有多个事件）
    single_data_all_times = [
        "1996-01-01", "1999-01-01", "2011-03-22", "2022-03-08", "2022-03-08",
        "2022-03-08", "2022-03-09", "2022-03-12", "2023-06-21", "2023-06-21"
    ]
    uniformity_level = detect_uniformity_no_deduplicate(single_data_all_times)
    print(f"不去重的均匀性等级：{uniformity_level}")  # 输出：不均匀（因2022-03-08集中3个事件，零间隔占比高）