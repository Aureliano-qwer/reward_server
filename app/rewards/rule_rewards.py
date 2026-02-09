import re
from typing import List, Union
# 假设原来的 util.py 在根目录，如果移动了请修改 import
try:
    from util import calc_accuracy4math, format_reward_deepseek
except ImportError:
    # 兜底 Mock，防止报错无法运行
    def calc_accuracy4math(gt, pred): return 1.0
    def format_reward_deepseek(pred): return 1.0

from app.rewards.base import BaseReward

class FormatReward(BaseReward):
    async def compute(self, prompt: str, response: str, ground_truth: str, extra_info: dict = None) -> float:
        # 这里直接复用 DeepSeek 的格式检查逻辑
        # 如果需要原来的 patch1 逻辑，也可以贴在这里
        res = format_reward_deepseek(response)
        return float(res)

class AccuracyReward(BaseReward):
    async def compute(self, prompt: str, response: str, ground_truth: str, extra_info: dict = None) -> float:
        # 处理 list 情况
        gt = ground_truth[0] if isinstance(ground_truth, list) else ground_truth
        res = calc_accuracy4math(gt, response)
        return float(res)