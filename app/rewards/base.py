# app/rewards/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseReward(ABC):
    @abstractmethod
    async def compute(self, prompt: str, response: str, ground_truth: str, extra_info: Optional[Dict] = None) -> float:
        """
        所有奖励类必须实现这个方法。
        
        """
        pass