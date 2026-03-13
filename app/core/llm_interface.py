"""统一 LLM 引擎抽象接口，供 vLLM 与 SGLang 适配器实现。"""
from abc import ABC, abstractmethod
from typing import AsyncIterator


class BaseLLMEngine(ABC):
    """LLM 推理引擎抽象基类。"""

    @abstractmethod
    async def generate(
        self, prompt: str, sampling_params: dict, request_id: str
    ) -> AsyncIterator[dict]:
        """
        异步生成文本。

        Args:
            prompt: 输入提示
            sampling_params: 统一采样参数字典，包含 temperature, max_new_tokens, stop 等
            request_id: 请求 ID

        Yields:
            每次 yield 一个 dict: {"finished": bool, "text": str, "finish_reason": str | None}
        """
        pass
