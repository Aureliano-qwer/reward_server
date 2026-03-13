import logging
from typing import Union

from app.core.config import settings
from app.core.llm_interface import BaseLLMEngine

logger = logging.getLogger(__name__)


def _create_engine() -> BaseLLMEngine:
    backend = (settings.INFERENCE_BACKEND or "vllm").strip().lower()
    if backend == "sglang":
        from app.core.engine_sglang import SGLangEngineAdapter
        return SGLangEngineAdapter()
    elif backend == "vllm":
        from app.core.engine_vllm import VLLMEngineAdapter
        return VLLMEngineAdapter()
    else:
        raise ValueError(
            f"不支持的 INFERENCE_BACKEND: {settings.INFERENCE_BACKEND}。"
            "可选: vllm, sglang"
        )


class LLMEngineManager:
    _instance: Union[BaseLLMEngine, None] = None

    @classmethod
    def get_instance(cls) -> BaseLLMEngine:
        """
        获取全局唯一的 LLM 引擎实例（vLLM 或 SGLang，由 INFERENCE_BACKEND 决定）。
        如果是第一次调用，则初始化引擎。
        """
        if cls._instance is None:
            cls._instance = _create_engine()
        return cls._instance