"""vLLM 推理引擎适配器。"""
import logging
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams as VLLMSamplingParams

from app.core.config import settings
from app.core.llm_interface import BaseLLMEngine

logger = logging.getLogger(__name__)


def _to_vllm_sampling_params(params: dict) -> VLLMSamplingParams:
    """将统一采样参数字典转为 vLLM SamplingParams。"""
    max_tokens = params.get("max_new_tokens", params.get("max_tokens", 8192))
    return VLLMSamplingParams(
        temperature=params.get("temperature", 0.2),
        max_tokens=max_tokens,
        stop=params.get("stop"),
    )


class VLLMEngineAdapter(BaseLLMEngine):
    """vLLM 引擎适配器，实现 BaseLLMEngine 接口。"""

    def __init__(self) -> None:
        logger.info(f"正在初始化 vLLM 引擎: {settings.MODEL_PATH} ...")
        engine_args = AsyncEngineArgs(
            model=settings.MODEL_PATH,
            tensor_parallel_size=settings.TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=settings.GPU_MEMORY_UTILIZATION,
            trust_remote_code=True,
            max_model_len=settings.MAX_MODEL_LEN,
        )
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info("vLLM 引擎初始化完成。")

    async def generate(self, prompt: str, sampling_params: dict, request_id: str):
        from vllm.utils import random_uuid

        rid = request_id or random_uuid()
        vllm_params = _to_vllm_sampling_params(sampling_params)
        async for request_output in self._engine.generate(prompt, vllm_params, rid):
            if request_output.outputs:
                out = request_output.outputs[0]
                yield {
                    "finished": request_output.finished,
                    "text": out.text,
                    "finish_reason": getattr(out, "finish_reason", None),
                }
            else:
                yield {
                    "finished": request_output.finished,
                    "text": "",
                    "finish_reason": None,
                }
