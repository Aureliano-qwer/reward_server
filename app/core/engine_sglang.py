"""SGLang 推理引擎适配器。"""
import logging

from app.core.config import settings
from app.core.llm_interface import BaseLLMEngine

logger = logging.getLogger(__name__)


def _to_sglang_sampling_params(params: dict) -> dict:
    """将统一采样参数字典转为 SGLang 格式。"""
    max_tokens = params.get("max_new_tokens", params.get("max_tokens", 8192))
    out = {
        "temperature": params.get("temperature", 0.2),
        "max_new_tokens": max_tokens,
    }
    if params.get("stop"):
        out["stop"] = params["stop"]
    return out


class SGLangEngineAdapter(BaseLLMEngine):
    """SGLang 引擎适配器，实现 BaseLLMEngine 接口。"""

    def __init__(self) -> None:
        try:
            import sglang as sgl
        except ImportError as e:
            raise ImportError(
                "INFERENCE_BACKEND=sglang 需要安装 sglang。请运行: pip install 'sglang[cu121]'"
            ) from e

        logger.info(f"正在初始化 SGLang 引擎: {settings.MODEL_PATH} ...")
        self._engine = sgl.Engine(
            model_path=settings.MODEL_PATH,
            tp_size=settings.TENSOR_PARALLEL_SIZE,
            mem_fraction_static=settings.MEM_FRACTION_STATIC,
            trust_remote_code=True,
            context_length=settings.MAX_MODEL_LEN,
        )
        logger.info("SGLang 引擎初始化完成。")

    async def generate(self, prompt: str, sampling_params: dict, request_id: str):
        sgl_params = _to_sglang_sampling_params(sampling_params)
        outputs = await self._engine.async_generate([prompt], sgl_params)
        text = outputs[0]["text"] if outputs else ""
        # SGLang 非 streaming 模式一次性返回，封装为单次 yield 以符合接口
        yield {
            "finished": True,
            "text": text,
            "finish_reason": None,
        }
