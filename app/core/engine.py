import logging
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from app.core.config import settings

logger = logging.getLogger(__name__)

class LLMEngineManager:
    _instance = None

    @classmethod
    def get_instance(cls) -> AsyncLLMEngine:
        """
        获取全局唯一的 vLLM 引擎实例。
        如果是第一次调用，则初始化引擎。
        """
        if cls._instance is None:
            logger.info(f"正在初始化 vLLM 引擎: {settings.MODEL_PATH} ...")
            engine_args = AsyncEngineArgs(
                model=settings.MODEL_PATH,
                tensor_parallel_size=settings.TENSOR_PARALLEL_SIZE,
                gpu_memory_utilization=settings.GPU_MEMORY_UTILIZATION,
                trust_remote_code=True,
                # disable_log_requests=True,
                max_model_len=settings.MAX_MODEL_LEN
            )
            cls._instance = AsyncLLMEngine.from_engine_args(engine_args)
            logger.info("vLLM 引擎初始化完成。")
        return cls._instance