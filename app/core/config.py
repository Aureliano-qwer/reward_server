import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 服务配置
    PORT: int = 23334
    HOST: str = "0.0.0.0"
    # 显式变量，优先于 HOST/PORT，避免被系统 HOST 环境变量污染
    APP_HOST: str | None = None
    APP_PORT: int | None = None
    LOG_DIR: str = "data/reward_log"
    
    # 模型配置
    MODEL_PATH: str = "openai/gpt-oss-120b"
    GPU_MEMORY_UTILIZATION: float = 0.9
    TENSOR_PARALLEL_SIZE: int = 8
    MAX_MODEL_LEN: int = 32768

    # 推理后端: "vllm" | "sglang"，默认 vllm 保持向后兼容
    INFERENCE_BACKEND: str = "vllm"
    # SGLang 专用: GPU 显存占比 (对应 vLLM 的 gpu_memory_utilization)
    MEM_FRACTION_STATIC: float = 0.83

    class Config:
        env_file = ".env"

settings = Settings()

# 确保日志目录存在
os.makedirs(settings.LOG_DIR, exist_ok=True)