import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 服务配置
    PORT: int = 23334
    HOST: str = "0.0.0.0"
    LOG_DIR: str = "data/reward_log"
    
    # 模型配置
    MODEL_PATH: str = "openai/gpt-oss-120b"
    GPU_MEMORY_UTILIZATION: float = 0.9
    TENSOR_PARALLEL_SIZE: int = 8
    MAX_MODEL_LEN: int = 32768

    class Config:
        env_file = ".env"

settings = Settings()

# 确保日志目录存在
os.makedirs(settings.LOG_DIR, exist_ok=True)