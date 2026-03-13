import logging
from logging.handlers import TimedRotatingFileHandler
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.engine import LLMEngineManager
from app.api.routes import router

# ================= 日志配置 =================
logger = logging.getLogger("app")
file_handler = TimedRotatingFileHandler(
    f"{settings.LOG_DIR}/reward.log",
    when="midnight", interval=1, backupCount=5
)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# 根 logger 配置
logging.basicConfig(level=logging.INFO)
logging.getLogger("app").addHandler(file_handler)
# 也可以把 vllm 的日志重定向过来，或者屏蔽

# ================= 生命周期 =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时：加载模型
    logging.info("Server Starting... Initializing Engine...")
    LLMEngineManager.get_instance() 
    yield
    # 关闭时
    logging.info("Server Shutting down...")

# ================= APP 初始化 =================
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    host = settings.APP_HOST if settings.APP_HOST is not None else settings.HOST
    port = settings.APP_PORT if settings.APP_PORT is not None else settings.PORT
    logger.info("Binding to %s:%s", host, port)
    uvicorn.run(app, host=host, port=port)