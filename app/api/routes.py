import json
import logging
from fastapi import APIRouter, Request
from app.api.schemas import RewardRequest, RewardResponse
from app.services.scorer import ScorerService

router = APIRouter()
logger = logging.getLogger(__name__)

scorer_service = ScorerService()

# 建议保留 /get_reward2 或改成更通用的 /compute_score
# 只要 Client 端的 url 配置对就行
@router.post("/compute_score", response_model=RewardResponse)
async def get_reward_endpoint(request: RewardRequest):
    """
    对应 Verl Client 的调用
    """
    # 1. 计算
    result = await scorer_service.calculate(request)
    
    # 2. 记录详细日志 (方便后面训练分析)
    # 将 Pydantic 对象转为 dict
    log_payload = {
        "input": request.model_dump(),
        "output": result.model_dump(),
        "timestamp": 1234567890 # 这里可以加个时间戳
    }
    logger.info(json.dumps(log_payload, ensure_ascii=False))
    
    # 3. 返回给 Client
    # FastAPI 会自动把 result (RewardResponse对象) 序列化为 JSON
    # JSON 结构: {"score": 1.0, "details": {...}, "reason": "..."}
    # 这完全符合 Client 代码: final_score = res.get("score")
    return result

@router.get("/health")
async def health():
    return {"status": "ok"}