from typing import Any, Dict, List, Union, Optional
from pydantic import BaseModel

class RewardRequest(BaseModel):
    # === 严格对应 Client 端 CallScoreFuncInput 发送的字段 ===
    data_source: str = ""
    solution_str: str = ""           # Client 发送了 prompt + response 的拼接
    ground_truth: Union[str, List[str], Any] = ""
    extra_info: Optional[Union[Dict, str, Any]] = None
    prompt_str: str = ""
    response_str: str = ""
    
    # 虽然 Client 没发 valid_response_length 给 API，但保留以便扩展
    valid_response_length: Optional[int] = None

class RewardResponse(BaseModel):
    # === 对应 Client 端 async_call_online_reward_model 的解析逻辑 ===
    # Client: final_score = res.get("score")
    score: float       
    
    # 其他信息会作为 res 返回给 Client 的 detail
    details: Dict[str, Any] = {}
    reason: str = ""