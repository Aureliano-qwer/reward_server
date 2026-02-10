import asyncio
import logging
import time  # <--- [新增] 引入时间模块
from app.api.schemas import RewardRequest, RewardResponse
from app.rewards.rule_rewards import FormatReward, AccuracyReward
from app.rewards.llm_judge import QwenJudgeReward, GptOssJudgeReward
from app.rewards.legacy_rules import LegacyRuleReward

logger = logging.getLogger(__name__)

class ScorerService:
    # [新增] 类级别的并发计数器，所有实例共享
    _active_requests = 0

    def __init__(self):
        # self.fmt_reward = FormatReward()
        self.acc_reward = AccuracyReward()
        self.llm_reward = GptOssJudgeReward() # QwenJudgeReward()
        self.legacy_rule = LegacyRuleReward()

    async def calculate(self, req: RewardRequest) -> RewardResponse:
        # 0. 计时与计数开始
        start_time = time.time()
        ScorerService._active_requests += 1
        
        # 提取 ID 方便追踪日志 (假设 extra_info 里有 test_id)
        req_id = "unknown"
        if isinstance(req.extra_info, dict):
            req_id = req.extra_info.get("test_id", "unknown")
        
        # [DEBUG] 打印请求进入时的并发数
        # 如果这个数字一直很小（例如 1-5），说明请求是被串行卡住发进来的
        # 如果瞬间飙升到 50-100，说明 FastAPI 层面的并发是正常的
        logger.info(f"🟢 [Req {req_id}] START | Active Concurrent: {ScorerService._active_requests}")

        try:
            # Client 可能会发 list 类型的 ground_truth，做个兼容
            ground_truth_str = req.ground_truth
            extra_info = req.extra_info
            if isinstance(req.ground_truth, list) and len(req.ground_truth) > 0:
                ground_truth_str = req.ground_truth[0]
            ground_truth_str = str(ground_truth_str)

            # ------------------------------------------------------------------
            # [调试代码] 打印接收到的 data_source
            # ------------------------------------------------------------------
            # print(f"DEBUG: data_source received: '{req.data_source}'")

            details = {}
            final_score = 0.0
            reason = "rule_based"
                
            # logger.info(f"[{req_id}] 正在调用 LLM Judge...")
            
            # [DEBUG] 记录 vLLM 推理耗时
            t_llm_start = time.time()
            
            # 关键点：这里是 await，让出控制权。
            # 如果 vLLM 是并行的，这里应该会有多个请求同时处于 await 状态。
            score = await self.llm_reward.compute(req.prompt_str, req.response_str, ground_truth_str, extra_info)
            
            t_llm_cost = time.time() - t_llm_start
            
            # [DEBUG] 打印单次推理耗时
            logger.info(f"🤖 [Req {req_id}] LLM Finish | Cost: {t_llm_cost:.4f}s | Score: {score}")

            details["llm_judge"] = score
            # details["format"] = await self.fmt_reward.compute("", req.response_str, "")
            
            t_rule_start = time.time()
            extra_info_dict = extra_info if isinstance(extra_info, dict) else {}
            
            # 1. 创建 Task (Coroutine)
            rule_score_task = self.legacy_rule.compute(
                prompt=req.prompt_str,
                response=req.response_str,
                ground_truth=ground_truth_str,
                extra_info=extra_info_dict
            )

            # 2. 等待结果 (只接收一个返回值！)
            rule_score = await rule_score_task
            
            # 3. 如果你需要 fmt_score 用于兼容旧逻辑，可以手动设为 1.0 或 0.0
            # fmt_score = 1.0 

            # 4. 计算总分
            final_score = rule_score * score
            reason = "LLM * Rule"
            
            details["rule_score"] = rule_score
            details["final_score"] = final_score
            
            logger.info(f"📏 [Req {req_id}] Rule Finish | Cost: {time.time() - t_rule_start:.4f}s | Final Score: {final_score}")

            # 返回扁平结构
            return RewardResponse(
                score=final_score,
                details=details,
                reason=reason
            )

        except Exception as e:
            logger.error(f"❌ [Req {req_id}] ERROR: {str(e)}")
            return RewardResponse(score=0.0, details={}, reason="error")
            
        finally:
            # 无论成功失败，必须减少计数器
            ScorerService._active_requests -= 1
            total_cost = time.time() - start_time
            
            # [DEBUG] 打印请求结束时的总耗时
            logger.info(f"🔴 [Req {req_id}] END | Total Cost: {total_cost:.4f}s | Active Remaining: {ScorerService._active_requests}")