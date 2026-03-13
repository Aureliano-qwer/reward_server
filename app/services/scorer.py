import asyncio
import logging
import time  # <--- [新增] 引入时间模块
from app.api.schemas import RewardRequest, RewardResponse
from app.rewards.rule_rewards import FormatReward, AccuracyReward
from app.rewards.llm_judge import QwenJudgeReward, GptOssJudgeReward
from app.rewards.legacy_rules import LegacyRuleReward, StrictRuleReward

logger = logging.getLogger(__name__)

class ScorerService:
    # [新增] 类级别的并发计数器，所有实例共享
    _active_requests = 0

    def __init__(self):
        # self.fmt_reward = FormatReward()
        self.acc_reward = AccuracyReward()
        self.llm_reward = None
        self.legacy_rule = LegacyRuleReward()
        self.strict_rule = StrictRuleReward()

    def _get_llm_reward(self):
        if self.llm_reward is None:
            # 懒加载 LLM Judge，避免模块导入阶段触发多进程初始化
            self.llm_reward = GptOssJudgeReward()  # QwenJudgeReward()
        return self.llm_reward

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
            
            # 获取数据来源，用于判断是否为测试集
            data_source = req.data_source or "unknown"
            
            # === [分流逻辑] ===
            # 如果是测试集 (data_source 包含 'test')，只跑 Rule Based (Strict)
            # === [分流逻辑] ===
            if "test" in data_source:
                logger.info(f"ℹ️ [Req {req_id}] Detected TEST dataset ('{data_source}'). Running Strict Rule AND LLM Judge for tracking.")
                
                # === [新增] 旁路执行 LLM Judge (仅用于记录打点，不参与最终计分) ===
                t_llm_start = time.time()
                llm_reward = self._get_llm_reward()
                score, reasoning = await llm_reward.compute(req.prompt_str, req.response_str, ground_truth_str, extra_info)
                t_llm_cost = time.time() - t_llm_start
                
                logger.info(f"🤖 [Req {req_id}] [TEST] LLM Finish | Cost: {t_llm_cost:.4f}s | Score: {score}")
                
                # 记录 LLM 表现与输出长度 (以字符数为单位)
                details["llm_judge"] = score
                details["llm_reasoning"] = reasoning
                details["llm_response_length"] = len(reasoning)  # <--- [新增] 长度统计
                
                # === [原有的 Strict Rule 逻辑] ===
                t_rule_start = time.time()
                extra_info_dict = extra_info if isinstance(extra_info, dict) else {}
                
                # 1. 创建 Task (Coroutine)
                rule_score_task = self.strict_rule.compute(
                    prompt=req.prompt_str,
                    response=req.response_str,
                    ground_truth=ground_truth_str,
                    extra_info=extra_info_dict
                )

                # 2. 等待结果
                rule_score = await rule_score_task
                
                # Test 模式下，Final Score 就是 Rule Score (硬指标)
                final_score = rule_score
                reason = "Strict Verdict (Test)"
                
                details["rule_score"] = rule_score
                details["final_score"] = final_score
                details["mode"] = "strict_test"
                
                logger.info(f"📏 [Req {req_id}] Rule Finish | Cost: {time.time() - t_rule_start:.4f}s | Final Score: {final_score}")

            # === [训练逻辑] ===
            # 如果是训练集，跑 LLM Judge + Rule (Reward Shaping)
            else:
                # logger.info(f"[{req_id}] 正在调用 LLM Judge...")
                
                # [DEBUG] 记录 vLLM 推理耗时
                t_llm_start = time.time()
                
                # 关键点：这里是 await，让出控制权。
                # 如果 vLLM 是并行的，这里应该会有多个请求同时处于 await 状态。
                llm_reward = self._get_llm_reward()
                score, reasoning = await llm_reward.compute(req.prompt_str, req.response_str, ground_truth_str, extra_info)


                # TODO：加个负分机制
                # score = score - 0.5
                
                t_llm_cost = time.time() - t_llm_start
                
                # [DEBUG] 打印单次推理耗时
                logger.info(f"🤖 [Req {req_id}] LLM Finish | Cost: {t_llm_cost:.4f}s | Score: {score}")
                # [日志] 可以选择把 reasoning 打印出来（如果不太长的话）
                logger.info(f"🤖 [Req {req_id}] Reason: {reasoning}...")

                details["llm_judge"] = score
                details["llm_reasoning"] = reasoning
                details["llm_response_length"] = len(reasoning)  # <--- [新增] 保持训练集也记录长度
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

                # 4. 计算总分 (混合分 - 加权求和)
                weight_rule = 0.4
                weight_llm = 0.6
                # final_score = (rule_score * weight_rule) + (score * weight_llm)
                final_score = score
                reason = "LLM + Rule"  # <--- [微调] 名字改成加号，符合现在逻辑
                
                details["rule_score"] = rule_score
                details["final_score"] = final_score
                details["mode"] = "train_shaping"
                
                logger.info(f"📏 [Req {req_id}] Rule Finish | Cost: {time.time() - t_rule_start:.4f}s | Final Score: {final_score}")

            # 返回扁平结构
            return RewardResponse(
                score=final_score,
                details=details,
                reason=reason
            )

        except Exception as e:
            logger.error(f"❌ [Req {req_id}] ERROR: {str(e)}")
            # 打印完整的 traceback 以便调试
            import traceback
            logger.error(traceback.format_exc())
            return RewardResponse(score=0.0, details={}, reason="error")
            
        finally:
            # 无论成功失败，必须减少计数器
            ScorerService._active_requests -= 1
            total_cost = time.time() - start_time
            
            # [DEBUG] 打印请求结束时的总耗时
            logger.info(f"🔴 [Req {req_id}] END | Total Cost: {total_cost:.4f}s | Active Remaining: {ScorerService._active_requests}")