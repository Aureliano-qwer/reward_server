# app/rewards/legacy_rules.py
# ... (把你上面的所有 import 和 def 函数都贴在这里) ...

from app.rewards.base import BaseReward
import re
import json


# ================= 配置区域 (方便调整权重) =================
WEIGHT_VERDICT_CORRECT = 1.0   # 结论正确的得分
WEIGHT_ERROR_TP = 1.0          # 猜对一个错误类型的得分
WEIGHT_ERROR_FP = -1.0         # 猜错扣分
MIN_ERROR_SCORE = 0.0          # 错误分析最低分
SCORE_CONSOLATION = 0.5        # 【新增】安慰分：当GT无具体错误信息但结论为Failed时，只要预测了错误就给半分
# ==========================================================



# ==========================================================
# 严格模式打分器 (用于 Test/Validation 集)
# ==========================================================
class StrictRuleReward(BaseReward):
    """
    Test 模式专用：只关心结论 (Verdict) 是否正确。
    忽略具体的错误类型分析，计算速度极快。
    """
    async def compute(self, prompt: str, response: str, ground_truth: str, extra_info: dict = None) -> float:
        # 0. 基础检查
        if not response:
            return 0.0

        # 1. 提取思考后的内容 (防止 <think> 里的内容干扰)
        if "</think>" in response:
            answer_content = response.split("</think>")[-1]
        else:
            answer_content = response

        # 2. 直接调用现有的 verdict 判断逻辑
        # get_verdict_score 只会返回 1.0 (正确) 或 0.0 (错误)
        try:
            score = get_verdict_score(answer_content, ground_truth)
        except Exception as e:
            # 记录日志或静默失败
            print(f"[StrictRule] Verdict check failed: {e}")
            score = 0.0

        return float(score)


# ==========================================================
# 训练过程的规则打分器 (用于训练集)
# ==========================================================
class LegacyRuleReward(BaseReward):
    """
    适配器模式：将旧的 compute_score 逻辑封装为新框架的组件
    """
    async def compute(self, prompt: str, response: str, ground_truth: str, extra_info: dict = None) -> float:
        # 1. 构造参数
        # 注意：旧代码的 compute_score 需要 (data_source, solution_str, ground_truth, extra_info)
        # 我们这里暂时把 data_source 设为空字符串，因为你的核心逻辑似乎不太依赖它
        # 如果需要，可以从 kwargs 里传，或者修改 BaseReward 接口
        
        # 2. 调用旧函数
        # response 对应 solution_str
        score = compute_score(
            data_source="", 
            solution_str=response, 
            ground_truth=ground_truth, 
            extra_info=extra_info
        )
        
        return float(score)



def compute_score(data_source, solution_str, ground_truth, extra_info):
    """
    主打分函数
    Args:
        solution_str: 模型输出的完整文本
        ground_truth: 期望的结论字符串 (如 "Result: Passed")
        extra_info: 包含真实错误列表的字典 (key: "error_types")
    """
    # 0. 基础入参检查
    if not solution_str:
        return 0.0
        
    # 1. 提取模型思考后的正文
    # 防御性编程：防止没有 <think> 标签时报错
    if "</think>" in solution_str:
        answer_content = solution_str.split("</think>")[-1]
    else:
        answer_content = solution_str

    # 2. 计算结论分 (独立计算)
    try:
        verdict_score = get_verdict_score(answer_content, ground_truth)
    except Exception:
        verdict_score = 0.0

    # 3. 计算错误分析分 (独立计算)
    # 【新增】最外层 try-except，确保 error 分析崩了不影响结论分，也不会让程序挂掉
    analysis_score = MIN_ERROR_SCORE
    try:
        # 获取 Ground Truth 信息
        gt_errors = []
        if extra_info and isinstance(extra_info, dict):
            gt_errors = extra_info.get("error_types", [])

        # 判断 Ground Truth 的结论是否为 Failed
        # 注意：ground_truth 格式通常为 "\nResult: Failed"
        is_gt_failed = "failed" in ground_truth.lower() if ground_truth else False
        is_gt_labels_missing = (len(gt_errors) == 0)

        # ==================== 核心修改区域 ====================
        # 情况 A: 题目实际上是错的 (Failed)，但数据集中没有标注具体错误类型 (error_types为空)
        # 策略: 只要模型预测了任意错误，就给安慰分，不再进行具体的集合匹配
        if is_gt_failed and is_gt_labels_missing:
            # 提取模型预测的错误列表
            model_preds = extract_error_info(answer_content)
            if model_preds and len(model_preds) > 0:
                analysis_score = SCORE_CONSOLATION
            else:
                # 题目错了，模型也没预测出错误（可能模型认为是Passed），这部分得0分
                analysis_score = 0.0
        
        # 情况 B: 标准情况 (有标注 或者 题目是Passed)
        # 策略: 走原有的集合交并集匹配逻辑
        else:
            analysis_score = get_error_score(answer_content, gt_errors)
        # ====================================================

    except Exception as e:
        print(f"[Warning] Error analysis failed: {e}") 
        analysis_score = MIN_ERROR_SCORE 

    # 4. 总分汇总
    total_score = verdict_score + analysis_score
    
    return total_score


def get_verdict_score(answer_content, ground_truth):
    """
    子函数：判断结论是否正确
    逻辑：提取 Result: Passed/Failed 进行比对，对上得 1 分，否则 0 分
    """
    if not ground_truth:
        return 0.0

    # 提取模型输出的 verdict
    pattern = r"Result:\s*(Passed|Failed)"
    matches = re.findall(pattern, answer_content, re.IGNORECASE)
    
    if not matches:
        return 0.0
    
    # 取最后一个匹配到的结论，防止模型中间由纠结变为确定
    model_verdict = matches[-1].strip().lower()
    
    # 清洗 Ground Truth
    gt_clean = ground_truth.strip().replace("Result:", "").strip().lower()
    
    if model_verdict == gt_clean:
        return WEIGHT_VERDICT_CORRECT
    
    return 0.0


def normalize_code_string(s):
    """
    【核心优化】: 彻底移除所有空白字符，用于代码比对
    input:  "return 1 / n"
    output: "return1/n"
    """
    if not s:
        return "None"
    return "".join(s.split())


def normalize_error(error_item):
    """
    标准化错误条目，用于集合运算
    
    return sample:
    ("Runtime Error", "<错误的行内容>")
    ("Wrong Answer", "None"<字符串None>)
    """
    try:
        e_type = error_item.get("type", "").strip()
        e_stmt = error_item.get("statement")
        
        # 只有特定的错误类型才需要比较 statement
        if e_type in ["Runtime Error", "Compilation Error"]:
            if e_stmt is not None:
                # 使用去空格后的字符串作为指纹
                e_stmt_norm = normalize_code_string(str(e_stmt))
            else:
                e_stmt_norm = "None"
        else:
            # 对于 WA, TLE, MLE，不比较 statement，只要类型对就行
            e_stmt_norm = "IgnoreStatement"
            
        return (e_type, e_stmt_norm)
        
    except (AttributeError, TypeError, Exception):
        return ("INVALID_FORMAT", "None")


def get_error_score(answer_content, gt_error_list):
    """
    子函数：判断错误类型猜测的准确性
    逻辑：
        - 猜对 (交集) 每个 +1
        - 猜错 (模型有但GT没有) 每个 -1
        - 最低 0 分
    """
    # 1. 提取并标准化模型的预测
    raw_pred_list = extract_error_info(answer_content)
    
    if not isinstance(raw_pred_list, list):
        return MIN_ERROR_SCORE

    # 1. 转换模型预测
    pred_set = set()
    for item in raw_pred_list:
        norm = normalize_error(item)
        if norm[0] != "INVALID_FORMAT":
            pred_set.add(norm)
    
    # 2. 转换 Ground Truth
    gt_set = set()
    for item in gt_error_list:
        norm = normalize_error(item)
        gt_set.add(norm)
    
    # 3. 计算交集和差集
    # 过滤掉那些格式错误的预测（"INVALID_FORMAT"），避免它们被计入 FP 扣分
    # (或者你也可以选择保留它们作为 FP 进行惩罚，这里我选择保留作为惩罚)
    
    tp_count = len(pred_set.intersection(gt_set))
    
    # 你的逻辑是 FP 扣分。
    # 这里有一个策略选择：
    # 如果 GT 是空的（代码Passed），模型预测了错误，这肯定是 FP。
    # 如果 GT 有错误，模型预测了多余的错误，也是 FP。
    fp_count = len(pred_set.difference(gt_set))
    
    score = (tp_count * WEIGHT_ERROR_TP) + (fp_count * WEIGHT_ERROR_FP)
    
    return max(score, MIN_ERROR_SCORE)


def extract_error_info(response_text: str) -> list:
    """
    提取 JSON 列表
    """
    try:
        if not response_text:
            return []

        # 缩小搜索范围
        section_match = re.search(r"(Section 2|Error Prediction).*?(Section 3|Verdict|$)", response_text, re.DOTALL | re.IGNORECASE)
        search_text = section_match.group(0) if section_match else response_text

        # 尝试提取 JSON 数组
        # 优化正则：允许由 ```json [...] ``` 包裹，也允许直接写 [...]
        # 并且兼容多行
        candidates = re.findall(r"\[\s*{.*?}\s*\]", search_text, re.DOTALL)
        
        for candidate in candidates:
            try:
                # 尝试修复常见的 JSON 格式错误（如尾部逗号），如果是严谨数据可跳过这一步
                # 这里直接用标准 loads
                parsed_data = json.loads(candidate)
                if isinstance(parsed_data, list):
                    return parsed_data
            except json.JSONDecodeError:
                continue

    except Exception:
        print("解析失败，返回空列表，得分为 0")
        pass

    return []