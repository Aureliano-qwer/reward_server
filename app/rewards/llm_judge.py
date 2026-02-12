from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from app.core.engine import LLMEngineManager
from app.rewards.base import BaseReward

class QwenJudgeReward(BaseReward):
    def __init__(self):
        # 懒加载：初始化类时获取引擎实例
        self.engine = LLMEngineManager.get_instance()

    async def compute(self, prompt: str, response: str, ground_truth: str, extra_info: dict = None) -> float:
        system_prompt = """You are a senior code auditing expert. Your task is to evaluate the analysis provided by a "Junior Code Analyst" (the Student Model) regarding a specific programming problem and its solution.

You need to compare the [Ground Truth] with the [Student's Analysis] to determine if the student's judgment is factually correct and logically sound.

### Evaluation Criteria:
1. **Fact-Checking**: Does the student's conclusion (Correct/Incorrect) match the Ground Truth?
2. **Hallucination Detection**:
   - If the Ground Truth says the code has a bug, does the student identify the *same* bug? (Reject if the student points out a non-existent or wrong error).
   - If the Ground Truth says the code is correct, does the student hallucinate a bug? (Reject if the student nitpicks correct code).

### Output Format:
First, provide a brief reasoning in a `<reasoning>` tag, analyzing whether the student's finding aligns with the Ground Truth.
Finally, output the score in a `<score>` tag:
- **[[1]]**: The student's analysis is completely correct (both the conclusion and the specific reason match the Ground Truth).
- **[[0]]**: The student's analysis is incorrect, hallucinatory, or identifies the wrong root cause.
"""

        user_prompt = f"""### [Problem & Code]
{prompt}

### [Ground Truth]
{ground_truth}

### [extra_info]
{extra_info}

### [Student's Analysis (To be Evaluated)]
{response}
"""
        judge_prompt = build_prompt(system_prompt, user_prompt, enable_think=True, name="qwen")
        # 确定性采样
        sampling_params = SamplingParams(
            temperature=0.0, 
            max_tokens=4096, 
            stop=["<|im_end|>"])
        request_id = random_uuid()

        # 调用 vLLM 引擎
        results_generator = self.engine.generate(judge_prompt, sampling_params, request_id)
        print("results_generator: ", results_generator)

        final_output = ""
        async for request_output in results_generator:
            if request_output.finished:
                final_output = request_output.outputs[0].text

        # 解析结果
        if "[[1]]" in final_output:
            return 1.0
        elif "[[0]]" in final_output:
            return 0.0

        print(f"DEBUG Unmatched Output: {final_output[:100]}...")
        return 0.0 # 兜底




def build_prompt(system_prompt, user_prompt, enable_think, name) -> str:
    if "qwen" in name or 'ds' in name:
        if enable_think:
            return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|><|im_start|>assistant\n<think>\n"
        else:
            return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"
    elif 'ds' in name:
        if enable_think:
            return f'<｜begin▁of▁sentence｜><｜User｜>{user_prompt}<｜Assistant｜><think>' # 忽略system_prompt
        else:
            return f'<｜begin▁of▁sentence｜><｜User｜>{user_prompt}<｜Assistant｜><think>\n\n</think>' # 忽略system_prompt
    elif "gpt" in name:
        user_message_formatted = f"<|start|>user<|message|>{user_prompt}<|end|>"
        if enable_think:
            return f'''<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-10-31
Reasoning: medium
# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>user<|message|>{system_prompt}\n\n{user_prompt}<|end|><|start|>assistant<|channel|>analysis<|message|>'''
        else:
            return f"{user_message_formatted}<|start|>assistant<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>" # 忽略system_prompt


class GptOssJudgeReward(BaseReward):
    def __init__(self):
        self.engine = LLMEngineManager.get_instance()

    async def compute(self, prompt: str, response: str, ground_truth: str, extra_info: dict = None) -> float:
        # 1. 独立的 System Prompt
        # 这里的指令依然要清晰，但不需要像之前那样极端精简
        # 你的 build_prompt 模板会自动把它放在 <|start|>system... 里
        system_prompt_ver1 = """You are a senior code auditing expert. Your task is to evaluate the analysis provided by a "Junior Code Analyst" (the Student Model) regarding a specific programming problem and its solution.

You need to compare the [Ground Truth] with the [Student's Analysis] to determine if the student's judgment is factually correct and logically sound.

### Evaluation Criteria:
1. **Fact-Checking**: Does the student's conclusion (Correct/Incorrect) match the Ground Truth?
2. **Hallucination Detection**:
   - If the Ground Truth says the code has a bug, does the student identify the *same* bug? (Reject if the student points out a non-existent or wrong error).
   - If the Ground Truth says the code is correct, does the student hallucinate a bug? (Reject if the student nitpicks correct code).

### Output Format:
First, provide a brief reasoning in a `<reasoning>` tag, analyzing whether the student's finding aligns with the Ground Truth.
Finally, output the score in a `<score>` tag:
- **[[1]]**: The student's analysis is completely correct (both the conclusion and the specific reason match the Ground Truth).
- **[[0]]**: The student's analysis is incorrect, hallucinatory, or identifies the wrong root cause.
"""
        system_prompt = """You are a Senior Competitive Programming Code Audit Expert with over 10 years of experience. Your task is to audit an analysis report generated by a "Junior Code Analyst" (Student Model) regarding a specific programming problem and its solution.

You have access to:
1. The <<<problem>>> description.
2. The user's submitted <<<code>>>.
3. The [Ground Truth] (whether the code is actually correct or incorrect, and why).
4. The [Student's Analysis] (the text you are auditing).

### Core Workflow (Mental Sandbox)
Before evaluating the student, you must perform these internal steps:
1. **Independent Pre-Analysis**: Read the problem and code. Mentally trace the logic. Identify actual complexity, potential syntax traps (e.g., Python indentation outside loops), or type errors (e.g., hashing mutable lists).
2. **Reality Check (Anti-Hallucination)**: verify if the variables, functions, or logic described by the student *actually exist* in the code.
   - Did the student claim "Division by Zero" in code that has no division?
   - Did they cite a variable name that doesn't exist?
   - Did they confuse the logic of Problem A with Problem B?
3. **Trace Verification**: If the student provides a counter-example (e.g., "Input [0,1] returns YES"), you must mentally simulate the code with that input. Do not trust the student's assertion blindly.

### Strict Evaluation Criteria
You must penalize **Hallucinations** severely. A "Correct" verdict based on "Hallucinated" reasoning is a **FAILURE**.

**Score [[1]] (Pass) conditions:**
- The student's conclusion (Passed/Failed) matches the [Ground Truth].
- **CRITICAL**: The specific reasoning matches the root cause.
    - If GT says "Time Limit Exceeded due to O(N^2)", the student must identify the complexity issue, not a fake syntax error.
    - If GT says "Logic Error in Loop", the student must point to that loop.

**Score [[0]] (Fail) conditions:**
- **Wrong Verdict**: Student says "Correct" when code is buggy, or vice versa.
- **Hallucinated Variables/Syntax**: Student points out errors in lines or variables that do not exist.
- **Generic/Lazy Analysis**: Student gives a template error (e.g., "Edge case unhandled") without specifying *which* edge case or *why* the logic fails.
- **Right Conclusion, Wrong Reason**: The code is indeed incorrect (matches GT), but the student identified a non-existent bug (e.g., claiming a syntax error where none exists) to reach that conclusion. This is a logical hallucination.

### Output Format
1. **Reasoning**: Inside `<reasoning>` tags, provide a rigorous audit.
   - First, state if the student's verdict matches the Ground Truth.
   - Second, perform the **Hallucination Check**: specificy if the student's cited variables/logic are real.
   - Third, verify the **Causal Link**: Did the student find the *actual* bug defined in the GT, or did they make up a fake bug?
2. **Score**: Inside `<score>` tags, output only [[1]] or [[0]].

Example Output Structure:
<reasoning>
The student correctly identified that the code fails, agreeing with the Ground Truth. However, the student claims the error is a "Division by Zero" on line 12. Upon auditing line 12, it is a simple addition operation. The actual error (per Ground Truth) is an index out of bounds error on line 5. Because the student hallucinated a non-existent arithmetic error to explain the failure, this analysis is untrustworthy.
</reasoning>
<score>
[[0]]
</score>
"""

        user_prompt = f"""### [Problem & Code]
{prompt}

### [Ground Truth]
{ground_truth}

### [extra_info]
{extra_info}

### [Student's Analysis (To be Evaluated)]
{response}
"""

        
        # 3. 构建 Prompt (启用思考模式)
        # enable_think=True -> 激活模板里的 "Reasoning: low" 和 System Prompt
        final_prompt = build_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt, 
            enable_think=True, # <--- 关键：允许思考，走 if 分支
            name="gpt"
        )

        # 4. 采样参数 (适度思考)
        sampling_params = SamplingParams(
            temperature=0.2,       # 稍微给一点温度，让思维链自然展开，但保持收敛
            max_tokens=8192,       # 既然允许思考，给 4096 token 足够它写一段简短分析 + 结果
            stop=["<|end|>"]       # 只要模型输出结束符就停
        )
        
        request_id = random_uuid()

        try:
            # 5. 执行推理
            results_generator = self.engine.generate(final_prompt, sampling_params, request_id)
            
            final_output = ""
            finish_reason = None  # [新增] 用于记录结束原因

            async for request_output in results_generator:
                if request_output.finished:
                    # 获取输出对象
                    output_obj = request_output.outputs[0]
                    final_output = output_obj.text
                    finish_reason = output_obj.finish_reason  # [新增] 获取 finish_reason

            # === [新增] 截断检测逻辑 ===
            if finish_reason == "length":
                print(f"\n{'='*20} ⚠️ DETECTED TRUNCATION ⚠️ {'='*20}")
                print(f"Req ID: {request_id}")
                print(f"Reason: Max tokens ({sampling_params.max_tokens}) reached.")
                print(f"Generated Content (Last 200 chars): ...{final_output[-200:]!r}")
                print(f"{'='*60}\n")
                # 这种情况下通常无法提取分数，直接返回 0.0 是对的，但我们需要知道原因
                return 0.0, final_output

            # 6. 结果解析
            # 模型现在的输出会包含：
            # ... analysis content ... <|end|><|start|>assistant<|channel|>final<|message|> [[1]]
            # 所以只要在整个字符串里找 [[1]] 即可
            
            if "[[1]]" in final_output:
                return 1.0, final_output
            elif "[[0]]" in final_output:
                return 0.0, final_output
            
            # [调试] 如果不是截断，但依然没找到标签，打印出来看看它到底说了啥
            print(f"⚠️ [Format Error] Output finished normally but no tag found. Content: {raw_text[:100]}...")
            return 0.0, final_output

        except Exception as e:
            print(f"ERROR in GptOssJudgeReward: {e}")
            return 0.0, f"Error: {str(e)}"