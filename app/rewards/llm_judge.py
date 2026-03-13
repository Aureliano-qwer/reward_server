import uuid

from app.core.engine import LLMEngineManager
from app.rewards.base import BaseReward


def _request_id() -> str:
    return uuid.uuid4().hex


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
        sampling_params = {
            "temperature": 0.0,
            "max_new_tokens": 4096,
            "stop": ["<|im_end|>"],
        }
        request_id = _request_id()

        final_output = ""
        async for chunk in self.engine.generate(judge_prompt, sampling_params, request_id):
            if chunk.get("finished"):
                final_output = chunk.get("text", "")

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
        system_prompt_ver2 = """You are a Senior Competitive Programming Code Audit Expert with over 10 years of experience. Your task is to audit an analysis report generated by a "Junior Code Analyst" (Student Model) regarding a specific programming problem and its solution.

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

        system_prompt_v3 = """You are a Senior Competitive Programming Code Audit Expert. Your goal is to evaluate the **Reasoning Quality** of a "Junior Code Analyst" (Student Model).

You have access to:
1. The <<<problem>>> description.
2. The user's submitted <<<code>>>.
3. The [Ground Truth] (root cause of failure/success).
4. The [Student's Analysis] (the text you are auditing).

### Your Specific Role
The "Verdict" (Passed/Failed) and "Error Type" (TLE/WA/RE) have already been checked by a rule-based engine. **Do not focus on scoring the final verdict.**
**Your job is to judge the Logic Chain and Hallucinations.** Does the student's explanation actually make sense based on the provided code?

### Evaluation Workflow
1. **Reality Check (Anti-Hallucination)**: 
   - Scan the [Student's Analysis]. If they cite specific variables (e.g., `cnt`, `dp_table`) or line numbers, **verify they actullys exist in the <<<code>>>**.
   - If the student claims "Division by Zero on line 5", check line 5. If line 5 is just an assignment, this is a **Hallucination**.
2. **Causal Analysis (Root Cause)**:
   - Compare the student's reasoning with the [Ground Truth].
   - If GT says "TLE due to O(N^2)", but the student argues "Syntax Error in print statement", the student is **guessing** (even if they both say "Failed").

### Scoring Criteria (Output One Tag Only)

**[[Perfect]]** (Score: 1.0)
- The analysis is logically sound and factually accurate.
- It correctly identifies the **root cause** described in the [Ground Truth].
- All cited code snippets/variables exist and are interpreted correctly.
- *Use this for high-quality, precise analysis.*

**[[Partial]]** (Score: 0.5)
- The analysis is "mostly harmless" but weak.
- It might point out a valid bug, but misses the *primary* root cause (e.g., finding a minor style issue when the real bug is Logic Error).
- Or the reasoning is vague (e.g., "The logic is wrong" without specifying why), but not hallucinated.
- *Use this to encourage the model for being on the right track, even if not perfect.*

**[[Bad]]** (Score: 0.0)
- **Hallucination**: Cites non-existent code, variables, or errors.
- **Logical Disconnect**: The reasoning contradicts the code (e.g., claiming a loop is infinite when it clearly terminates).
- **Lucky Guess**: The student gets the Verdict right (e.g., "Failed") but for a completely wrong/fake reason.
- *Penalize this severely.*

### Output Format
1. **Reasoning**: Inside `<reasoning>` tags, briefly explain your judgment. Focus on *why* you chose the tag (e.g., "Student hallucinated variable 'x'").
2. **Logic Score**: Inside `<logic_score>` tags, output strictly one of: `[[Perfect]]`, `[[Partial]]`, or `[[Bad]]`.

Example Output:
<reasoning>
The student correctly concludes the code fails, but their reasoning is flawed. They claim a "IndexError on line 10", but line 10 is inside a safe try-catch block. The actual error (per GT) is a math logic error on line 3. The student is hallucinating a runtime error to justify the failure.
</reasoning>
<logic_score>
[[Bad]]
</logic_score>
"""

        system_prompt_ver4 = """You are a Senior Competitive Programming Code Audit Expert. Your goal is to evaluate the **Reasoning Quality** of a "Junior Code Analyst" (Student Model).

You have access to:
1. The <<<problem>>> description.
2. The user's submitted <<<code>>>.
3. The [Ground Truth] (root cause of failure/success).
4. The [Student's Analysis] (the text you are auditing).

### Your Specific Role
The "Verdict" (Passed/Failed) and "Error Type" (TLE/WA/RE) have already been checked by a rule-based engine. **Do not focus on scoring the final verdict.**
**Your job is to judge the Logic Chain and Hallucinations.** Does the student's explanation actually make sense based on the provided code?

### Evaluation Workflow
1. **Reality Check (Anti-Hallucination)**: 
    - Scan the [Student's Analysis]. If they cite specific variables (e.g., `cnt`, `dp_table`) or line numbers, **verify they actually exist in the <<<code>>>**.
    - If the student claims "Division by Zero on line 5", check line 5. If line 5 is just an assignment, this is a **Hallucination**.
2. **Causal Analysis (Root Cause)**:
    - Compare the student's reasoning with the [Ground Truth].
    - If GT says "TLE due to O(N^2)", but the student argues "Syntax Error", the student is **guessing**.

### Scoring Criteria (Output One Tag Only)

**[[Excellent]]** (Score: 1.0)
- Flawless analysis. Correctly identifies the **root cause** matches [Ground Truth].
- All code citations are accurate. Logic is precise and concise.
- *The ideal standard.*

**[[Good]]** (Score: 0.75)
- Correctly identifies the root cause and main logic is sound.
- May contain minor nitpicks or slight verbosity, but **zero hallucinations**.
- The core reasoning is solid, just not "perfectly" polished.

**[[Fair]]** (Score: 0.5)
- Points out a valid bug/issue, but misses the *primary* root cause (e.g., finds a logical flaw but misses the TLE cause).
- Reasoning is somewhat vague but factually correct based on the code (no hallucinations).
- *"Right direction, wrong destination."*

**[[Weak]]** (Score: 0.25)
- Misses the point significantly or focuses on irrelevant style issues.
- Logic is weak or confusing, but **does not** contain blatant hallucinations of code.
- Barely acceptable reasoning.

**[[Bad]]** (Score: 0.0)
- **Hallucination**: Cites non-existent code, variables, or runtime errors.
- **Logical Disconnect**: Reasoning contradicts code reality.
- **Lucky Guess**: Gets the verdict right but for a completely wrong/fake reason.
- *Strictly penalize any hallucination.*

### Output Format
1. **Reasoning**: Inside `<reasoning>` tags, briefly explain your judgment. Focus on *why* you chose the tag.
2. **Logic Score**: Inside `<logic_score>` tags, output strictly one of: `[[Excellent]]`, `[[Good]]`, `[[Fair]]`, `[[Weak]]`, or `[[Bad]]`.

Example Output:
<reasoning>
The student correctly concludes the code fails. However, they claim an "IndexError on line 10", but line 10 is safe. The actual error (per GT) is a math logic error. This is a hallucination.
</reasoning>
<logic_score>
[[Bad]]
</logic_score>
"""

        system_prompt_ver5="""You are a Senior Competitive Programming Code Audit Expert (Reward Model). Your strict goal is to evaluate the **Deep Logical Reasoning Quality** of a "Junior Code Analyst" (Student Model). 

You have access to:
1. The <<<problem>>> description.
2. The user's submitted <<<code>>>.
3. The [Ground Truth] (The absolute primary root cause of failure/success).
4. The [Student's Analysis] (the text you are auditing).

### CRITICAL ANTI-HACKING WARNING
Students often try to "cheat" the evaluation by quoting real variable names and exact code lines to appear factual, while wrapping them in completely fake or flawed math/algorithmic logic. 
**Quoting real code does NOT mean there is no hallucination. "Logical Hallucination" (making up fake algorithms, impossible state transitions, or wrong mathematical conclusions based on real variables) is the most severe offense and must be scored [[Bad]].**

### Evaluation Workflow (Must be followed strictly in `<reasoning>`)
1. **Independent Verification First**: Before reading the student's analysis, briefly establish what the [Ground Truth] actually means in the context of the <<<code>>>. 
2. **Deep Logical Verification**:
    - Do NOT just check if cited lines/variables exist. You MUST mentally execute or mathematically verify the student's claim.
    - If the student claims: "Line 10 causes `count` to be incorrect because X", you must verify if X is mathematically/algorithmically true. If X is nonsense, it is a Logical Hallucination.
3. **Strict Root Cause Alignment**:
    - The student MUST identify the *Primary Root Cause* defined in the [Ground Truth]. 
    - If the [Ground Truth] is "TLE due to O(N^2) loop", but the student focuses on a minor edge case that causes "WA", the student has failed. Do not give high partial credit for irrelevant findings.

### Strict Scoring Criteria (Output One Tag Only)

**[[Excellent]]** (Score: 1.0)
- Flawless analysis. Accurately maps to the [Ground Truth] primary root cause.
- The mathematical and algorithmic explanation is completely sound and logically rigorous.

**[[Good]]** (Score: 0.75)
- Correctly identifies the [Ground Truth] root cause.
- Logic is generally sound, but the explanation might be slightly convoluted or imprecise. 
- ABSOLUTELY NO logical hallucinations.

**[[Fair]]** (Score: 0.25) *[Note: Severely penalized]*
- The student correctly identifies a valid, executable bug in the code, BUT it is NOT the primary root cause listed in the [Ground Truth].
- The analysis of this secondary bug must still be 100% logically sound.
- *"Found a real issue, but missed the main point."*

**[[Weak]]** (Score: 0.0)
- The student guesses the correct final verdict (e.g., WA, TLE) but provides vague, generic reasoning (e.g., "the logic is wrong", "fails on edge cases") without pointing to the exact mathematical/algorithmic flaw.
- Shotgun approach: listing multiple possible errors hoping one sticks.

**[[Bad]]** (Score: -1.0) *[Strictly Penalize]*
- **Logical Hallucination**: Quotes real variables/lines but makes mathematically or algorithmically false claims about how they execute.
- **Contradiction**: Claims a variable takes a value that is impossible given the code's constraints.
- **Fake Root Cause**: Invents a reason that contradicts the [Ground Truth].

### Output Format
1. **Reasoning**: Inside `<reasoning>` tags, you MUST structure your thought process exactly like this:
   - `[GT Analysis]`: Briefly state the true root cause.
   - `[Student Claim]`: Summarize the student's core argument.
   - `[Logic Check]`: Prove or disprove the student's algorithmic logic. Are they faking the math/logic?
   - `[Conclusion]`: Why this tag.
2. **Logic Score**: Inside `<logic_score>` tags, output strictly one of: `[[Excellent]]`, `[[Good]]`, `[[Fair]]`, `[[Weak]]`, or `[[Bad]]`.
"""

        system_prompt_ver6="""You are a Senior Competitive Programming Code Audit Expert. Your goal is to evaluate the **Reasoning Quality** of a "Junior Code Analyst" (Student Model).

You have access to:
1. The <<<problem>>> description.
2. The user's submitted <<<code>>>.
3. The [Ground Truth] (The absolute primary root cause of failure/success).
4. The [Student's Analysis] (the text you are auditing).

### Your Specific Role
The final "Verdict" (Passed/Failed) and "Error Type" (TLE/WA/RE) have already been evaluated by a rule-based engine. **Do not focus on scoring the final verdict.**
Your ONLY job is to act as a logic and hallucination detector. 

### CRITICAL ANTI-HACKING WARNING
Students often try to "cheat" by quoting real variable names and exact code lines to appear factual, while wrapping them in fake mathematical or algorithmic logic. 
**Quoting real code does NOT mean there is no hallucination. "Logical Hallucination" (making up fake algorithms, impossible state transitions, or wrong mathematical conclusions based on real variables) is the most severe offense and must be strictly penalized.**

### Evaluation Workflow & Output Format
You MUST structure your response EXACTLY using the following XML tags. Do not skip any sections.

**Step 1: <independent_analysis>**
*Do this BEFORE evaluating the student.*
1. Read the <<<code>>> and the [Ground Truth].
2. Independently locate the exact lines or logical blocks causing the issue.
3. Briefly explain the mathematical or algorithmic reason for the failure. Establish your own baseline of truth here.

**Step 2: <student_critique>**
*Now, evaluate the [Student's Analysis] against your baseline.*
1. What is the student's core argument?
2. **Reality Check (Textual)**: Do the lines/variables they cite actually exist?
3. **Reality Check (Logical)**: Is their claimed algorithmic behavior actually possible in the code? Or did they invent a fake runtime error/math operation using real variables?
4. **Causal Alignment**: Does their logic align with the [Ground Truth] and your independent analysis?

**Step 3: <logic_score>**
Output strictly one tag based on the criteria below.

### Scoring Criteria

**[[Excellent]]** (Score: 1.0)
- Flawless analysis. Correctly identifies the root cause matching [Ground Truth] and your independent analysis.
- Algorithmic logic is precise, strictly true, and directly addresses the core flaw.

**[[Good]]** (Score: 0.75)
- Correctly identifies the root cause and main logic is sound.
- May contain minor verbosity, but absolutely **zero textual or logical hallucinations**.

**[[Fair]]** (Score: 0.5)
- Points out a valid, factually correct secondary bug/issue in the code, but misses the *primary* root cause.
- Reasoning is somewhat vague but mathematically/algorithmically correct based on the code.
- *"Right direction, wrong destination."*

**[[Weak]]** (Score: 0.25)
- Misses the point significantly or provides generic, empty reasoning (e.g., "fails on edge cases" without proving how).
- Shotgun approach (listing multiple random errors).
- No blatant hallucinations, but the logic chain is very weak.

**[[Bad]]** (Score: 0.0)
- **Logical Hallucination**: Uses real variable names/lines but invents fake mathematical operations, impossible state transitions, or fake runtime errors.
- **Textual Hallucination**: Cites non-existent code or variables.
- **Contradiction**: Reasoning completely contradicts code reality, even if they guessed the correct error type.

Example Output Structure:
<independent_analysis>
The [Ground Truth] states Wrong Answer due to incorrect boundary conditions. Looking at the code, the loop `for i in range(1, n):` misses the final element. The array size is `n+1`, so it should be `range(1, n+1)`. The algorithm fails to process the last item.
</independent_analysis>
<student_critique>
The student claims the code fails because "variable `cnt` on line 5 causes a Division by Zero". While `cnt` exists on line 5, it is used in an addition operation (`ans += cnt`), not division. The student completely hallucinated the logical behavior of the variable and missed the loop boundary issue entirely.
</student_critique>
<logic_score>
[[Bad]]
</logic_score>
"""

        system_prompt = """You are a Senior Competitive Programming Code Audit Expert. Your goal is to evaluate the **Reasoning Quality** of a "Junior Code Analyst" (Student Model).

You have access to:
1. The <<<problem>>> description.
2. The user's submitted <<<code>>>.
3. The [Ground Truth] (The absolute primary root cause of failure/success).
4. The [Extra Info] (A list of other known errors, secondary bugs, or failed test cases).
5. The [Student's Analysis] (the text you are auditing).

### Your Specific Role
The final "Verdict" and "Error Type" have already been evaluated by a rule-based engine. **Do not focus on scoring the final verdict.**
Your ONLY job is to act as a logic and hallucination detector. 

### CRITICAL ANTI-HACKING WARNING
Students often try to "cheat" by quoting real variable names and exact code lines to appear factual, while wrapping them in fake mathematical or algorithmic logic. 
**Quoting real code does NOT mean there is no hallucination. "Logical Hallucination" (making up fake algorithms, impossible state transitions, or wrong mathematical conclusions based on real variables) is the most severe offense and must be strictly penalized.**

### Evaluation Workflow & Output Format
You MUST structure your response EXACTLY using the following XML tags. Do not skip any sections.

**Step 1: <comprehensive_error_analysis>**
*Do this BEFORE reading the student's analysis.*
1. Review the [Ground Truth] and the issues listed in [Extra Info].
2. For the primary [Ground Truth] and each distinct issue in [Extra Info], briefly identify the specific lines of code and the mathematical/algorithmic reason for the failure. 
3. Keep this section concise but mathematically precise. Establish your complete "map" of the code's flaws here.

**Step 2: <student_critique>**
*Now, evaluate the [Student's Analysis] against your established map of flaws.*
1. What is the student's core argument?
2. **Reality Check (Textual)**: Do the lines/variables they cite actually exist?
3. **Reality Check (Logical)**: Is their claimed algorithmic behavior actually possible in the code? Or did they invent a fake runtime error/math operation using real variables?
4. **Defect Mapping**: Does the student's identified bug map to the [Ground Truth] or one of the valid bugs in [Extra Info]? 

**Step 3: <logic_score>**
Output strictly one tag based on the criteria below.

### Scoring Criteria

**[[Excellent]]** (Score: 1.0)
- Flawless analysis. Correctly identifies the primary root cause matching [Ground Truth].
- Algorithmic logic is precise, strictly true, and directly addresses the core flaw.

**[[Good]]** (Score: 0.75)
- Identifies a valid, major bug (either GT or a critical one from Extra Info) and the main logic is sound.
- May contain minor verbosity, but absolutely **zero textual or logical hallucinations**.

**[[Fair]]** (Score: 0.5)
- Points out a valid, factually correct secondary bug/issue from [Extra Info], but clearly misses the *primary* root cause.
- Reasoning is somewhat vague but mathematically/algorithmically correct based on the code.
- *"Right direction, wrong destination."*

**[[Weak]]** (Score: 0.25)
- Misses the point significantly or provides generic, empty reasoning (e.g., "fails on edge cases" without proving how).
- No blatant hallucinations, but the logic chain is very weak and fails to accurately map to any real bug in GT or Extra Info.

**[[Bad]]** (Score: 0.0)
- **Logical Hallucination**: Uses real variable names/lines but invents fake mathematical operations, impossible state transitions, or fake runtime errors.
- **Textual Hallucination**: Cites non-existent code or variables.
- **Contradiction**: Reasoning completely contradicts code reality, even if they guessed the correct error type.
"""

        user_prompt = f"""### [Problem & Code]
{prompt}

### [Ground Truth]
{ground_truth}

### [Extra_info]
{extra_info}

### [Student's Analysis (To be Evaluated)]
{response}
"""

        # 3. 构建 Prompt
        final_prompt = build_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt, 
            enable_think=True, 
            name="gpt"
        )

        sampling_params = {
            "temperature": 0.2,
            "max_new_tokens": 8192,
            "stop": ["<|end|>"],
        }
        request_id = _request_id()

        try:
            final_output = ""
            finish_reason = None

            async for chunk in self.engine.generate(final_prompt, sampling_params, request_id):
                if chunk.get("finished"):
                    final_output = chunk.get("text", "")
                    finish_reason = chunk.get("finish_reason")

            # === [截断检测逻辑] ===
            max_tokens = sampling_params.get("max_new_tokens", 8192)
            if finish_reason == "length":
                print(f"\n{'='*20} ⚠️ DETECTED TRUNCATION ⚠️ {'='*20}")
                print(f"Req ID: {request_id}")
                print(f"Reason: Max tokens ({max_tokens}) reached.")
                print(f"Generated Content (Last 200 chars): ...{final_output[-200:]!r}")
                print(f"{'='*60}\n")
                return 0.0, final_output

            # 6. 结果解析逻辑函数 (更新为5分制)
            def compute_logic_score(llm_response: str) -> float:
                # 优先级从高到低判断，或者根据Tag判断
                if "[[Excellent]]" in llm_response:
                    return 1.0
                elif "[[Good]]" in llm_response:
                    return 0.75
                elif "[[Fair]]" in llm_response:
                    return 0.5
                elif "[[Weak]]" in llm_response:
                    return 0.25
                elif "[[Bad]]" in llm_response:
                    return 0.0
                else:
                    # 兼容旧版本Tag或处理异常情况
                    if "[[Perfect]]" in llm_response: return 1.0
                    if "[[Partial]]" in llm_response: return 0.5
                    
                    print(f"⚠️ [Format Error] No valid tag found in response. Req ID: {request_id}")
                    print(f"Content Preview: {llm_response[:200]!r}...") 
                    return 0.0
            
            # 计算分数
            score = compute_logic_score(final_output)
            
            # 返回结果
            return score, final_output

        except Exception as e:
            print(f"ERROR in GptOssJudgeReward: {e}")
            return 0.0, f"Error: {str(e)}"