import asyncio
import os
import sys

# 确保能找到 app 模块
sys.path.append(os.getcwd())

from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from app.core.engine import LLMEngineManager
from app.core.config import settings

# 强制覆盖配置，方便本地调试（可选）
# settings.MODEL_PATH = "/path/to/your/model" 

async def test_generation():
    print(f"🚀 [Debug] 正在加载引擎: {settings.MODEL_PATH}")
    print(f"⚙️  [Debug] 并行度 TP: {settings.TENSOR_PARALLEL_SIZE}")
    
    # 1. 获取引擎实例
    try:
        engine = LLMEngineManager.get_instance()
    except Exception as e:
        print(f"❌ 引擎加载失败: {e}")
        return

    # 2. 构造测试 Prompt (复用你的逻辑)
    # 注意：我特意去掉了末尾的换行符，防止模型误判
    prompt_text = "请计算 25 * 25 是多少？"
    response_text = "答案是 625。"
    ground_truth = "625"
    
    judge_prompt = f"""<|im_start|>system
你是一个公正的判卷专家。请判断考生回答是否正确。正确输出 [[1]]，错误输出 [[0]]。<|im_end|>
<|im_start|>user
【题目】：{prompt_text}
【标准答案】：{ground_truth}
【考生回答】：{response_text}
<|im_end|>
<|im_start|>assistant
"""  # <--- 注意这里，如果最后有换行，有些模型会发疯

    print("-" * 40)
    print(f"📝 [Debug] 发送 Prompt:\n{repr(judge_prompt)}")
    print("-" * 40)

    # 3. 设置采样参数
    # 增加 max_tokens 防止被截断，top_p=1 稍微放宽一点
    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=32768, 
        stop=["<|im_end|>"]
    )
    request_id = random_uuid()

    # 4. 执行生成
    print("⏳ [Debug] 开始推理...")
    results_generator = engine.generate(judge_prompt, sampling_params, request_id)

    final_output = ""
    token_count = 0
    
    async for request_output in results_generator:
        # 实时打印生成的每一个 token，看看是不是生成了空
        if not request_output.finished:
             # vLLM 0.x 版本 outputs[0].text 是累积的
             pass 
        
        if request_output.finished:
            final_output = request_output.outputs[0].text
            token_count = len(request_output.outputs[0].token_ids)

    print("-" * 40)
    print(f"✅ [Debug] 推理完成！")
    print(f"📊 [Debug] 生成长度: {token_count} tokens")
    print(f"🔎 [Debug] 原始输出内容 (repr): {repr(final_output)}")
    print("-" * 40)

    # 5. 模拟判分逻辑
    if "[[1]]" in final_output:
        print("🎉 判定结果: 1.0 (正确)")
    elif "[[0]]" in final_output:
        print("❌ 判定结果: 0.0 (错误)")
    else:
        print("⚠️ 判定结果: 0.0 (未匹配到格式)")

if __name__ == "__main__":
    asyncio.run(test_generation())