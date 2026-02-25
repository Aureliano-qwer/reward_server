import asyncio
import aiohttp
import time
import json
import numpy as np
import random  # 移到顶部
from dataclasses import dataclass

# ================= 配置区域 =================
# 务必确认端口号！之前日志里是 23334，这里写的是 23334，请确保一致
TARGET_URL = "http://10.103.26.49:23334/compute_score"

# 并发数
CONCURRENCY = 200

# 总请求数
TOTAL_REQUESTS = 200

# 基础 Payload 模板
PAYLOAD_TEMPLATE = {
    "data_source": "cloud_stress_test", # <--- 必须带 cloud 触发 LLM
    "prompt_str": "",                   # 待填充
    "response_str": "",                 # 待填充
    "ground_truth": "",                 # 待填充
    "solution_str": "",                 # 待填充
    "extra_info": {"test_id": "benchmark_001"}
}
# ===========================================

@dataclass
class RequestResult:
    req_id: int
    success: bool
    latency: float
    status_code: int
    error_msg: str = ""

# 【修正 1】增加 payload 参数
async def send_request(session, payload, req_id):
    start_time = time.time()
    try:
        # 【修正 2】使用传入的 payload，而不是全局变量
        async with session.post(TARGET_URL, json=payload) as response:
            await response.read() # 等待完整响应
            latency = time.time() - start_time
            
            # 如果是第一个请求，打印一下返回内容，确保通了
            if req_id == 0:
                try:
                    print(f"🔍 [Req 0] Server Response: {json.loads(await response.text())}")
                except:
                    pass

            return RequestResult(
                req_id=req_id,
                success=response.status == 200, 
                latency=latency, 
                status_code=response.status
            )
    except Exception as e:
        latency = time.time() - start_time
        # print(f"Request {req_id} failed: {e}") # 报错太多可以注释掉
        return RequestResult(req_id=req_id, success=False, latency=latency, status_code=0, error_msg=str(e))

async def worker(session, queue, results):
    while not queue.empty():
        req_id = await queue.get()
        
        # === 生成随机题目，击穿 vLLM 缓存 ===
        num1 = random.randint(10, 9999) # 数字大一点
        num2 = random.randint(10, 9999)
        ans = num1 * num2
        
        # 构造动态 payload
        dynamic_payload = PAYLOAD_TEMPLATE.copy()
        dynamic_payload["prompt_str"] = f"请计算 {num1} * {num2} 是多少？"
        dynamic_payload["response_str"] = f"答案是 {ans}。"
        dynamic_payload["ground_truth"] = str(ans)
        dynamic_payload["solution_str"] = f"请计算 {num1} * {num2} 是多少？ 答案是 {ans}。"
        
        # 传递 payload
        result = await send_request(session, dynamic_payload, req_id)
        results.append(result)
        
        if len(results) % 10 == 0:
            print(f"进度: {len(results)}/{TOTAL_REQUESTS} | 最新状态码: {result.status_code}", end="\r")
            
        queue.task_done()

async def main():
    print(f"🚀 开始压测: {TARGET_URL}")
    print(f"📊 模式: 随机数学题 (击穿 KV Cache)")
    print(f"⚙️  并发数: {CONCURRENCY} | 总请求数: {TOTAL_REQUESTS}")
    print("-" * 40)

    queue = asyncio.Queue()
    for i in range(TOTAL_REQUESTS):
        queue.put_nowait(i)

    results = []
    
    start_time = time.time()
    
    # 【修正 3】客户端优化：解除连接限制，设置超时
    # limit=0 表示不限制连接数，防止客户端排队
    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
    # timeout 设置为 60秒，防止 vLLM 排队时客户端提前断开
    timeout = aiohttp.ClientTimeout(total=60)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # 创建 workers
        tasks = [asyncio.create_task(worker(session, queue, results)) for _ in range(CONCURRENCY)]
        await queue.join()
        for task in tasks:
            task.cancel()

    total_time = time.time() - start_time
    
    # 统计结果
    success_count = sum(1 for r in results if r.success)
    latencies = [r.latency for r in results]
    
    print("\n" + "=" * 40)
    print("✅ 测试结束")
    print("=" * 40)
    print(f"⏱️  总耗时:       {total_time:.2f} 秒")
    print(f"🚀 QPS (吞吐量): {TOTAL_REQUESTS / total_time:.2f} 请求/秒")
    print(f"📈 成功率:       {success_count / TOTAL_REQUESTS * 100:.2f}%")
    
    if latencies:
        print("-" * 40)
        print(f"🐢 平均延迟:     {np.mean(latencies):.4f} s")
        print(f"⚡ P50 延迟:     {np.percentile(latencies, 50):.4f} s")
        print(f"🐌 P95 延迟:     {np.percentile(latencies, 95):.4f} s")
        print(f"🔥 P99 延迟:     {np.percentile(latencies, 99):.4f} s")
    else:
        print("❌ 所有请求均失败，请检查网络或服务日志！")
        # 打印前3个错误看看
        for r in results[:3]:
            if not r.success:
                print(f"Error Sample: {r.error_msg}")

    print("=" * 40)

if __name__ == "__main__":
    asyncio.run(main())