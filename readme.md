# Universal Reward Model Service (URMS) 技术文档

**版本**: 1.0.0

**最后更新**: 2026-02-07

---

## 1. 项目简介

**Universal Reward Model Service** 是一个高性能、模块化的奖励模型服务系统，专为 LLM 的强化学习（RLHF/RLAIF）或拒绝采样（Rejection Sampling）设计。

### 核心特性

* **混合打分机制**：支持 **规则打分**（格式检查、数学准确率）与 **模型打分**（LLM-as-a-judge）的动态路由。
* **In-Process 推理**：内嵌 `vLLM` 异步引擎（AsyncLLMEngine），无需通过 HTTP 调用外部模型服务，显著降低延迟并提升吞吐量。
* **高并发架构**：基于 FastAPI + Asyncio，充分利用 Python 异步特性，实现 CPU 密集型任务（规则计算）与 IO/GPU 密集型任务（模型推理）的并行处理。
* **模块化设计**：策略层（Rewards）与业务层（Service）解耦，易于扩展新的打分规则。

---

## 2. 系统架构

### 2.1 目录结构

```text
my_reward_server/
├── app/
│   ├── api/             # 接口层：处理 HTTP 请求与响应标准化
│   ├── core/            # 核心层：配置管理、vLLM 引擎单例
│   ├── rewards/         # 策略层：具体的打分算法实现 (BaseReward 及其子类)
│   ├── services/        # 业务层：调度器，负责组装策略并计算总分
│   └── main.py          # 程序入口
├── data/logs/           # 运行日志
├── .env                 # 环境变量配置文件
└── requirements.txt     # 项目依赖

```

### 2.2 数据流向

1. **Request**: 用户发送包含 `prompt`, `response`, `ground_truth` 的 JSON 请求。
2. **API Layer**: `api/routes.py` 接收请求，通过 Pydantic 进行数据校验。
3. **Service Layer**: `services/scorer.py` 根据 `data_source` 字段决定打分策略（走规则还是走模型）。
4. **Reward Layer**:
* 如果是规则：并行调用 `rewards/rule_rewards.py`。
* 如果是模型：调用 `rewards/llm_judge.py`，通过 `core/engine.py` 获取 vLLM 实例进行推理。


5. **Response**: 汇总分数与细节，返回 JSON 响应，并异步写入日志。

---

## 3. 环境与依赖

### 3.1 硬件要求

* **GPU**: 必须支持 CUDA。显存大小需覆盖模型权重 + KV Cache（推荐 24GB+ 用于 8B 模型）。
* **CPU**: 建议 8 核以上，用于处理并发请求和规则计算。
* **RAM**: 32GB+。

### 3.2 软件依赖

* Python 3.10+
* CUDA 11.8 或 12.1
* 核心库：`vllm`, `fastapi`, `uvicorn`, `pydantic`, `pydantic-settings`

---

## 4. 安装与配置

### 4.1 安装步骤

1. **克隆代码仓库**
```bash
git clone <your-repo-url>
cd my_reward_server

```


2. **创建虚拟环境**
```bash
conda create -n reward_server python=3.10
conda activate reward_server

```


3. **安装依赖**
```bash
pip install -r requirements.txt

```



### 4.2 配置文件 (.env)

在项目根目录下创建 `.env` 文件，配置核心参数：

```ini
# 服务配置
PORT=6009
HOST=0.0.0.0
LOG_DIR=data/reward_log

# 模型配置 (必须修改为你本地的实际路径)
MODEL_PATH=/mnt/shared-storage-user/yuanfei/hf_hub/Qwen3-8B/
# 显存占用比例 (0.1 - 0.95)，预留一部分给系统
GPU_MEMORY_UTILIZATION=0.8
# 张量并行 (8B模型通常为1，70B可能需要4或8)
TENSOR_PARALLEL_SIZE=1
# 最大上下文长度
MAX_MODEL_LEN=8192

```

---

## 5. 启动服务

由于 vLLM 引擎会独占 GPU 显存，**严禁使用多 Worker 模式启动**。

### 开发模式 (热重载，不推荐用于加载大模型)

```bash
python -m app.main

```

### 生产模式 (推荐)

使用提供的启动脚本或直接运行模块：

```bash
# 确保指定了正确的 GPU
export CUDA_VISIBLE_DEVICES=0 
python -m app.main

```

*注意：服务启动时会进行模型加载（Warmup），通常需要 1-3 分钟，期间日志会显示 `Loading model weights...`。*

---

## 6. API 接口参考

### 6.1 获取奖励 (`POST /get_reward2`)

计算给定问答对的奖励分数。

**Endpoint**: `http://<IP>:6009/get_reward2`

**Request Body (JSON)**:

| 字段 | 类型 | 必填 | 说明 | 示例 |
| --- | --- | --- | --- | --- |
| `data_source` | string | 否 | 数据源标签，用于触发模型打分 | `"cloud_v1"` (含 "cloud" 触发模型裁判) |
| `prompt_str` | string | 是 | 原始题目 | `"1+1=?"` |
| `response_str` | string/list | 是 | 待打分的回答 | `"答案是2"` |
| `ground_truth` | string/any | 是 | 标准答案 | `"2"` |

**Response Body (JSON)**:

```json
{
  "score": {
    "score": 1.0,           // 总分
    "details": {            // 分数构成明细
      "format": 1.0,
      "accuracy": 1.0,
      "judge": 0.0
    },
    "reason": "rule_based"  // 打分策略来源 ("rule_based" 或 "llm_judge")
  }
}

```

**调用示例 (cURL)**:

```bash
curl -X POST "http://127.0.0.1:6009/get_reward2" \
     -H "Content-Type: application/json" \
     -d '{
           "data_source": "math_dataset",
           "prompt_str": "计算 25 * 4",
           "response_str": "<think>...</think><answer>100</answer>",
           "ground_truth": "100"
         }'

```

---

## 7. 开发者指南 (Extension Guide)

### 如何添加一个新的奖励规则？

假设你需要添加一个 **“代码风格检查”** 奖励。

1. **新建奖励类**：
在 `app/rewards/` 下创建 `code_style.py`：
```python
from app.rewards.base import BaseReward

class CodeStyleReward(BaseReward):
    async def compute(self, prompt: str, response: str, ground_truth: str) -> float:
        # 简单的逻辑：有注释就给分
        if "#" in response or "//" in response:
            return 1.0
        return 0.0

```


2. **注册到调度器**：
修改 `app/services/scorer.py`：
```python
# 导入新类
from app.rewards.code_style import CodeStyleReward

class ScorerService:
    def __init__(self):
        # ... 原有初始化
        self.style_reward = CodeStyleReward() # 实例化

    async def calculate(self, req: RewardRequest) -> ScoreDetail:
        # ...
        # 加入到计算任务中
        style_task = self.style_reward.compute(...)
        # 更新 asyncio.gather 和加权逻辑
        # ...

```



---

## 8. 故障排除 (Troubleshooting)

### Q1: 启动时报错 `CUDA out of memory`

* **原因**: 显存不足以加载模型。
* **解决**:
1. 减小 `.env` 中的 `GPU_MEMORY_UTILIZATION` (例如从 0.8 改为 0.6)。
2. 减小 `MAX_MODEL_LEN`。
3. 确认没有其他进程（如其他训练任务）占用这块 GPU。



### Q2: 服务启动后无响应，日志卡在 `Initializing vLLM engine`

* **原因**: 模型权重较大，正在从磁盘加载到内存。
* **解决**: 耐心等待 1-3 分钟。如果长时间无反应，检查磁盘 I/O 是否正常。

### Q3: 多个请求同时发过来，报错 `RuntimeError: ...`

* **原因**: 可能使用了多进程启动 (`uvicorn ... --workers 2`) 导致争抢 GPU。
* **解决**: 必须使用单进程启动 (`workers=1`)，并发由内部的 `AsyncLLMEngine` 处理。

### Q4: 想要更换打分模型？

* **解决**: 修改 `.env` 中的 `MODEL_PATH` 重启服务即可。无需修改代码。

---

## 9. 日志说明

日志文件位于 `data/reward_log/reward.log`。
日志格式为 JSON Lines，包含完整的输入数据和打分结果，可直接用于后续的数据分析或模型迭代。

**日志示例**:

```json
{
  "cur_date": "2026-02-07 10:00:00",
  "input_data": { ... },
  "score": { "score": 1.0, ... },
  "judge_type": "rule_based"
}

```