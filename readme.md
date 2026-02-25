# Universal Reward Model Service (URMS) 技术文档

**版本**: 1.1.0 (Updated)
**最后更新**: 2026-02-25

---

## 1. 项目简介

**Universal Reward Model Service** 是一个高性能、模块化的奖励模型服务系统，专为 LLM 的强化学习（PPO/GRPO/RLAIF）或拒绝采样（Rejection Sampling）设计。

### 核心特性

* **混合打分机制 (Rule + LLM)**：支持 **规则打分**（格式检查、正则匹配）与 **模型打分**（LLM-as-a-judge）的无缝结合，保证打分的下限与上限。
* **5 级细粒度裁判 (5-Point Judge)**：LLM Judge 采用 5 分制（`Excellent`, `Good`, `Fair`, `Weak`, `Bad`），有效解决 GRPO 训练中 Advantage 分布坍塌（梯度稀疏）的问题，提供稠密的奖励信号。
* **鲁棒的异常防护**：内置针对长文本的 **截断检测 (Truncation Detection)** 与 **幻觉防护 (Anti-Hallucination)**，有效防止模型通过瞎编乱造骗取高分。
* **In-Process 推理**：内嵌 `vLLM` 异步引擎（AsyncLLMEngine），极低延迟，支持高并发。

---

## 2. 系统架构

### 2.1 目录结构

```text
my_reward_server/
├── app/
│   ├── api/             # 接口层：处理 HTTP 请求与响应标准化 (routes.py)
│   ├── core/            # 核心层：配置管理、vLLM 引擎单例 (engine.py)
│   ├── rewards/         # 策略层：具体的打分算法实现 (如 llm_judge.py)
│   ├── services/        # 业务层：调度器，负责组装策略并计算总分 (scorer.py)
│   └── main.py          # 程序入口
├── data/logs/           # 运行日志
├── .env                 # 环境变量配置文件
└── requirements.txt     # 项目依赖

```

### 2.2 数据流向

1. **Request**: 用户 (训练框架) 发送包含 `prompt_str`, `response_str`, `ground_truth` 等字段的 JSON 请求。
2. **API Layer**: `/compute_score` 接收请求，解析并分发。
3. **Service Layer**: 判定是纯规则任务，还是需要启用 LLM Judge 进行代码/逻辑审计。
4. **Reward Layer**:
* 若触发 LLM Judge，构建带 `<reasoning>` 和 `<logic_score>` 的 System Prompt。
* 调用 `vLLM` 引擎生成审计结果，解析出 0.0 ~ 1.0 的 5 档分数。
* 检测是否发生 `finish_reason == "length"`（截断），若截断则分数归零。


5. **Response**: 将规则分与逻辑分加权汇总，返回最终的打分细节，并异步写入日志。

---

## 3. 环境与依赖

### 3.1 硬件要求

* **GPU**: 必须支持 CUDA。显存大小需覆盖模型权重 + KV Cache（推荐 24GB+ 用于 8B 模型）。
* **CPU**: 建议 8 核以上，用于处理并发请求和规则计算。
* **RAM**: 32GB+ (注意：`/dev/shm` 共享内存需要充足配置，建议 8Gi+)。

### 3.2 软件依赖

* Python 3.10+
* CUDA 11.8 或 12.1
* 核心库：`vllm`, `fastapi`, `uvicorn`, `pydantic`

---

## 4. 安装与配置

### 4.1 配置文件 (.env)

在项目根目录下创建 `.env` 文件，配置核心参数：

```ini
# 服务配置
PORT=23334
HOST=0.0.0.0
LOG_DIR=data/reward_log

# 模型配置
MODEL_PATH: str = "openai/gpt-oss-120b"
GPU_MEMORY_UTILIZATION: float = 0.9
TENSOR_PARALLEL_SIZE: int = 8
MAX_MODEL_LEN: int = 32768

```

---

## 5. 启动服务

> ⚠️ **重要警告**：
> 1. vLLM 引擎会独占 GPU 显存，**严禁使用多 Worker 模式启动**。
> 2. **代码修改后必须手动重启** (Ctrl+C 杀掉进程后重新运行)。严禁使用 `--reload` 热重载，否则会导致 GPU 显存泄漏 (OOM) 和僵尸进程。
> 
> 

### 生产模式 / 稳定模式 (强烈推荐)

```bash
# 3. 启动服务
bash launch_reward_api_rjob.sh
```

*注意：服务启动时会进行模型加载（Warmup），通常需要 1-3 分钟，期间日志会显示 `Loading model weights...`。*

---

## 6. API 接口参考

### 6.1 获取奖励 (`POST /compute_score`)

计算给定回复的奖励分数。

**Endpoint**: `http://<IP>:6009/compute_score`

**Request Body (JSON)**:

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `data_source` | string | 否 | 数据源标签（决定是否触发 test/train 模式） |
| `prompt_str` | string | 是 | 原始题目 / 指令 |
| `response_str` | string | 是 | 待打分的模型输出 |
| `ground_truth` | string | 是 | 题目对应的标准答案 / 预期结果 |
| `extra_info` | dict | 否 | 包含额外的环境信息或错误类型字典 |

**Response Body (JSON)**:

```json
{
  "output": {
    "score": 1.4,             // 综合总分
    "details": {
      "llm_judge": 1.0,       // 逻辑审计分 (0.0, 0.25, 0.5, 0.75, 1.0)
      "llm_reasoning": "<reasoning>...</reasoning>", // Judge 的推理链
      "rule_score": 2.0,      // 规则硬匹配分
      "mode": "train_shaping" 
    },
    "reason": "LLM + Rule"    // 得分来源
  }
}

```

---

## 7. 故障排除 (Troubleshooting)

### Q1: 启动时或推理中途 worker 进程突然崩溃，报错 `RuntimeError: CUDA driver error: invalid argument` 或 `EngineCore_DP0 failed to start`

* **原因**: 这是 vLLM 最新的 V1 架构使用了 PyTorch Symmetric Memory 进行跨进程 IPC 通信，但当前的 Docker/K8s 环境的 `/dev/shm` 权限不足或 CUDA 驱动不兼容导致的。
* **解决**: 在启动服务前，设置环境变量退回稳定的 V0 引擎：`export VLLM_USE_V1=0`，或者禁用对称内存 `export VLLM_DISABLE_SYMMETRIC_MEMORY=1`。

### Q2: 启动时报错 `CUDA out of memory`

* **原因**: 显存不足以加载模型权重 + KV Cache。
* **解决**:
1. 减小 `.env` 中的 `GPU_MEMORY_UTILIZATION` (例如从 0.8 改为 0.7)。
2. 确认服务器上没有其他进程占用该 GPU (`nvidia-smi`)。



### Q3: LLM Judge 的分数全是 0，且日志出现 `⚠️ DETECTED TRUNCATION`

* **原因**: 模型的回答过长，达到了 `max_tokens` (8192) 上限被强制截断，导致没有输出合法的 `[[Tag]]`。
* **解决**: URMS 已经内置了截断防护（截断按 0 分处理）。如果截断率极高，说明模型陷入了“循环废话”，建议在强化学习训练端调高 KL 惩罚，或检查 Prompt 是否引发了无限重复。

### Q4: 优势 (Advantage) 曲线几乎是直线，模型学不到东西？

* **原因**: 原有的 3 分制（好/坏）导致组内分数方差极小。
* **解决**: URMS 现已升级为 **5分制 (Excellent/Good/Fair/Weak/Bad)**。如果依然方差过小，可以考虑微调 LLM Judge 生成时的 `temperature` (如 0.2 -> 0.4) 增加打分的多样性。