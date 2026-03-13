#!/bin/bash
# ============================================================
# 启动 Universal Reward Model Service (URMS)
# 集成 vLLM 或 SGLang In-Process 推理
# 绑定到 0.0.0.0，默认端口 6009
#
# 推理框架切换（环境变量，默认 vllm）:
#   export INFERENCE_BACKEND=vllm   # 使用 vLLM (默认)
#   export INFERENCE_BACKEND=sglang # 使用 SGLang (需安装 sglang)
# ============================================================

# 1. 配置区域 (请根据实际情况修改)
# ------------------------------------------------------------
# 指定使用的 GPU ID (例如 "0" 或 "0,1")

export INFERENCE_BACKEND=sglang

export HF_HUB_CACHE=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/
export HF_HUB_OFFLINE=1
export VLLM_USE_MODELSCOPE=0
export VLLM_USE_HARMONY=0
export TIKTOKEN_RS_CACHE_DIR=/mnt/shared-storage-user/ailab-llmkernel/huangzixian/hf_hub/o200k_base_tokeinzer

# 你的项目根目录
PROJECT_ROOT="/mnt/shared-storage-user/ailab-llmkernel/yangkaichen/Code_judgement/reward_server"

# Conda 环境路径 (请修改为你创建的 reward_server 环境路径)
# 如果不确定，可以用 'conda env list' 查看
# CONDA_ENV_PATH="/mnt/shared-storage-user/ailab-llmkernel/huangzixian/miniconda3/envs/ykc-codejudge"  # vllm
CONDA_ENV_PATH="/mnt/shared-storage-user/ailab-llmkernel/huangzixian/miniconda3/envs/ykc-rewardapi-sglang" # sglang
# 或者如果是 reference 里的 hzx:
# CONDA_ENV_PATH="/mnt/shared-storage-user/yuanfei/miniconda3/envs/hzx"

# 服务配置
HOST="0.0.0.0"
PORT=23334
SERVICE_NAME="reward-server-qwen8b"

# 显式导出给 Python，避免系统 HOST 环境变量污染监听地址
export APP_HOST="${APP_HOST:-$HOST}"
export APP_PORT="${APP_PORT:-$PORT}"

# ============================================================

# 2. 环境初始化
# ------------------------------------------------------------
echo "正在切换到项目目录: $PROJECT_ROOT"
cd "$PROJECT_ROOT" || { echo "❌ 错误: 找不到项目目录 $PROJECT_ROOT"; exit 1; }


source /mnt/shared-storage-user/ailab-llmkernel/huangzixian/miniconda3/bin/activate
conda activate "$CONDA_ENV_PATH"


if [ $? -ne 0 ]; then
    echo "❌ 错误: 无法激活 Conda 环境: $CONDA_ENV_PATH"
    exit 1
fi
echo "✅ Conda environment activated: $(basename $CONDA_PREFIX)"

# 3. 获取本机 IP
# ------------------------------------------------------------
LOCAL_IP=$(hostname -I | awk '{for(i=1;i<=NF;i++){if($i ~ /^10\./ || $i ~ /^192\./){print $i; exit}}}')
if [ -z "$LOCAL_IP" ]; then
  LOCAL_IP="localhost"
fi

# 4. 打印启动横幅
# ------------------------------------------------------------
echo "============================================================"
echo "🚀 启动 Reward Model Service"
echo "项目路径:  $PROJECT_ROOT"
echo "显卡设备:  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "监听地址:  $HOST:$PORT"
echo "------------------------------------------------------------"
echo "✅ 本机访问:  http://127.0.0.1:${PORT}/compute_score"
echo "🌐 局域网访问: http://${LOCAL_IP}:${PORT}/compute_score"
echo "------------------------------------------------------------"
echo "示例调用 (测试服务是否正常):"
echo "curl -X POST http://${LOCAL_IP}:${PORT}/compute_score \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{"
echo "    \"data_source\": \"cloud_test\","
echo "    \"prompt_str\": \"1+1=?\","
echo "    \"response_str\": \"答案是2\","
echo "    \"ground_truth\": \"2\""
echo "  }'"
echo "------------------------------------------------------------"
echo "🧩 端口占用检查："
echo "    netstat -ntlp | grep ${PORT}"
echo "============================================================"
echo

# 5. 启动服务 (后台运行)
# ------------------------------------------------------------
# 使用时间戳区分日志，避免覆盖
LOG_FILE="logs/service_${SERVICE_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "正在启动 Python 服务... 日志将写入 $LOG_FILE"
echo "注意：初次启动需要加载模型权重，可能需要 1-3 分钟，请耐心等待..."

# 使用 uvicorn 启动，workers 必须为 1 (vLLM/SGLang 独占 GPU)
nohup python -m app.main > "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo "✅ 服务已在后台启动 (PID: $SERVER_PID)"
echo "正在等待服务端口 $PORT 就绪..."

# 6. 健康检查循环
# ------------------------------------------------------------
# 循环检查端口是否被监听，最多等待 180 秒
MAX_RETRIES=60
count=0
while [ $count -lt $MAX_RETRIES ]; do
    if ss -ntlp | grep -q ":$PORT "; then
        echo "🎉 服务启动成功！端口 $PORT 已在监听。"
        break
    fi
    
    # 检查进程是否意外挂掉
    if ! ps -p $SERVER_PID > /dev/null; then
        echo "❌ 错误: 服务进程 (PID: $SERVER_PID) 已退出。"
        echo "请查看日志获取详细错误信息:"
        echo "------------------------------------------------------------"
        tail -n 20 "$LOG_FILE"
        echo "------------------------------------------------------------"
        exit 1
    fi

    echo -n "."
    sleep 3
    count=$((count + 1))
done

if [ $count -ge $MAX_RETRIES ]; then
    echo
    echo "⚠️  警告: 等待超时，服务可能仍在加载模型，或者发生了错误。"
    echo "请检查日志文件: $LOG_FILE"
fi

echo "============================================================"
echo "🎉 服务已就绪！脚本进入前台保持模式..."
echo "正在实时输出日志内容 ($LOG_FILE):"
echo "============================================================"

# 【关键修改】
# 使用 tail -f 挂起脚本，阻止其退出。
# 这样容器会一直运行，且你能在 K8s/Docker 日志里直接看到输出。
tail -f "$LOG_FILE" &

# 等待该 tail 进程（或者等待 Python 进程）
wait $!