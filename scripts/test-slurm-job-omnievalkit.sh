#!/bin/bash
#SBATCH -J omnievalkit-gemma4-12b      # job name
#SBATCH -o %x.o%j            # single STDOUT/STDERR output file jobname.o<job number>
#SBATCH -p sae          # request gpushort partition
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8                 # 8 cores
#SBATCH --cpus-per-gpu=8     # 8 cores per GPU
#SBATCH -t 24:0:0             # 24 hour runtime (required to run on the short partition)
#SBATCH --mem-per-cpu=10G    # 10 * 8 = 80G total system RAM
#SBATCH --gres=gpu:1         # request 1 GPU of any type
#SBATCH --constraint=40G|80G

echo "Allocated GPU: $SLURM_JOB_GPUS"

module load java
module load cuda
module load ffmpeg
module load use.own uv

# uv sync --all-extras
# uv pip install --force-reinstall --no-cache torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
source .venv/bin/activate


# ====================================================================
# LLM JUDGE BACKEND SELECTION
#   Set USE_HF_LLM_JUDGE=true to use a HuggingFace model as judge.
#   Default (unset or false): use Ollama with an OpenAI-compatible API.
# ====================================================================
USE_HF_LLM_JUDGE="${USE_HF_LLM_JUDGE:-false}"
EVAL_LLM_MODEL=Qwen/Qwen2.5-3B

if [ "${USE_HF_LLM_JUDGE}" = "true" ]; then
    # ---- HuggingFace judge ----
    export EVAL_LLM_MODEL="${EVAL_LLM_MODEL:-Qwen/Qwen3-8B}"
    # Unset OpenAI-compatible vars to avoid confusion
    unset EVAL_BASE_URL EVAL_API_KEY
    unset OPENAI_API_BASE OPENAI_API_KEY
    echo "LLM Judge backend: HuggingFace (${EVAL_LLM_MODEL})"
else
    # ---- Ollama judge (OpenAI-compatible API) ----
    export EVAL_BASE_URL="${EVAL_BASE_URL:-http://localhost:11434/v1/chat/completions}"
    export EVAL_MODEL="${EVAL_MODEL:-qwen3.5:9b}"
    export EVAL_API_KEY="${EVAL_API_KEY:-ollama}"
    export MODEL_GPT_4O_MINI="${EVAL_MODEL}"
    export MODEL_GPT_4O="${EVAL_MODEL}"
    unset EVAL_LLM_MODEL

    echo ""
    echo "======================================================"
    echo "=== OLLAMA DIAGNOSTICS ==="
    echo "======================================================"

    # Kill any lingering ollama from previous jobs
    pkill -f "ollama serve" 2>/dev/null || true

    echo "1. Starting ollama serve in background..."
    ollama serve > /dev/null 2>&1 &  
    OLLAMA_PID=$!
    echo "   Ollama PID: ${OLLAMA_PID}"

    # Wait for ollama to be ready
    echo "2. Waiting for ollama to be ready..."
    for i in $(seq 1 30); do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "   Ollama is ready after ${i}s"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "   ERROR: Ollama failed to start within 30s"
            kill ${OLLAMA_PID} 2>/dev/null
            exit 1
        fi
        sleep 1
    done

    echo "3. Ensuring model ${EVAL_MODEL} is available..."
    ollama pull ${EVAL_MODEL} 2>&1

    echo "======================================================"
    echo ""
fi

# ====================================================================
# EXPANDED DIAGNOSTIC PRINTS: System Driver vs. Python Package
# ====================================================================
echo ""
echo "======================================================"
echo "=== SYSTEM & CUDA HARDWARE DIAGNOSTICS ==="
echo "======================================================"

echo "1. Host Driver Status (NVIDIA-SMI):"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "  ERROR: nvidia-smi command not found on this node."
fi

echo ""
echo "2. System CUDA Compiler (NVCC Version):"
if command -v nvcc &> /dev/null; then
    nvcc --version
else
    echo "  NOTICE: System nvcc not found in PATH (using node default environment)."
fi

echo ""
echo "3. Shell Executable Resolution:"
echo "  Active torchrun path: $(which torchrun)"
echo "  Active python path:   $(which python)"

echo ""
echo "4. PyTorch & CUDA Library Diagnostics:"
python -c "
import torch
import sys
print('  Python Version:     ', sys.version.split()[0])
print('  Torch Version:      ', torch.__version__)
print('  Compiled with CUDA: ', torch.version.cuda)
print('  CUDA Available:     ', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  GPU Device Name:    ', torch.cuda.get_device_name(0))
    print('  Device Capability:  ', torch.cuda.get_device_capability(0))
else:
    print('  WARNING: PyTorch cannot initialize the allocated GPU!')
"
echo "======================================================"
echo ""

# Explicitly append the local nvidia dependencies to your library path
# export LD_LIBRARY_PATH=$HOME/OmniEvalKit/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$HOME/OmniEvalKit/.venv/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
# Ollama endpoint configuration for LLM-as-Judge



# Force HF to look strictly at local files and skip the gating check
export HF_HUB_OFFLINE="0"
MODEL_PATH="google/gemma-4-12B-it"         
PT_PATH=""
MODEL_TYPE="gemma4_omni"
MODEL_NAME="gemma4-12b-it"
ANSWER_PATH="./results"
GPUS_PER_NODE=1
BATCH_SIZE=4
MAX_SAMPLES=""
GENERATE_METHOD="generate"

# ===================== 评测数据集选择（取消注释你需要的场景） =====================

# --- 场景 1: ASR 语音识别 ---
EVAL_DATASETS="--eval_daily_omni --eval_futureomni --eval_jointavbench --eval_omnibench --eval_worldsense --eval_video_holmes --eval_videomme_short --eval_futureomni --eval_avut_benchmark_human"

OPT_ARGS=""
OPT_ARGS+=" --model_path ${MODEL_PATH}"
OPT_ARGS+=" --model_type ${MODEL_TYPE}"
OPT_ARGS+=" --model_name ${MODEL_NAME}"
OPT_ARGS+=" --answer_path ${ANSWER_PATH}"
OPT_ARGS+=" --generate_method ${GENERATE_METHOD}"
OPT_ARGS+=" --batchsize ${BATCH_SIZE}"

if [ -n "${PT_PATH}" ]; then
    OPT_ARGS+=" --pt_path ${PT_PATH}"
fi

if [ -n "${MAX_SAMPLES}" ]; then
    OPT_ARGS+=" --max_sample_num ${MAX_SAMPLES}"
fi

# 多GPU模型分片（如需跨卡分布模型，取消注释下一行）
OPT_ARGS+=" --attn_implementation sdpa"
# OPT_ARGS+=" --auto_device_map"

# 量化（如需降低显存占用，取消注释下一行）
# OPT_ARGS+=" --quantization 4bit"

OPT_ARGS+=" ${EVAL_DATASETS}"

MASTER_PORT=${MASTER_PORT:-29500}

# Point Python's ChatClient at the OpenAI-compatible endpoint (Ollama mode only)
if [ -n "${EVAL_BASE_URL+x}" ]; then
    export OPENAI_API_BASE="${EVAL_BASE_URL}"
fi
if [ -n "${EVAL_API_KEY+x}" ]; then
    export OPENAI_API_KEY="${EVAL_API_KEY}"
fi

CMD="torchrun --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} eval_main.py ${OPT_ARGS}"

echo "========================================"
echo "OmniEvalKit Evaluation"
echo "========================================"
echo "Model:    ${MODEL_PATH}"
echo "Type:     ${MODEL_TYPE}"
echo "GPUs:     ${GPUS_PER_NODE}"
echo "Batch:    ${BATCH_SIZE}"
echo "Output:   ${ANSWER_PATH}"
echo "Datasets: ${EVAL_DATASETS}"
if [ "${USE_HF_LLM_JUDGE}" = "true" ]; then
    echo "LLM-Judge: HuggingFace (${EVAL_LLM_MODEL})"
else
    echo "LLM-Judge: Ollama (${EVAL_MODEL} @ ${EVAL_BASE_URL})"
fi
echo "========================================"
echo ""
echo "CMD: ${CMD}"
echo ""

${CMD}

if [ "${USE_HF_LLM_JUDGE}" != "true" ] && [ -n "${OLLAMA_PID+x}" ]; then
    echo ""
    echo "Stopping ollama serve (PID ${OLLAMA_PID})..."
    kill ${OLLAMA_PID} 2>/dev/null || true
fi
