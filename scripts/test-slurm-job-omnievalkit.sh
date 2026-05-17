#!/bin/bash
#SBATCH -J test-omnievalkit      # job name
#SBATCH -o %x.o%j            # single STDOUT/STDERR output file jobname.o<job number>
#SBATCH -p gpushort          # request gpushort partition
#SBATCH -n 8                 # 8 cores
#SBATCH --cpus-per-gpu=8     # 8 cores per GPU
#SBATCH -t 1:0:0             # 1 hour runtime (required to run on the short partition)
#SBATCH --mem-per-cpu=10G    # 10 * 8 = 80G total system RAM
#SBATCH --gres=gpu:1         # request 1 GPU of any type
#SBATCH --constraint=40G|80G

echo "Allocated GPU: $SLURM_JOB_GPUS"

module load python
module load ffmpeg
module load use.own uv

uv sync --all-extras
uv pip install --force-reinstall --no-cache torch torchaudio --index-url https://download.pytorch.org/whl/cu126
source .venv/bin/activate

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
export LD_LIBRARY_PATH=$HOME/OmniEvalKit/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/OmniEvalKit/.venv/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

MODEL_PATH="/gpfs/scratch/qp252970/models/MiniCPM-o-4_5"         
PT_PATH=""
MODEL_TYPE="minicpmo"
MODEL_NAME="my_eval"
ANSWER_PATH="./results"
GPUS_PER_NODE=1
BATCH_SIZE=1
MAX_SAMPLES="100"
GENERATE_METHOD="chat"

# ===================== 评测数据集选择（取消注释你需要的场景） =====================

# --- 场景 1: ASR 语音识别 ---
EVAL_DATASETS="--eval_omnibench"

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
# OPT_ARGS+=" --auto_device_map --attn_implementation flash_attention_2"

# 量化（如需降低显存占用，取消注释下一行）
# OPT_ARGS+=" --quantization 4bit"

OPT_ARGS+=" ${EVAL_DATASETS}"

MASTER_PORT=${MASTER_PORT:-29500}

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
echo "========================================"
echo ""
echo "CMD: ${CMD}"
echo ""

${CMD}
