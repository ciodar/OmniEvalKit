#!/bin/bash
# Omni-DuplexEval 流式推理示例脚本
# 运行 MiniCPM-o-4.5 双工推理，导出响应供下游评估使用
#
# 用法:
#   bash scripts/eval_omniduplexeval.sh [model_path] [splits]

MODEL_PATH="${1:-/path/to/MiniCPM-o-4_5}"
SPLITS="${2:-all}"

torchrun --nproc_per_node=1 eval_main.py \
  --model_type minicpmo_duplex_demo \
  --model_path "$MODEL_PATH" \
  --model_name omniduplexeval \
  --eval_omniduplexeval \
  --omniduplexeval_splits $SPLITS \
  --omniduplexeval_response_root ./results/omniduplexeval_responses \
  --omniduplexeval_fps 1 \
  --batchsize 1 \
  --answer_path ./results

echo "Done. Responses saved to ./results/omniduplexeval_responses/"
echo "Run upstream evaluation separately:"
echo "  python scripts/evaluate_omniduplexeval.py --response_root ./results/omniduplexeval_responses"
