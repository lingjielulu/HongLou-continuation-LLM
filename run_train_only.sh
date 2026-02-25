#!/bin/bash
# 仅运行训练（跳过基线评估）
# 用法：bash run_train_only.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p outputs/eval_reports outputs/logs

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True conda run -n stone python3 scripts/train.py --config configs/training_config.yaml 2>&1 | tee outputs/train.log
