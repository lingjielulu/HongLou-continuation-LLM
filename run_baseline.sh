#!/bin/bash
# Legacy：旧 LoRA 基线评估 + 训练启动脚本
# 当前主线请使用 scripts/prompt_baseline_generate.py。
# 用法：bash run_baseline.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p outputs/eval_reports outputs/logs

echo "=== [1/2] 基线评估 ==="
conda run -n stone python3 scripts/baseline_eval.py 2>&1 | tee outputs/baseline_eval.log

echo ""
echo "=== [2/2] 开始 QLoRA 训练 ==="
conda run -n stone python3 scripts/train.py --config configs/training_config.yaml 2>&1 | tee outputs/train.log
