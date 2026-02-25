"""
合并 LoRA 权重到基座模型（部署用）
文档参考：README.md §8.4

用法：
    python scripts/merge_lora.py \
        --lora_weights outputs/checkpoint-best \
        --output_dir models/honglou_merged

    # 也可手动指定基座模型路径
    python scripts/merge_lora.py \
        --base_model /path/to/Qwen3-8B-FP8 \
        --lora_weights outputs/checkpoint-best \
        --output_dir models/honglou_merged
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

LOCAL_FP8_PATH = (
    "/home/lulingjie/.cache/huggingface/hub"
    "/models--Qwen--Qwen3-8B-FP8/snapshots"
    "/220b46e3b2180893580a4454f21f22d3ebb187d3"
)


def merge_lora(lora_path: str, output_dir: str, base_model_id: str | None = None):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 解析基座模型路径
    if base_model_id is None:
        base_model_id = (
            LOCAL_FP8_PATH if Path(LOCAL_FP8_PATH).exists()
            else "Qwen/Qwen3-8B-FP8"
        )

    print(f"加载 FP8 基座模型：{base_model_id}")
    # 合并时用 torch_dtype="auto" 保留 FP8 格式，或用 bfloat16 转换为全精度
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    print(f"挂载 LoRA 权重：{lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)

    print("合并权重...")
    model = model.merge_and_unload()

    print(f"保存合并模型：{output_path}")
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    print(f"\n完成。合并模型保存至：{output_path}")
    print("使用方法：")
    print(f"  from transformers import AutoModelForCausalLM")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{output_path}')")


def main():
    parser = argparse.ArgumentParser(description="合并 LoRA 权重")
    parser.add_argument("--base_model",   default=None, help="基座模型路径（默认使用本地 FP8 缓存）")
    parser.add_argument("--lora_weights", required=True, help="LoRA checkpoint 路径")
    parser.add_argument("--output_dir",   required=True, help="合并模型输出路径")
    args = parser.parse_args()

    merge_lora(args.lora_weights, args.output_dir, args.base_model)


if __name__ == "__main__":
    main()
