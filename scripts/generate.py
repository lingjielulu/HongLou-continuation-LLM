"""
生成推理脚本
文档参考：README.md §8

用法：
    # 单次生成
    python scripts/generate.py \
        --checkpoint outputs/checkpoint-best \
        --prompt "话说那日贾宝玉在怡红院中闲坐，忽见"

    # 交互模式
    python scripts/generate.py \
        --checkpoint outputs/checkpoint-best \
        --interactive

    # 批量生成（从文件读取 prompt）
    python scripts/generate.py \
        --checkpoint outputs/checkpoint-best \
        --prompts_file configs/eval_prompts.txt \
        --output outputs/generated.txt
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).parent.parent


# ─────────────────────────────────────────────────────────────
# FP8 → BF16 反量化（与 train.py 相同逻辑）
# ─────────────────────────────────────────────────────────────
def _replace_fp8_recursive(module, seen_ids):
    from transformers.integrations import FP8Linear
    replaced = 0
    for child_name in list(module._modules.keys()):
        child = module._modules[child_name]
        if child is None:
            continue
        if isinstance(child, FP8Linear) and id(child) not in seen_ids:
            seen_ids.add(id(child))
            dev = child.weight.device
            out_f, in_f = child.weight.shape
            w_cpu  = child.weight.data.to("cpu").to(torch.float32)
            si_cpu = child.weight_scale_inv.data.to("cpu").float()
            if child.block_size is None:
                w_bf16 = (w_cpu * si_cpu.item()).to(torch.bfloat16)
            else:
                bh, bw = child.block_size
                n_out = (out_f + bh - 1) // bh
                n_in  = (in_f  + bw - 1) // bw
                if n_out * bh != out_f or n_in * bw != in_f:
                    w_cpu = nn.functional.pad(w_cpu, (0, n_in * bw - in_f, 0, n_out * bh - out_f))
                w_cpu  = w_cpu.view(n_out, bh, n_in, bw)
                si_cpu = si_cpu.view(n_out, 1, n_in, 1)
                w_bf16 = (w_cpu * si_cpu).view(n_out * bh, n_in * bw)[:out_f, :in_f].to(torch.bfloat16)
            new_lin = nn.Linear(in_f, out_f, bias=child.bias is not None, dtype=torch.bfloat16)
            new_lin.weight.data = w_bf16
            if child.bias is not None:
                new_lin.bias.data = child.bias.data.to("cpu").to(torch.bfloat16)
            del child, w_cpu, si_cpu, w_bf16
            setattr(module, child_name, new_lin)
            new_lin = new_lin.to(dev)
            setattr(module, child_name, new_lin)
            torch.cuda.empty_cache()
            replaced += 1
        else:
            replaced += _replace_fp8_recursive(child, seen_ids)
    return replaced
LOCAL_FP8_PATH = (
    "/home/lulingjie/.cache/huggingface/hub"
    "/models--Qwen--Qwen3-8B-FP8/snapshots"
    "/220b46e3b2180893580a4454f21f22d3ebb187d3"
)

# ─────────────────────────────────────────────────────────────
# 默认生成参数（README §8.2）
# ─────────────────────────────────────────────────────────────
DEFAULT_GEN_CONFIG = {
    "max_new_tokens":    512,
    "temperature":       0.8,
    "top_p":             0.9,
    "top_k":             50,
    "repetition_penalty": 1.1,
    "do_sample":         True,
}


# ─────────────────────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, merged: bool = False):
    """
    加载模型。

    merged=True  时加载已合并的完整模型（无需 PEFT）
    merged=False 时加载 LoRA 适配器 + 量化基座
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel, PeftConfig

    if merged:
        print(f"加载合并模型：{checkpoint_path}")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    else:
        peft_cfg = PeftConfig.from_pretrained(checkpoint_path)
        # 优先使用本地 FP8 缓存路径
        base_model_id = (
            LOCAL_FP8_PATH
            if Path(LOCAL_FP8_PATH).exists()
            else peft_cfg.base_model_name_or_path
        )

        print(f"加载 FP8 基座模型：{base_model_id}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="auto",
            torch_dtype="auto",     # 自动识别 FP8 权重
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        n = _replace_fp8_recursive(base_model, set())
        print(f"[FP8→BF16] 已将 {n} 个 FP8Linear 反量化为 BF16 Linear")
        print(f"加载 LoRA 权重：{checkpoint_path}")
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        # 将 LoRA 合并进 BF16 基座，消除 PEFT 包装层和类型不一致问题
        print("[Merge] 合并 LoRA 到基座模型...")
        model = model.merge_and_unload()
        model = model.to(torch.bfloat16)
        print("[Merge] 完成，模型为纯 BF16")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────────────────────
# 生成函数
# ─────────────────────────────────────────────────────────────
def generate(
    model,
    tokenizer,
    prompt: str,
    device: str = "cuda",
    gen_config: dict | None = None,
) -> str:
    """给定 prompt，生成续写文本"""
    cfg = {**DEFAULT_GEN_CONFIG, **(gen_config or {})}

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            pad_token_id=tokenizer.eos_token_id,
            **cfg,
        )

    gen_ids  = output_ids[0][input_len:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return gen_text


# ─────────────────────────────────────────────────────────────
# 交互模式
# ─────────────────────────────────────────────────────────────
def interactive_mode(model, tokenizer, device: str):
    print("\n=== 红楼梦文风生成（交互模式）===")
    print("输入 prompt 后回车生成，输入 'quit' 退出，输入 'config' 查看生成参数")
    print("-" * 60)

    gen_config = dict(DEFAULT_GEN_CONFIG)

    while True:
        try:
            prompt = input("\nPrompt> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n退出")
            break

        if not prompt:
            continue
        if prompt.lower() == "quit":
            break
        if prompt.lower() == "config":
            print("当前生成参数：")
            for k, v in gen_config.items():
                print(f"  {k}: {v}")
            continue

        print("\n[生成中...]\n")
        result = generate(model, tokenizer, prompt, device, gen_config)
        print("=" * 60)
        print(prompt + result)
        print("=" * 60)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="红楼梦文风生成")
    parser.add_argument("--checkpoint", required=True, help="LoRA checkpoint 或合并模型路径")
    parser.add_argument("--merged", action="store_true", help="使用合并后的完整模型")
    parser.add_argument("--prompt", default=None, help="单条 prompt")
    parser.add_argument("--prompts_file", default=None, help="批量 prompt 文件路径")
    parser.add_argument("--output",       default=None, help="批量生成输出文件")
    parser.add_argument("--interactive",  action="store_true", help="交互模式")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature",    type=float, default=0.8)
    parser.add_argument("--top_p",          type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(args.checkpoint, merged=args.merged)

    gen_config = {
        "max_new_tokens":    args.max_new_tokens,
        "temperature":       args.temperature,
        "top_p":             args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "do_sample":         True,
    }

    if args.interactive:
        interactive_mode(model, tokenizer, device)

    elif args.prompt:
        print(f"\nPrompt：{args.prompt}\n")
        result = generate(model, tokenizer, args.prompt, device, gen_config)
        print("=" * 60)
        print(args.prompt + result)
        print("=" * 60)

    elif args.prompts_file:
        prompts_path = Path(args.prompts_file)
        prompts = [
            line.strip()
            for line in prompts_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        print(f"批量生成 {len(prompts)} 条 prompt...")

        out_lines = []
        for i, prompt in enumerate(prompts):
            print(f"  [{i+1}/{len(prompts)}] {prompt[:30]}...")
            result = generate(model, tokenizer, prompt, device, gen_config)
            out_lines.append(f"=== Prompt {i+1} ===\n{prompt}\n\n=== Generated ===\n{result}\n")

        output_text = "\n".join(out_lines)
        if args.output:
            Path(args.output).write_text(output_text, encoding="utf-8")
            print(f"结果保存至：{args.output}")
        else:
            print(output_text)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
