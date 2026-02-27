"""与 LoRA 模型交互式对话

用法：
    # 使用训练好的 LoRA（推荐）
    python scripts/chat_lora.py \
        --lora_checkpoint outputs/lora_20260226_1801_sw/best

    # 使用纯基座模型（对比用）
    python scripts/chat_lora.py

    # 自定义 system prompt（默认为红楼梦写作风格）
    python scripts/chat_lora.py --lora_checkpoint ... --system ""

命令：
    输入文字后回车 → 发送给模型
    /clear          → 清空对话历史（保留 system prompt）
    /history        → 显示当前对话历史
    /system <text>  → 临时修改 system prompt
    /exit 或 Ctrl+C → 退出
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent

LOCAL_FP8_PATH = (
    "/home/lulingjie/.cache/huggingface/hub"
    "/models--Qwen--Qwen3-8B-FP8/snapshots"
    "/220b46e3b2180893580a4454f21f22d3ebb187d3"
)

DEFAULT_SYSTEM = """你是一位专业的《红楼梦》续写作家，深谙曹雪芹的文风与叙事技法。

【写作要求】
1. 语言风格：文言夹白话的章回体小说文风，与曹雪芹原著前80回保持一致。多用"话说""却说""只见""正是"等章回体习语。
2. 人物性格：严格遵循原著中各人物已有的性格、语气、习惯，不得出现性格突变。
3. 叙事节奏：有场景描写、人物对话、心理刻画，不可单纯概述情节。
4. 格式要求：每个自然段开头用全角空格"　　"缩进两格。
5. 禁止：不得输出现代白话解释、注释、分析或任何非小说正文的内容。"""


# ─────────────────────────────────────────────────────────────
# 模型加载（复用 generate_all_instruct.py 的逻辑）
# ─────────────────────────────────────────────────────────────
def _dequantize_fp8(module, seen_ids: set) -> int:
    import torch.nn as nn
    try:
        from transformers.integrations import FP8Linear
    except ImportError:
        return 0
    replaced = 0
    for name in list(module._modules.keys()):
        child = module._modules[name]
        if child is None:
            continue
        if isinstance(child, FP8Linear) and id(child) not in seen_ids:
            seen_ids.add(id(child))
            dev = child.weight.device
            out_f, in_f = child.weight.shape
            w_cpu  = child.weight.data.to("cpu").float()
            si_cpu = child.weight_scale_inv.data.to("cpu").float()
            if child.block_size is None:
                w_bf16 = (w_cpu * si_cpu.item()).to(torch.bfloat16)
            else:
                bh, bw = child.block_size
                n_out = (out_f + bh - 1) // bh
                n_in  = (in_f  + bw - 1) // bw
                if n_out * bh != out_f or n_in * bw != in_f:
                    import torch.nn.functional as F
                    w_cpu = F.pad(w_cpu, (0, n_in * bw - in_f, 0, n_out * bh - out_f))
                w_cpu  = w_cpu.view(n_out, bh, n_in, bw)
                si_cpu = si_cpu.view(n_out, 1, n_in, 1)
                w_bf16 = (w_cpu * si_cpu).view(n_out * bh, n_in * bw)[:out_f, :in_f].to(torch.bfloat16)
            new_lin = nn.Linear(in_f, out_f, bias=child.bias is not None, dtype=torch.bfloat16)
            new_lin.weight.data = w_bf16
            if child.bias is not None:
                new_lin.bias.data = child.bias.data.to("cpu").bfloat16()
            del child, w_cpu, si_cpu, w_bf16
            setattr(module, name, new_lin.to(dev))
            torch.cuda.empty_cache()
            replaced += 1
        else:
            replaced += _dequantize_fp8(child, seen_ids)
    return replaced


def load_model(lora_checkpoint: str | None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if lora_checkpoint:
        from peft import PeftModel, PeftConfig
        import transformers.trainer as _t
        _t.validate_quantization_for_training = lambda m: None

        cfg = PeftConfig.from_pretrained(lora_checkpoint)
        base_id = LOCAL_FP8_PATH if Path(LOCAL_FP8_PATH).exists() else cfg.base_model_name_or_path

        print(f"\n加载 FP8 基座：{base_id}")
        model = AutoModelForCausalLM.from_pretrained(
            base_id, device_map="auto", torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("FP8→BF16 反量化...", end=" ", flush=True)
        n = _dequantize_fp8(model, set())
        print(f"{n} 层完成")

        print(f"挂载 LoRA：{lora_checkpoint}")
        model = PeftModel.from_pretrained(model, lora_checkpoint)
        model.eval()
        print("LoRA 模型加载完成\n")
    else:
        print(f"\n加载基座模型（无 LoRA）：{LOCAL_FP8_PATH}")
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_FP8_PATH, device_map="auto", torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_FP8_PATH, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        print("基座模型加载完成\n")

    return model, tokenizer


# ─────────────────────────────────────────────────────────────
# 生成
# ─────────────────────────────────────────────────────────────
def chat_generate(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = 800,
    temperature: float = 0.75,
    top_p: float = 0.92,
    repetition_penalty: float = 1.1,
) -> str:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = output_ids[0][input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────
# 交互循环
# ─────────────────────────────────────────────────────────────
def chat_loop(model, tokenizer, system_prompt: str, max_new_tokens: int):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    model_tag = "LoRA+Qwen3" if hasattr(model, "peft_config") else "Qwen3-基座"
    print(f"{'='*60}")
    print(f"  {model_tag} 交互对话")
    print(f"  /clear 清空历史  /history 显示历史  /exit 退出")
    print(f"{'='*60}\n")

    while True:
        try:
            user_input = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not user_input:
            continue

        # 内置命令
        if user_input == "/exit":
            print("退出。")
            break
        elif user_input == "/clear":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            print("[对话历史已清空]\n")
            continue
        elif user_input == "/history":
            for i, m in enumerate(messages):
                role = "系统" if m["role"] == "system" else ("你" if m["role"] == "user" else "模型")
                preview = m["content"][:80].replace("\n", " ")
                print(f"  [{i}] {role}: {preview}{'...' if len(m['content']) > 80 else ''}")
            print()
            continue
        elif user_input.startswith("/system "):
            system_prompt = user_input[8:].strip()
            messages = [m for m in messages if m["role"] != "system"]
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})
            print(f"[system prompt 已更新]\n")
            continue

        messages.append({"role": "user", "content": user_input})

        print("模型：", end="", flush=True)
        response = chat_generate(model, tokenizer, messages, max_new_tokens=max_new_tokens)
        print(response)
        print()

        messages.append({"role": "assistant", "content": response})


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="与 LoRA 模型交互式对话")
    parser.add_argument("--lora_checkpoint", default=None,
                        help="LoRA checkpoint 路径（如 outputs/lora_20260226_1801_sw/best）；省略则使用纯基座")
    parser.add_argument("--system", default=DEFAULT_SYSTEM,
                        help="System prompt（留空则不设置）")
    parser.add_argument("--max_new_tokens", type=int, default=800,
                        help="每次回复最大 token 数（默认 800）")
    args = parser.parse_args()

    model, tokenizer = load_model(args.lora_checkpoint)
    chat_loop(model, tokenizer, args.system, args.max_new_tokens)


if __name__ == "__main__":
    main()
