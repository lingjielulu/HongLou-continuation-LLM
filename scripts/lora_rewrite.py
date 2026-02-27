"""LoRA 风格重写脚本

工作流：
    instruct 版（情节连贯）→ 按段提取开头 seed → LoRA 续写（文言风格）→ 拼接保存

用法：
    python scripts/lora_rewrite.py \\
        --input  outputs/chapter_081_instruct_scenes.txt \\
        --checkpoint outputs/checkpoint-best \\
        --output outputs/chapter_081_lora_rewrite.txt

参数说明：
    --seed_chars    每段取多少字作为 LoRA 的 prompt 种子（默认 45）
    --max_new_tokens 每段 LoRA 最大生成 token 数（默认 350）
"""

import argparse
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent

LOCAL_FP8_PATH = (
    "/home/lulingjie/.cache/huggingface/hub"
    "/models--Qwen--Qwen3-8B-FP8/snapshots"
    "/220b46e3b2180893580a4454f21f22d3ebb187d3"
)

# ─────────────────────────────────────────────────────────────
# LoRA 模型加载（与 generate.py 相同逻辑）
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
                    w_cpu = nn.functional.pad(
                        w_cpu, (0, n_in * bw - in_f, 0, n_out * bh - out_f))
                w_cpu  = w_cpu.view(n_out, bh, n_in, bw)
                si_cpu = si_cpu.view(n_out, 1, n_in, 1)
                w_bf16 = (w_cpu * si_cpu).view(
                    n_out * bh, n_in * bw)[:out_f, :in_f].to(torch.bfloat16)
            new_lin = nn.Linear(in_f, out_f, bias=child.bias is not None,
                                dtype=torch.bfloat16)
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


def load_lora_model(checkpoint_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, PeftConfig

    peft_cfg = PeftConfig.from_pretrained(checkpoint_path)
    base_id = (LOCAL_FP8_PATH if Path(LOCAL_FP8_PATH).exists()
               else peft_cfg.base_model_name_or_path)

    print(f"加载 FP8 基座：{base_id}")
    base = AutoModelForCausalLM.from_pretrained(
        base_id, device_map="auto", torch_dtype="auto", trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    n = _replace_fp8_recursive(base, set())
    print(f"[FP8→BF16] {n} 层")
    model = PeftModel.from_pretrained(base, checkpoint_path)
    model = model.merge_and_unload().to(torch.bfloat16)
    print("[Merge] 完成")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return model, tok


# ─────────────────────────────────────────────────────────────
# 单段生成
# ─────────────────────────────────────────────────────────────
GEN_CONFIG = dict(
    temperature=0.82,
    top_p=0.92,
    top_k=50,
    repetition_penalty=1.15,
    do_sample=True,
)

# 非叙事内容信号
_NON_NARRATIVE_RE = re.compile(
    r"^(>|#{1,4}\s|注释[：:]|章节简析|赏析|第\d+章\s)", re.MULTILINE)

# 章回结束套语
_CHAPTER_END_RE = re.compile(r"(欲知后事|且听下回|下回分解|未知后事|下回再表)")


def generate_para(model, tok, seed: str, max_new_tokens: int = 350) -> str:
    """以 seed 为 prompt，生成一段续写"""
    inputs = tok(seed, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.eos_token_id,
            **GEN_CONFIG,
        )

    gen = tok.decode(out[0][input_len:], skip_special_tokens=True)

    # 截断非叙事内容
    m = _NON_NARRATIVE_RE.search(gen)
    if m:
        gen = gen[:m.start()].rstrip()

    # 截断章回结束套语（保留，但标记）
    me = _CHAPTER_END_RE.search(gen)
    if me:
        gen = gen[:me.start()].rstrip()

    return gen


# ─────────────────────────────────────────────────────────────
# 段落分割
# ─────────────────────────────────────────────────────────────
# 清理 instruct 输出中残留的结尾套语变体
_CLEANUP_RE = re.compile(
    r"(正是[：:][^\n]*\n)?[　 ]*(正是[：：][^\n。]*[。]?\n)?"
    r"[　 ]*[欲未]知后事[，,].*?[解。\n]+"
    r"|下回再表[。]?"
    r"|且看下文[。]?"
)


def split_paragraphs(text: str) -> list[str]:
    """将 instruct 输出拆成段落列表，同时清理残留套语"""
    # 先清理残留结尾套语
    text = _CLEANUP_RE.sub("", text).strip()
    # 按空行拆段
    raw = re.split(r"\n{2,}", text)
    paras = []
    for p in raw:
        p = p.strip()
        if not p:
            continue
        # 一个"段落"块内若有多行（同场景内连续行），逐行处理
        lines = [l.strip() for l in p.splitlines() if l.strip()]
        # 将同一块内的相邻行合并成一个段落（保留整段作为一个处理单元）
        paras.append("\n".join(lines))
    return paras


def extract_seed(para: str, seed_chars: int) -> str:
    """从段落开头提取 seed：取到第 seed_chars 字或第一个句号（取较短者）"""
    # 去掉段落缩进空格
    text = para.lstrip("　 ")
    # 找第一个完整句子的结尾（不超过 seed_chars 字）
    m = re.search(r"[，。！？；]", text[:seed_chars + 20])
    if m and m.end() <= seed_chars + 10:
        return text[:m.end()]
    return text[:seed_chars]


# ─────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────
def rewrite(model, tok, instruct_text: str,
            seed_chars: int = 45, max_new_tokens: int = 350) -> str:
    paras = split_paragraphs(instruct_text)
    print(f"\n共 {len(paras)} 个段落需要重写")

    rewritten = []
    for i, para in enumerate(paras):
        seed = extract_seed(para, seed_chars)
        print(f"\n  [段落 {i+1}/{len(paras)}]")
        print(f"  原文开头：{para[:40]}…")
        print(f"  Seed ({len(seed)}字)：{seed}")

        new_para = generate_para(model, tok, seed, max_new_tokens)
        full_para = seed + new_para

        print(f"  重写结果 ({len(full_para)}字)：{full_para[:50]}…")
        rewritten.append("\n　　" + full_para.lstrip("　 "))

    # 拼接，章回标题从第一段提取
    result = "\n".join(rewritten)

    # 确保只有末尾一个章回结尾套语
    result = result.rstrip()
    if not _CHAPTER_END_RE.search(result[-60:]):
        result += "\n　　欲知后事如何，且听下回分解。"

    return result


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LoRA 风格重写")
    parser.add_argument("--input",       required=True, help="instruct 输出文件路径")
    parser.add_argument("--checkpoint",  required=True, help="LoRA checkpoint 路径")
    parser.add_argument("--output",      default=None,  help="输出文件路径")
    parser.add_argument("--seed_chars",  type=int, default=45,
                        help="每段取前 N 字作为 seed（默认 45）")
    parser.add_argument("--max_new_tokens", type=int, default=350,
                        help="每段最大生成 token 数（默认 350）")
    parser.add_argument("--dry_run", action="store_true",
                        help="只显示段落分割和 seed，不生成")
    args = parser.parse_args()

    instruct_text = Path(args.input).read_text(encoding="utf-8")

    if args.dry_run:
        paras = split_paragraphs(instruct_text)
        print(f"共 {len(paras)} 个段落：")
        for i, p in enumerate(paras):
            seed = extract_seed(p, args.seed_chars)
            print(f"\n[{i+1}] seed: {repr(seed)}")
            print(f"     原文: {p[:60]}…")
        return

    model, tok = load_lora_model(args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    result = rewrite(model, tok, instruct_text,
                     seed_chars=args.seed_chars,
                     max_new_tokens=args.max_new_tokens)

    print(f"\n{'='*60}")
    print(f"重写完成，共 {len(result)} 字")
    print("=" * 60)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result, encoding="utf-8")
        print(f"已保存至：{out}")
    else:
        print("\n" + result)


if __name__ == "__main__":
    main()
