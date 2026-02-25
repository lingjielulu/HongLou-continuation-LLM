"""
绘制训练指标图表。
用法：conda run -n stone python3 scripts/plot_metrics.py
输出：outputs/training_metrics.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ── 从 TensorBoard 读取训练数据 ───────────────────────────────
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    run_dirs = sorted((ROOT / 'outputs' / 'runs').iterdir())
    ea = EventAccumulator(str(run_dirs[-1]))
    ea.Reload()
    tb_loss    = [(e.step, e.value) for e in ea.Scalars('train/loss')]
    tb_gnorm   = [(e.step, e.value) for e in ea.Scalars('train/grad_norm')]
    tb_lr      = [(e.step, e.value) for e in ea.Scalars('train/learning_rate')]
    train_steps = [x[0] for x in tb_loss]
    train_loss  = [x[1] for x in tb_loss]
    grad_norm   = [x[1] for x in tb_gnorm]
    lr_steps    = [x[0] for x in tb_lr]
    lr_vals     = [x[1] for x in tb_lr]
except Exception as e:
    print(f"TensorBoard read failed ({e}), using hardcoded data")
    train_steps = [10, 20, 30, 40, 50, 60, 70]
    train_loss  = [2.6852, 2.2574, 2.1216, 2.0438, 2.0087, 1.889, 1.880]
    grad_norm   = [0.0123, 0.0098, 0.0088, 0.0097, 0.0107, 0.0103, 0.0098]
    lr_steps    = train_steps
    lr_vals     = [0.000198, 0.000179, 0.000145, 0.000102, 0.0000592, 0.0000240, 0.0000035]

# ── 从 eval reports 读取 PPL / n-gram ────────────────────────
reports_dir = ROOT / 'outputs' / 'eval_reports'
eval_data = []
for f in sorted(reports_dir.glob('step_*.json')):
    d = json.loads(f.read_text(encoding='utf-8'))
    eval_data.append(d)
eval_data.sort(key=lambda x: x['step'])

eval_steps = [d['step'] for d in eval_data]
ppl        = [d['ppl']  for d in eval_data]
gram2      = [d.get('ngram', {}).get('2gram', None) for d in eval_data]
gram3      = [d.get('ngram', {}).get('3gram', None) for d in eval_data]
gram4      = [d.get('ngram', {}).get('4gram', None) for d in eval_data]

# ── 绘图 ──────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.suptitle(
    'Honglou LoRA Training Metrics  (Qwen3-8B FP8->BF16 + LoRA r=16, alpha=32)',
    fontsize=12, fontweight='bold', y=0.99
)
gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.32)

# ── 1. Train Loss ─────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(train_steps, train_loss, 'o-', color='#2196F3', lw=2, ms=6, label='Train Loss')
ax1.set_xlabel('Optimizer Step')
ax1.set_ylabel('Cross-Entropy Loss')
ax1.set_title('Training Loss (train/loss)')
ax1.grid(True, alpha=0.3)
for x, y in zip(train_steps, train_loss):
    ax1.annotate(f'{y:.3f}', (x, y), textcoords='offset points',
                 xytext=(0, 7), fontsize=7.5, ha='center')
# epoch 分隔线
for ep_step in [25, 50, 75]:
    ax1.axvline(ep_step, color='gray', ls='--', alpha=0.35, lw=1)
    ax1.text(ep_step + 0.5, max(train_loss) * 0.99,
             f'epoch {ep_step//25}', fontsize=7, color='gray', va='top')

# ── 2. Eval PPL ───────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
pt_colors = ['#FF5722'] + ['#4CAF50'] * (len(eval_steps) - 1)
for s, p, c in zip(eval_steps, ppl, pt_colors):
    ax2.scatter(s, p, color=c, s=90, zorder=5)
    ax2.annotate(f'{p:.2f}', (s, p), textcoords='offset points',
                 xytext=(0, 9), fontsize=8.5, ha='center')
ax2.plot(eval_steps, ppl, '--', color='gray', lw=1, alpha=0.4)
ax2.axhline(ppl[0], color='#FF5722', ls=':', alpha=0.45,
            label=f'Baseline (no finetune) PPL={ppl[0]:.1f}')
ax2.set_xlabel('Step')
ax2.set_ylabel('Perplexity (PPL)')
ax2.set_title('Validation Perplexity  (lower = better)')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# ── 3. N-gram StyleOverlap ────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
g2_clean = [v for v in gram2 if v is not None]
g3_clean = [v for v in gram3 if v is not None]
g4_clean = [v for v in gram4 if v is not None]
s_clean  = [s for s, v in zip(eval_steps, gram2) if v is not None]

ax3.plot(s_clean, g2_clean, 's-', color='#9C27B0', lw=2, ms=6, label='2-gram')
ax3.plot(s_clean, g3_clean, '^-', color='#FF9800', lw=2, ms=6, label='3-gram')
ax3.plot(s_clean, g4_clean, 'D-', color='#009688', lw=2, ms=6, label='4-gram')
ax3.set_xlabel('Step')
ax3.set_ylabel('StyleOverlap')
ax3.set_title('N-gram Style Overlap  (generated text vs training corpus)')
ax3.legend(fontsize=9)
ax3.set_ylim(0, 1.05)
ax3.grid(True, alpha=0.3)
for vals, color in [(g2_clean, '#9C27B0'), (g3_clean, '#FF9800'), (g4_clean, '#009688')]:
    for s, v in zip(s_clean, vals):
        ax3.annotate(f'{v:.3f}', (s, v), textcoords='offset points',
                     xytext=(0, 6), fontsize=6.5, ha='center', color=color)

# ── 4. Grad Norm + LR ─────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
cg, cl = '#F44336', '#607D8B'
l1, = ax4.plot(train_steps, grad_norm, 'o-', color=cg, lw=2, ms=5, label='Grad Norm')
ax4.set_xlabel('Step')
ax4.set_ylabel('Gradient L2 Norm', color=cg)
ax4.tick_params(axis='y', labelcolor=cg)
ax4r = ax4.twinx()
l2, = ax4r.plot(lr_steps, lr_vals, 's--', color=cl, lw=1.5, ms=4, label='LR (cosine)')
ax4r.set_ylabel('Learning Rate', color=cl)
ax4r.tick_params(axis='y', labelcolor=cl)
ax4r.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax4.set_title('Gradient Norm & LR Schedule (cosine warmup)')
ax4.legend(handles=[l1, l2], loc='upper right', fontsize=8)
ax4.grid(True, alpha=0.3)

# ── 保存 ──────────────────────────────────────────────────────
out = ROOT / 'outputs' / 'training_metrics.png'
plt.savefig(str(out), dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved: {out}')
