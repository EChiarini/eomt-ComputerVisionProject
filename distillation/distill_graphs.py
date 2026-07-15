# Figures for the knowledge-distillation part of the project (English labels).
#
# Produces 6 figures in docs/figures/:
#   1. validation mIoU trajectory (baseline vs distilled, + teacher reference)
#   2. final mIoU comparison (teacher / baseline / distilled)
#   3. efficiency (parameters and speed: teacher vs student)
#   4. accuracy-vs-speed trade-off scatter (paper Fig.1 style)
#   5. per-class IoU (teacher / baseline / distilled) — needs inference on val
#   6. confusion matrices (baseline vs distilled) — needs inference on val
#
# Figures 1-4 use already-known numbers (from logs and measurements): instant, no GPU.
# Figures 5-6 run inference with the 3 checkpoints over the validation set (a few min).
#
# Usage: uv run --no-project python -m distillation.distill_graphs
#        (set MPLBACKEND=Agg to avoid GUI-backend issues)

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend: no windows, save to file only
import matplotlib.pyplot as plt

OUT_DIR = Path("results/distillation")

# --- consistent palette across all figures ---
C_TEACHER = "#4C4C4C"   # dark grey
C_BASELINE = "#E8734A"  # orange
C_DISTILL = "#3E8E7E"   # teal

# ═══════════════════════════════════════════════════════════════════════════
# KNOWN DATA (exact, from training logs and efficiency measurements)
# ═══════════════════════════════════════════════════════════════════════════
VAL_EPOCHS = [4, 8, 12, 16, 20]
MIOU_BASELINE = [47.1, 50.0, 50.5, 52.0, 51.4]
MIOU_DISTILL = [50.5, 50.6, 51.8, 51.9, 52.7]
MIOU_TEACHER = 56.1  # reference (Russo's fine-tuning)

FINAL = {"Teacher\n(EoMT-S)": 56.1, "Baseline\n(no KD)": 51.4, "Distilled\n(KD)": 52.7}
PARAMS_M = {"Teacher (EoMT-S)": 23.93, "Student (EoMT-Tiny)": 6.23}
SPEED_IMGS = {"Teacher (EoMT-S)": 28.52, "Student (EoMT-Tiny)": 61.97}

# (speed img/s, mIoU, params M) for the trade-off scatter
TRADEOFF = {
    "Teacher (EoMT-S)": (28.52, 56.1, 23.93),
    "Baseline (no KD)": (61.97, 51.4, 6.23),
    "Distilled (KD)": (61.97, 52.7, 6.23),
}

# figure 5-6 (inference)
NUM_CLASSES = 7
IMG_SIZE = (640, 640)
TEACHER_CKPT = "checkpoints/loveda_ft_eomt_small_640_best.bin"
BASELINE_CKPT = "eomt/413yl1ap/checkpoints/epoch=19-step=12520.ckpt"
DISTILL_CKPT = "eomt/cju4qzzo/checkpoints/best-19.ckpt"


# ───────────────────────────────────────────────────────────────────────────
# FIGURE 1 — validation mIoU trajectory
# ───────────────────────────────────────────────────────────────────────────
def fig_trajectory():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(MIOU_TEACHER, color=C_TEACHER, ls="--", lw=1.5,
               label=f"Teacher EoMT-S ({MIOU_TEACHER})")
    ax.plot(VAL_EPOCHS, MIOU_BASELINE, "o-", color=C_BASELINE, lw=2, ms=7,
            label="Student baseline (no KD)")
    ax.plot(VAL_EPOCHS, MIOU_DISTILL, "s-", color=C_DISTILL, lw=2, ms=7,
            label="Student distilled (KD)")
    ax.annotate(f"{MIOU_BASELINE[-1]}", (VAL_EPOCHS[-1], MIOU_BASELINE[-1]),
                textcoords="offset points", xytext=(6, -14), color=C_BASELINE, fontweight="bold")
    ax.annotate(f"{MIOU_DISTILL[-1]}", (VAL_EPOCHS[-1], MIOU_DISTILL[-1]),
                textcoords="offset points", xytext=(6, 6), color=C_DISTILL, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mIoU (LoveDA val)")
    ax.set_title("Validation trajectory: distillation stays above the baseline")
    ax.set_xticks(VAL_EPOCHS)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "distill_miou_trajectory.png")


# ───────────────────────────────────────────────────────────────────────────
# FIGURE 2 — final mIoU comparison
# ───────────────────────────────────────────────────────────────────────────
def fig_final_bars():
    fig, ax = plt.subplots(figsize=(6, 4.5))
    names = list(FINAL.keys())
    vals = list(FINAL.values())
    colors = [C_TEACHER, C_BASELINE, C_DISTILL]
    bars = ax.bar(names, vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v}",
                ha="center", va="bottom", fontweight="bold")
    ax.annotate("", xy=(2, 52.7), xytext=(1, 51.4),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.3))
    ax.text(1.5, 53.4, "+1.3", ha="center", fontweight="bold")
    ax.set_ylabel("mIoU (LoveDA val)")
    ax.set_title("Final mIoU: teacher vs students")
    ax.set_ylim(45, 58)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "distill_final_miou.png")


# ───────────────────────────────────────────────────────────────────────────
# FIGURE 3 — efficiency (parameters + speed)
# ───────────────────────────────────────────────────────────────────────────
def fig_efficiency():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    names = ["Teacher\n(EoMT-S)", "Student\n(EoMT-Tiny)"]
    colors = [C_TEACHER, C_DISTILL]

    p = list(PARAMS_M.values())
    b1 = ax1.bar(names, p, color=colors, width=0.55)
    for bar, v in zip(b1, p):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.3, f"{v}M", ha="center", fontweight="bold")
    ax1.set_ylabel("Parameters (millions)")
    ax1.set_title(f"Parameters  ({p[0]/p[1]:.2f}x smaller)")
    ax1.grid(True, axis="y", alpha=0.3)

    s = list(SPEED_IMGS.values())
    b2 = ax2.bar(names, s, color=colors, width=0.55)
    for bar, v in zip(b2, s):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.7, f"{v}", ha="center", fontweight="bold")
    ax2.set_ylabel("Images / second (RTX 3060)")
    ax2.set_title(f"Inference speed  ({s[1]/s[0]:.2f}x faster)")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Efficiency: distilled student vs teacher", y=1.02, fontweight="bold")
    fig.tight_layout()
    _save(fig, "distill_efficiency.png")


# ───────────────────────────────────────────────────────────────────────────
# FIGURE 4 — accuracy vs speed trade-off (paper Fig.1 style)
# ───────────────────────────────────────────────────────────────────────────
def fig_tradeoff():
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"Teacher (EoMT-S)": C_TEACHER, "Baseline (no KD)": C_BASELINE, "Distilled (KD)": C_DISTILL}
    for name, (speed, miou, params) in TRADEOFF.items():
        # bubble area proportional to parameter count (the 3rd dimension: model size)
        ax.scatter(speed, miou, s=params * 45, color=colors[name], alpha=0.75,
                   edgecolors="black", linewidths=1.2, zorder=3, label=f"{name} — {params}M")
        # value label placed beside the bubble (avoids clipping on small bubbles)
        ax.annotate(f"{miou}", (speed, miou), textcoords="offset points",
                    xytext=(-26, 0), ha="right", va="center", fontweight="bold",
                    color=colors[name], zorder=4)
    # arrow: same speed, KD lifts mIoU over baseline
    ax.annotate("", xy=(61.97, 52.5), xytext=(61.97, 51.6),
                arrowprops=dict(arrowstyle="->", color=C_DISTILL, lw=2), zorder=2)
    ax.text(63.5, 52.0, "+1.3 mIoU\n(same speed)", color=C_DISTILL, fontweight="bold", va="center")
    ax.set_xlabel("Inference speed (images/second, RTX 3060)  →  faster")
    ax.set_ylabel("mIoU (LoveDA val)  →  more accurate")
    ax.set_title("Accuracy vs. speed trade-off (bubble size = #parameters)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_xlim(20, 72)
    ax.set_ylim(49, 58)
    fig.tight_layout()
    _save(fig, "distill_tradeoff.png")


# ───────────────────────────────────────────────────────────────────────────
# FIGURES 5 & 6 — per-class IoU + confusion matrices (single inference pass)
# ───────────────────────────────────────────────────────────────────────────
def fig_inference_based():
    import torch
    import torch.nn.functional as F
    from models.vit import ViT
    from models.eomt import EoMT
    from training.lightning_module import LightningModule as _Base
    from datasets.loveda_semantic import LoveDASemantic
    from finetuning.loveda import IGNORE_IDX, CLASS_NAMES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inference] device={device} — running the 3 models over the val set")

    def load(net, ckpt):
        c = torch.load(ckpt, map_location="cpu", weights_only=True)
        if "state_dict" in c:
            c = c["state_dict"]
        cleaned = {}
        for k, v in c.items():
            k = k.replace("._orig_mod", "")
            if not k.startswith("network."):
                continue
            cleaned[k[len("network."):]] = v
        net.load_state_dict(cleaned, strict=False)
        return net.to(device).eval()

    def mk_teacher():
        e = ViT(img_size=IMG_SIZE, patch_size=16, backbone_name="vit_small_patch14_reg4_dinov2", ckpt_path="skip")
        return EoMT(encoder=e, num_classes=NUM_CLASSES, num_q=200, num_blocks=3, masked_attn_enabled=False)

    def mk_student():
        e = ViT(img_size=IMG_SIZE, patch_size=16, backbone_name="vit_tiny_patch16_224.augreg_in21k_ft_in1k", ckpt_path="skip")
        return EoMT(encoder=e, num_classes=NUM_CLASSES, num_q=100, num_blocks=3, masked_attn_enabled=False)

    @torch.no_grad()
    def predict(model, img_uint8):
        hw = img_uint8.shape[-2:]
        x = img_uint8[None].float().to(device) / 255.0
        x = F.interpolate(x, IMG_SIZE, mode="bilinear")
        ml, cl = model(x)
        pp = _Base.to_per_pixel_logits_semantic(ml[-1], cl[-1])
        pp = F.interpolate(pp, hw, mode="bilinear")
        return pp.argmax(1)[0].cpu()

    val = LoveDASemantic(path="data/loveda", num_workers=0).setup().val_dataset
    models = {
        "Teacher": load(mk_teacher(), TEACHER_CKPT),
        "Baseline": load(mk_student(), BASELINE_CKPT),
        "Distilled": load(mk_student(), DISTILL_CKPT),
    }
    inter = {m: np.zeros(NUM_CLASSES) for m in models}
    union = {m: np.zeros(NUM_CLASSES) for m in models}
    # confusion matrices (rows = true class, cols = predicted) for baseline & distilled
    conf = {"Baseline": np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64),
            "Distilled": np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)}

    n = len(val)
    for i in range(n):
        img, tgt_dict = val[i]
        img = torch.as_tensor(img)
        target = _Base.to_per_pixel_targets_semantic([tgt_dict], IGNORE_IDX)[0]
        valid = target != IGNORE_IDX
        t_np = target[valid].cpu().numpy()
        for name, model in models.items():
            pred = predict(model, img)
            for c in range(NUM_CLASSES):
                p = (pred == c) & valid
                t = target == c
                inter[name][c] += (p & t).sum().item()
                union[name][c] += (p | t).sum().item()
            if name in conf:
                p_np = pred[valid].cpu().numpy()
                idx = t_np * NUM_CLASSES + p_np
                conf[name] += np.bincount(idx, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
        if (i + 1) % 200 == 0:
            print(f"[inference] {i + 1}/{n}")

    iou = {m: np.where(union[m] > 0, inter[m] / np.maximum(union[m], 1), np.nan) for m in models}

    print("\n[per-class IoU] (640px inference, global IoU on val):")
    print(f"{'class':<14}{'Teacher':>9}{'Baseline':>10}{'Distill':>9}{'d(KD)':>8}")
    for c, cname in enumerate(CLASS_NAMES):
        dt, db, dd = iou['Teacher'][c], iou['Baseline'][c], iou['Distilled'][c]
        print(f"{cname:<14}{dt*100:>8.1f} {db*100:>9.1f} {dd*100:>8.1f} {(dd-db)*100:>+7.1f}")

    _fig_per_class(iou, CLASS_NAMES)
    _fig_confusion(conf, CLASS_NAMES)


def _fig_per_class(iou, class_names):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(NUM_CLASSES)
    w = 0.26
    ax.bar(x - w, iou["Teacher"] * 100, w, label="Teacher (EoMT-S)", color=C_TEACHER)
    ax.bar(x, iou["Baseline"] * 100, w, label="Baseline (no KD)", color=C_BASELINE)
    ax.bar(x + w, iou["Distilled"] * 100, w, label="Distilled (KD)", color=C_DISTILL)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_ylabel("IoU (%)")
    ax.set_title("Per-class IoU: where distillation helps (640px inference)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save(fig, "distill_per_class_iou.png")


def _fig_confusion(conf, class_names):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, name in zip(axes, ["Baseline", "Distilled"]):
        m = conf[name].astype(float)
        row = m.sum(axis=1, keepdims=True)
        norm = np.divide(m, np.maximum(row, 1))  # row-normalized = recall per true class
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{name} (no KD)" if name == "Baseline" else f"{name} (KD)")
        for r in range(NUM_CLASSES):
            for c in range(NUM_CLASSES):
                v = norm[r, c]
                if v >= 0.01:
                    ax.text(c, r, f"{v*100:.0f}", ha="center", va="center",
                            color="white" if v > 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="row-normalized (recall)")
    fig.suptitle("Confusion matrices (row-normalized): baseline vs distilled", fontweight="bold")
    _save(fig, "distill_confusion_matrices.png")


def _save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def main():
    # fast figures (known numbers only)
    fig_trajectory()
    fig_final_bars()
    fig_efficiency()
    fig_tradeoff()
    # inference-based figures (slower)
    fig_inference_based()
    print("\nDone. 6 figures in docs/figures/")


if __name__ == "__main__":
    main()
