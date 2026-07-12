# Visualize utils.

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from training.lightning_module import LightningModule as _BaseModule

from finetuning.loveda import CLASS_NAMES, IGNORE_IDX, PALETTE


def _savefig(fig, save_path):
    """Save figure creating parent folders as needed."""
    if save_path is None:
        return
    from pathlib import Path

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Figura salvata in {path}")


def colorize(mask: torch.Tensor) -> np.ndarray:
    """(H,W) long con classi 0..6 e 255=ignore -> immagine RGB uint8."""
    mask = mask.cpu().numpy()
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)

    for cls_idx, color in enumerate(PALETTE):
        out[mask == cls_idx] = color
    out[mask == IGNORE_IDX] = (0, 0, 0)

    return out


def legend_handles():
    return [
        Patch(facecolor=np.array(color) / 255.0, edgecolor="k", label=name)
        for name, color in zip(CLASS_NAMES, PALETTE)
    ]


def get_val_samples(datamodule, indices):
    """Coppie (img uint8, target per-pixel) dal validation set, per gli indici dati."""
    imgs, targets = [], []
    for i in indices:
        img, target = datamodule.val_dataset[i]
        imgs.append(torch.as_tensor(img))
        targets.append(
            _BaseModule.to_per_pixel_targets_semantic([target], IGNORE_IDX)[0]
        )
    return torch.stack(imgs), torch.stack(targets)


def show_samples(imgs, targets, preds_by_name=None, figsize_per_cell=3.2, save_path=None):
    """Griglia: righe = immagini; colonne = immagine, GT, una per ogni modello.

    preds_by_name: dict {"nome modello": tensor (N,H,W)} oppure None.
    """
    preds_by_name = preds_by_name or {}
    n_rows = len(imgs)
    n_cols = 2 + len(preds_by_name)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_cell * n_cols, figsize_per_cell * n_rows),
        squeeze=False,
    )
    col_titles = ["Immagine", "Ground truth", *preds_by_name.keys()]

    for row in range(n_rows):
        cells = [imgs[row].permute(1, 2, 0).cpu().numpy(), colorize(targets[row])]
        cells += [colorize(preds[row]) for preds in preds_by_name.values()]

        for col, cell in enumerate(cells):
            axes[row][col].imshow(cell)
            axes[row][col].axis("off")
            if row == 0:
                axes[row][col].set_title(col_titles[col])

    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        ncol=len(CLASS_NAMES),
        bbox_to_anchor=(0.5, -0.02),
    )
    plt.tight_layout()

    _savefig(fig, save_path)

    plt.show()
    
    
def plot_training_curves(run_name, log_root="lightning_logs", save_path=None):
    """Plot train loss and validation mIoU from the CSVLogger metrics.csv"""
    import pandas as pd
    from pathlib import Path

    # skip aborted runs (version dirs without a metrics.csv)
    versions = sorted(
        (
            p
            for p in Path(log_root, run_name).glob("version_*")
            if (p / "metrics.csv").exists()
        ),
        key=lambda p: p.stat().st_mtime,
    )
    if not versions:
        raise FileNotFoundError(f"No runs with metrics.csv in {log_root}/{run_name}")
    metrics = pd.read_csv(versions[-1] / "metrics.csv")

    loss_col = next((c for c in metrics.columns if "train_loss_total" in c), None)
    iou_col = next(
        (c for c in metrics.columns if "val_iou_all" in c and "block" not in c), None
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    if loss_col is not None:
        loss = metrics.dropna(subset=[loss_col])
        axes[0].plot(loss["step"], loss[loss_col])
    axes[0].set(xlabel="step", ylabel="loss totale", title="Training loss")
    axes[0].grid(alpha=0.3)

    if iou_col is not None:
        val = metrics.dropna(subset=[iou_col])
        axes[1].plot(val["epoch"], val[iou_col] * 100, marker="o")
    axes[1].set(xlabel="epoca", ylabel="mIoU (%)", title="Validation mIoU")
    axes[1].grid(alpha=0.3)

    fig.suptitle(run_name)
    plt.tight_layout()

    _savefig(fig, save_path)

    plt.show()
    return metrics


def plot_confusion_matrix(cm, title="Confusion Matrix (rows = GT)", save_path=None):
    """cm: (7,7) row-normalized tensor from evaluator.compute_confusion_matrix."""
    cm = cm.numpy() if hasattr(cm, "numpy") else np.asarray(cm)

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground truth")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = float(cm[i, j])
            ax.text(
                j, i, f"{v:.2f}",
                ha="center", va="center", fontsize=8,
                color="white" if v > 0.5 else "black",
            )

    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved in {save_path}")

    plt.show()


def plot_comparison_bars(results_by_name: dict, save_path=None):
    """Bar chart for each class: one bar for each model in results_by_name."""
    names = list(results_by_name.keys())
    n_models = len(names)
    x = np.arange(len(CLASS_NAMES))
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, name in enumerate(names):
        ious = [
            results_by_name[name]["iou_per_class"][cls] * 100 for cls in CLASS_NAMES
        ]
        ax.bar(x + (i - (n_models - 1) / 2) * width, ious, width, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=20)
    ax.set_ylabel("IoU (%)")
    ax.set_title("IoU per class on LoveDA val")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved in {save_path}")

    plt.show()