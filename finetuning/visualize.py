# Visualize utils.

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from training.lightning_module import LightningModule as _BaseModule

from finetuning.loveda import CLASS_NAMES, IGNORE_IDX, PALETTE


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

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figura salvata in {save_path}")

    plt.show()