# validation previews are saved to disk

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from training.mask_classification_semantic import MaskClassificationSemantic
from finetuning.visualize import colorize


class MaskClassificationSemanticLocal(MaskClassificationSemantic):
    """per-epoch validation preview is saved as a PNG in results/finetuning/val_previews/."""

    preview_dir = Path("results/finetuning/val_previews")

    @torch.compiler.disable
    def plot_semantic(self, img, target, logits, log_prefix, block_idx, batch_idx, cmap="tab20"):
        if self.trainer.sanity_checking or block_idx != self.network.num_blocks:
            return

        pred = torch.argmax(logits, dim=0)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
        cells = [
            (img.permute(1, 2, 0).cpu().numpy(), "Immagine"),
            (colorize(target), "Ground truth"),
            (colorize(pred), "Predizione"),
        ]
        for ax, (content, title) in zip(axes, cells):
            ax.imshow(content)
            ax.set_title(title)
            ax.axis("off")
        fig.tight_layout()

        self.preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            self.preview_dir / f"epoch_{self.current_epoch:03d}.png",
            dpi=110,
            bbox_inches="tight",
        )
        plt.close(fig)
