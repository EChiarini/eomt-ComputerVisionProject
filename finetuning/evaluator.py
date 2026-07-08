# building model with pre saved weights and evalutation

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassJaccardIndex

from models.eomt import EoMT
from models.vit import ViT
from training.lightning_module import LightningModule as _BaseModule

from finetuning.loveda import IGNORE_IDX, NUM_CLASSES, CLASS_NAMES


def build_eomt_small(num_classes: int,img_size: tuple[int, int] = (640, 640),num_q: int = 200,num_blocks: int = 3,backbone_name: str = "vit_small_patch14_reg4_dinov2",) -> EoMT:
    """EoMT-S"""
    encoder = ViT(img_size=img_size, backbone_name=backbone_name)
    return EoMT(
        encoder=encoder,
        num_classes=num_classes,
        num_q=num_q,
        num_blocks=num_blocks,
        masked_attn_enabled=False,
    )
 

def load_network_weights(network: EoMT, ckpt_path, drop_class_head: bool = False) -> EoMT:
    """Load i .bin and .ckpt."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    state = {}
    for key, value in ckpt.items():
        key = key.replace("._orig_mod", "")
        if key.startswith("network."):
            state[key[len("network."):]] = value

    if drop_class_head:
        state = {k: v for k, v in state.items() if "class_head" not in k}

    missing, unexpected = network.load_state_dict(state, strict=False)
    missing = [k for k in missing if not (drop_class_head and "class_head" in k)]
    if missing or unexpected:
        raise ValueError(f"Missing keys: {missing}\n      Unexpected keys: {unexpected}")

    print(f"Loaded {len(state)} tensors from {Path(ckpt_path).name}")
    return network


@torch.no_grad()
def predict(network: EoMT,imgs: torch.Tensor,class_mapping: torch.Tensor = None,input_size: tuple[int, int] = (640, 640),amp: bool = True,) -> torch.Tensor:
    """Semantic prediction for pixel: (B,3,H,W) uint8 -> (B,H,W) long."""
    device = next(network.parameters()).device
    out_size = imgs.shape[-2:]

    x = imgs.to(device).float() / 255.0
    x = F.interpolate(x, input_size, mode="bilinear")

    with torch.autocast("cuda", torch.float16, enabled=amp and device.type == "cuda"):
        mask_logits_per_layer, class_logits_per_layer = network(x)
        mask_logits = F.interpolate(
            mask_logits_per_layer[-1], input_size, mode="bilinear"
        )
        probs = _BaseModule.to_per_pixel_logits_semantic(
            mask_logits, class_logits_per_layer[-1]
        )

    probs = probs.float()

    if class_mapping is not None:
        grouped = torch.zeros(
            probs.shape[0], NUM_CLASSES, *probs.shape[-2:], device=probs.device
        )
        grouped.index_add_(1, class_mapping.to(probs.device), probs)
        probs = grouped

    probs = F.interpolate(probs, out_size, mode="bilinear")
    return probs.argmax(dim=1)


@torch.no_grad()
def evaluate_on_loveda(network: EoMT,datamodule,class_mapping: torch.Tensor = None,input_size: tuple[int, int] = (640, 640),device: str = "cuda",amp: bool = True,max_batches: int = None,) -> dict:
    """mIoU (global and per class) on LoveDA validation set."""
    network = network.to(device).eval()
    metric = MulticlassJaccardIndex(
        num_classes=NUM_CLASSES,
        ignore_index=IGNORE_IDX,
        average=None,
        validate_args=False,
    ).to(device)

    loader = datamodule.val_dataloader()
    start = time.time()

    for batch_idx, (imgs, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        imgs = torch.stack(list(imgs)).to(device)
        per_pixel = _BaseModule.to_per_pixel_targets_semantic(
            list(targets), IGNORE_IDX
        )
        target = torch.stack(per_pixel).to(device)

        preds = predict(network, imgs, class_mapping, input_size, amp)
        metric.update(preds, target)

        if batch_idx % 50 == 0:
            print(f"  batch {batch_idx}/{len(loader)}", end="\r")

    iou_per_class = metric.compute().cpu()
    elapsed = time.time() - start

    results = {
        "miou": float(iou_per_class.mean()),
        "iou_per_class": {
            name: float(iou) for name, iou in zip(CLASS_NAMES, iou_per_class)
        },
        "eval_seconds": round(elapsed, 1),
        "num_batches": batch_idx + 1,
    }
    print(f"\nmIoU: {results['miou'] * 100:.2f}  ({elapsed:.0f}s)")
    return results


def save_results(results: dict, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results saved in {path}")


def load_results(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_checkpoints(root=".") -> list[Path]:
    """Find saved .ckpt (wandb or lightning_logs), most recent first."""
    root = Path(root)
    ckpts = list(root.glob("eomt-loveda/**/checkpoints/*.ckpt"))
    ckpts += list(root.glob("lightning_logs/**/checkpoints/*.ckpt"))
    return sorted(ckpts, key=lambda p: p.stat().st_mtime, reverse=True)
