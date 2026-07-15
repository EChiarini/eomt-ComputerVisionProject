# Confronto qualitativo tra i tre modelli su immagini di LoveDA val.
#
# Produce una griglia: immagine | ground truth | teacher | baseline | distillato.
# Le immagini NON sono scelte a caso: si calcola il mIoU per-immagine dei due
# studenti su un campione della validation e si selezionano
#   - le 2 immagini dove la distillazione migliora di piu' rispetto al baseline,
#   - 1 immagine "tipica" (delta mediano),
#   - 1 immagine dove il distillato va peggio (caso negativo, per onesta').
#
# Uso:
#   uv run --no-project python -m distillation.qualitative_comparison
#
# Output: docs/figures/confronto_qualitativo.png + statistiche a terminale.

from pathlib import Path

import torch
import torch.nn.functional as F

from models.vit import ViT
from models.eomt import EoMT
from training.lightning_module import LightningModule as _BaseModule
from datasets.loveda_semantic import LoveDASemantic
from finetuning.loveda import IGNORE_IDX, NUM_CLASSES
from finetuning.visualize import show_samples

IMG_SIZE = (640, 640)
SCAN_N = 300  # quante immagini val scansionare per la selezione
TEACHER_CKPT = "checkpoints/loveda_ft_eomt_small_640_best.bin"
BASELINE_CKPT = "eomt/413yl1ap/checkpoints/epoch=19-step=12520.ckpt"
DISTILL_CKPT = "eomt/cju4qzzo/checkpoints/best-19.ckpt"
OUT_PATH = Path("results/distillation/confronto_qualitativo.png")


def load_state_into(network: torch.nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    cleaned = {}
    for key, value in ckpt.items():
        key = key.replace("._orig_mod", "")
        if not key.startswith("network."):
            continue  # scarta criterion.*, metrics.*, ecc.
        cleaned[key[len("network.") :]] = value
    missing, unexpected = network.load_state_dict(cleaned, strict=False)
    assert not unexpected, f"chiavi inattese da {ckpt_path}: {unexpected[:5]}"
    print(f"[load] {ckpt_path}: {len(cleaned)} tensori, missing={len(missing)}")
    return network


def build_teacher() -> EoMT:
    encoder = ViT(img_size=IMG_SIZE, patch_size=16, backbone_name="vit_small_patch14_reg4_dinov2", ckpt_path="skip")
    return EoMT(encoder=encoder, num_classes=NUM_CLASSES, num_q=200, num_blocks=3, masked_attn_enabled=False)


def build_student() -> EoMT:
    encoder = ViT(img_size=IMG_SIZE, patch_size=16, backbone_name="vit_tiny_patch16_224.augreg_in21k_ft_in1k", ckpt_path="skip")
    return EoMT(encoder=encoder, num_classes=NUM_CLASSES, num_q=100, num_blocks=3, masked_attn_enabled=False)


@torch.no_grad()
def predict(model: EoMT, img_uint8: torch.Tensor, device) -> torch.Tensor:
    """img (3,H,W) uint8 -> mappa di classi predette (H,W) long, alla risoluzione originale."""
    orig_hw = img_uint8.shape[-2:]
    x = img_uint8[None].float().to(device) / 255.0
    x = F.interpolate(x, IMG_SIZE, mode="bilinear")
    mask_logits_list, class_logits_list = model(x)
    per_pixel = _BaseModule.to_per_pixel_logits_semantic(
        mask_logits_list[-1], class_logits_list[-1]
    )  # (1, C, 160, 160)
    per_pixel = F.interpolate(per_pixel, orig_hw, mode="bilinear")
    return per_pixel.argmax(1)[0].cpu()  # (H, W)


def per_image_miou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """mIoU sulle sole classi presenti nel ground truth (ignora IGNORE_IDX)."""
    valid = target != IGNORE_IDX
    ious = []
    for cls in target[valid].unique():
        p = (pred == cls) & valid
        t = target == cls
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union > 0:
            ious.append(inter / union)
    return sum(ious) / len(ious) if ious else float("nan")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    datamodule = LoveDASemantic(path="data/loveda", num_workers=0).setup()
    val_dataset = datamodule.val_dataset
    n_scan = min(SCAN_N, len(val_dataset))
    print(f"Validation set: {len(val_dataset)} immagini, ne scansiono {n_scan}")

    baseline = load_state_into(build_student(), BASELINE_CKPT).to(device).eval()
    distill = load_state_into(build_student(), DISTILL_CKPT).to(device).eval()

    # --- fase 1: scansione con i due studenti per selezionare le immagini ---
    deltas = []  # (delta, idx, miou_baseline, miou_distill)
    for idx in range(n_scan):
        img, target_dict = val_dataset[idx]
        img = torch.as_tensor(img)
        target = _BaseModule.to_per_pixel_targets_semantic([target_dict], IGNORE_IDX)[0]

        miou_b = per_image_miou(predict(baseline, img, device), target)
        miou_d = per_image_miou(predict(distill, img, device), target)
        deltas.append((miou_d - miou_b, idx, miou_b, miou_d))

        if (idx + 1) % 50 == 0:
            print(f"  scansionate {idx + 1}/{n_scan}")

    deltas.sort(reverse=True)
    best_two = deltas[:2]                    # dove la KD aiuta di piu'
    median_one = deltas[len(deltas) // 2]     # caso tipico
    worst_one = deltas[-1]                    # dove la KD peggiora (onesta')
    selected = best_two + [median_one, worst_one]

    print("\nImmagini selezionate (delta = distillato - baseline, mIoU per-immagine):")
    labels = ["miglioramento #1", "miglioramento #2", "caso mediano", "caso peggiore"]
    for label, (delta, idx, miou_b, miou_d) in zip(labels, selected):
        print(f"  [{label}] idx={idx}: baseline={miou_b:.3f}  distillato={miou_d:.3f}  delta={delta:+.3f}")

    # --- fase 2: teacher solo sulle selezionate, poi rendering ---
    teacher = load_state_into(build_teacher(), TEACHER_CKPT).to(device).eval()

    imgs, targets = [], []
    preds = {"Teacher (EoMT-S)": [], "Baseline (no KD)": [], "Distillato (KD)": []}
    for _, idx, _, _ in selected:
        img, target_dict = val_dataset[idx]
        img = torch.as_tensor(img)
        imgs.append(img)
        targets.append(_BaseModule.to_per_pixel_targets_semantic([target_dict], IGNORE_IDX)[0])
        preds["Teacher (EoMT-S)"].append(predict(teacher, img, device))
        preds["Baseline (no KD)"].append(predict(baseline, img, device))
        preds["Distillato (KD)"].append(predict(distill, img, device))

    imgs = torch.stack(imgs)
    targets = torch.stack(targets)
    preds = {name: torch.stack(p) for name, p in preds.items()}

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    show_samples(imgs, targets, preds_by_name=preds, save_path=str(OUT_PATH))


if __name__ == "__main__":
    main()
