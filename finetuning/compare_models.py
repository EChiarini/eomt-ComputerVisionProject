# Confronto di efficienza tra i modelli EoMT (teacher vs studente).
#
# Misura, per ogni architettura, sulla GPU disponibile:
#   - numero di parametri
#   - GFLOPs per un singolo forward a risoluzione di inferenza (640x640)
#   - velocita' reale (immagini/secondo)
#
# Nota sui FLOPs: fvcore non ha un handler per F.scaled_dot_product_attention
# (l'attention fusa usata in models/eomt.py), quindi il numero riportato
# SOTTOSTIMA il costo reale dell'attention. Entrambi i modelli usano lo
# stesso codice di attention, quindi il bias e' lo stesso per entrambi:
# il confronto RELATIVO (rapporto teacher/studente) resta valido, il valore
# ASSOLUTO no. Uso consigliato: riportare nel report solo i rapporti (Nx).
#
# Uso (va lanciato come modulo, altrimenti "models" non si importa):
#   uv run --no-project python -m finetuning.compare_models
#
# Per aggiungere/modificare i modelli da confrontare, modifica il dict MODELS
# in fondo al file (es. quando arriva il checkpoint reale di Russo, o se
# num_q/num_blocks del suo fine-tuning sono diversi da quanto assunto qui).

import time
import warnings

import torch

from models.vit import ViT
from models.eomt import EoMT

IMG_SIZE = (640, 640)
NUM_CLASSES = 7  # LoveDA: 7 classi semantic
WARMUP_ITERS = 5
BENCH_ITERS = 20


def build_teacher() -> EoMT:
    # EoMT-S: ViT-Small DINOv2. num_q=200/num_blocks=3 assunti dal checkpoint
    # COCO panoptic originale (tue-mps/coco_panoptic_eomt_small_640_2x) --
    # da correggere se il fine-tuning di Russo ha cambiato questi valori.
    encoder = ViT(img_size=IMG_SIZE, patch_size=16, backbone_name="vit_small_patch14_reg4_dinov2")
    return EoMT(
        encoder=encoder,
        num_classes=NUM_CLASSES,
        num_q=200,
        num_blocks=3,
        masked_attn_enabled=False,  # modalita' inferenza: un solo forward del mask module
    )


def build_student() -> EoMT:
    # EoMT-Tiny: ViT-Tiny ImageNet, stessa config di configs/dinov2/loveda/semantic/eomt_tiny_640*.yaml
    encoder = ViT(img_size=IMG_SIZE, patch_size=16, backbone_name="vit_tiny_patch16_224.augreg_in21k_ft_in1k")
    return EoMT(
        encoder=encoder,
        num_classes=NUM_CLASSES,
        num_q=100,
        num_blocks=3,
        masked_attn_enabled=False,
    )


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def measure_flops(model: torch.nn.Module, device: torch.device) -> float:
    from fvcore.nn import FlopCountAnalysis

    dummy = torch.randn(1, 3, *IMG_SIZE, device=device)
    model.eval()
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")  # silenzia i warning "unsupported operator" di fvcore
        flops = FlopCountAnalysis(model, dummy)
        total = flops.total()
    return total


def measure_speed(model: torch.nn.Module, device: torch.device, batch_size: int = 1) -> float:
    dummy = torch.randn(batch_size, 3, *IMG_SIZE, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(BENCH_ITERS):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    return (BENCH_ITERS * batch_size) / elapsed


MODELS = {
    "Teacher (EoMT-S)": build_teacher,
    "Studente (EoMT-Tiny)": build_student,
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Risoluzione: {IMG_SIZE}, batch inferenza: 1, iterazioni: {BENCH_ITERS} (+{WARMUP_ITERS} warmup)\n")

    rows = []
    for name, build_fn in MODELS.items():
        model = build_fn().to(device)
        n_params = count_params(model)
        gflops = measure_flops(model, device) / 1e9
        img_s = measure_speed(model, device, batch_size=1)
        rows.append((name, n_params, gflops, img_s))

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    header = f"{'Modello':<25} {'Param (M)':>12} {'GFLOPs*':>10} {'img/s (bs=1)':>14}"
    print(header)
    print("-" * len(header))
    for name, n_params, gflops, img_s in rows:
        print(f"{name:<25} {n_params / 1e6:>12.2f} {gflops:>10.1f} {img_s:>14.2f}")
    print("\n* GFLOPs sottostimati (SDPA non contato da fvcore) -- usare solo i rapporti tra modelli, non il valore assoluto.\n")

    if len(rows) >= 2:
        base_name, base_params, base_gflops, base_speed = rows[0]
        for name, n_params, gflops, img_s in rows[1:]:
            print(
                f"{name} vs {base_name}: "
                f"{base_params / n_params:.2f}x meno parametri, "
                f"{base_gflops / gflops:.2f}x meno FLOPs, "
                f"{img_s / base_speed:.2f}x piu' veloce"
            )


if __name__ == "__main__":
    main()
