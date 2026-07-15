# EoMT on LoveDA — Fine-tuning & Knowledge Distillation

Computer Vision master's project by **Emiliano Chiarini** and **Luca Di Mauro**, built on
**EoMT** ("Your ViT is Secretly an Image Segmentation Model", Kerssies et al., CVPR 2025 Highlight —
[paper](https://arxiv.org/abs/2503.19108) · [official repo](https://github.com/tue-mps/eomt)).

We take EoMT to a domain far from its training data, **LoveDA**, semantic segmentation of
aerial imagery (7 land-cover classes) — and run two experiments:

| Experiment | Result (mIoU on LoveDA val) |
|---|---|
| **1. Domain adaptation** — fine-tune EoMT-S from COCO weights | zero-shot **23.3** → fine-tuned **56.1** |
| &nbsp;&nbsp;&nbsp;ablation: fine-tune from DINOv2 weights only | **56.1** → the gain comes from DINOv2, not from COCO |
| **2. Compression** — distill EoMT-S (teacher) into EoMT-Tiny | distilled **52.7** vs. baseline **51.4** (3.8× smaller, 2.2× faster) |

## Setup

Windows Python ≥ 3.13.

```bash
python -m venv .venv && source .venv/Scripts/activate   # or bin/activate on Linux
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128 pandas
pip install -r requirements.txt
```


Dataset and pre-trained weights are downloaded automatically by the first notebook
(LoveDA from Zenodo; EoMT-S COCO checkpoint from HuggingFace).

## Part 1 — Fine-tuning (notebooks)

Run the notebooks in [`finetuning/`](finetuning/) in order:

| Notebook | What it does |
|---|---|
| `0_Setup` | environment check, dataset + checkpoint download, sanity checks |
| `1_Baseline_Generico` | zero-shot evaluation of the COCO model (COCO→LoveDA class mapping) |
| `2_FineTuning` | fine-tuning on LoveDA (+ DINOv2-only ablation) |
| `3_Confronto` | evaluation, per-class comparison, confusion matrices, qualitative results |

Outputs (metrics, figures) land in `results/finetuning/`.

## Part 2 — Knowledge distillation (CLI)

Teacher = the fine-tuned EoMT-S (exported by Part 1 to `checkpoints/loveda_ft_eomt_small_640_best.bin`).
Student = EoMT-Tiny (ViT-Tiny, no DINOv2). Per-pixel KL on the collapsed semantic maps,
applied to all deep-supervised outputs (`training/mask_classification_semantic_distill.py`).

```bash
# baseline student (no distillation)
python main.py fit -c configs/dinov2/loveda/semantic/eomt_tiny_640.yaml --compile_disabled

# distilled student
python main.py fit -c configs/dinov2/loveda/semantic/eomt_tiny_640_distill.yaml --compile_disabled
```

## What we added vs. the original EoMT repo

Everything below is new; every other file is the unmodified official EoMT codebase.

**Shared — LoveDA dataset**
```
datasets/loveda_semantic.py                LoveDA DataModule (zip-based, Urban+Rural)
```

**Part 1 — Fine-tuning**
```
configs/dinov2/loveda/semantic/eomt_small_640.yaml   fine-tuning config
finetuning/
  0_Setup.ipynb  1_Baseline_Generico.ipynb           notebooks (run in order)
  2_FineTuning.ipynb  3_Confronto.ipynb
  loveda.py          dataset constants (classes, palette, URLs)
  coco_classes.py    COCO->LoveDA class mapping for the zero-shot baseline
  download.py        dataset + checkpoint download
  evaluator.py       model build, weight loading, mIoU + confusion matrix
  visualize.py       curves, per-class bars, confusion matrix, qualitatives
  semantic_module.py training subclass that saves previews to disk (no wandb)
```

**Part 2 — Knowledge distillation**
```
configs/dinov2/loveda/semantic/eomt_tiny_640.yaml          student baseline config
configs/dinov2/loveda/semantic/eomt_tiny_640_distill.yaml   student + distillation config
distillation/
  mask_classification_semantic_distill.py    KD LightningModule (teacher + KL loss)
  mask_classification_semantic_distill.ipynb  training notebook
  compare_models.py  distill_graphs.py  qualitative_comparison.py   analysis scripts
```

**Results & misc**
```
results/finetuning/     Part 1 metrics, figures, per-epoch previews
results/distillation/   Part 2 metrics and figures
.python-version         pinned Python for the project venv
```
Modified from upstream: `README.md`, `.gitignore`, `model_zoo/dinov2.md`.

## Credits

```BibTeX
@inproceedings{kerssies2025eomt,
  author    = {Kerssies, Tommie and Cavagnero, Niccol\`{o} and Hermans, Alexander and Norouzi, Narges and Averta, Giuseppe and Leibe, Bastian and Dubbelman, Gijs and {de Geus}, Daan},
  title     = {{Your ViT is Secretly an Image Segmentation Model}},
  booktitle = {CVPR},
  year      = {2025},
}
```

LoveDA: Wang et al., *LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive
Semantic Segmentation*, NeurIPS 2021 D&B ([repo](https://github.com/Junjue-Wang/LoveDA)).
Original EoMT license: [MIT](LICENSE).
