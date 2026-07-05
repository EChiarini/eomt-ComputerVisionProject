# Download datatset LoveDA & checkpoints

from pathlib import Path

import requests
from tqdm.auto import tqdm

from finetuning.loveda import COCO_CKPT_FILENAME, COCO_CKPT_URL, ZENODO_FILES

CHUNK_SIZE = 1 << 20  # 1 MiB


def download_file(url: str, dest: Path, skip_if_exists: bool = True) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if skip_if_exists and dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} esiste già ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar, open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
                bar.update(len(chunk))

    tmp.replace(dest)
    return dest


def download_loveda(data_dir="data/loveda") -> Path:
    """Download Train.zip e Val.zip of LoveDA from Zenodo."""
    data_dir = Path(data_dir)
    for filename, url in ZENODO_FILES.items():
        download_file(url, data_dir / filename)
    return data_dir


def download_coco_checkpoint(ckpt_dir="checkpoints") -> Path:
    """Download the EoMT-S COCO panoptic checkpoint from HuggingFace."""
    return download_file(COCO_CKPT_URL, Path(ckpt_dir) / COCO_CKPT_FILENAME)
