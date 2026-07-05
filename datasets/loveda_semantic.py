# ---------------------------------------------------------------
# LoveDA semantic segmentation DataModule.
#
# LoveDA (Wang et al., NeurIPS 2021 Datasets & Benchmarks):
# remote sensing semantic segmentation, 7 classes, Urban + Rural domains.
# https://github.com/Junjue-Wang/LoveDA — data on Zenodo (record 5706578).
#
# Expected files in --data.path (no extraction needed):
#   Train.zip  ->  Train/{Urban,Rural}/images_png/*.png
#                  Train/{Urban,Rural}/masks_png/*.png
#   Val.zip    ->  Val/{Urban,Rural}/images_png/*.png
#                  Val/{Urban,Rural}/masks_png/*.png
#
# Mask values: 0 = no-data (ignored), 1..7 = classes (shifted to 0..6).
# ---------------------------------------------------------------


from pathlib import Path
from typing import Union
from torch.utils.data import ConcatDataset, DataLoader

from datasets.lightning_data_module import LightningDataModule
from datasets.dataset import Dataset
from datasets.transforms import Transforms

# 0 is the no-data value: not mapped -> pixels stay at ignore_idx
CLASS_MAPPING = {i: i - 1 for i in range(1, 8)}

CLASS_NAMES = [
    "background",
    "building",
    "road",
    "water",
    "barren",
    "forest",
    "agriculture",
]


class LoveDASemantic(LightningDataModule):
    def __init__(
        self,
        path,
        num_workers: int = 4,
        batch_size: int = 4,
        img_size: tuple[int, int] = (640, 640),
        num_classes: int = 7,
        color_jitter_enabled=True,
        scale_range=(0.5, 2.0),
        check_empty_targets=True,
        domains: tuple[str, ...] = ("Urban", "Rural"),
    ) -> None:
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(ignore=["_class_path"])

        self.domains = domains

        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

    @staticmethod
    def target_parser(target, **kwargs):
        masks, labels = [], []

        for label_id in target[0].unique():
            cls_id = label_id.item()

            if cls_id not in CLASS_MAPPING:
                continue

            masks.append(target[0] == label_id)
            labels.append(CLASS_MAPPING[cls_id])

        return masks, labels, [False for _ in range(len(masks))]

    def _build_datasets(self, split: str, transforms=None):
        zip_path = Path(self.path, f"{split}.zip")

        return [
            Dataset(
                zip_path=zip_path,
                target_zip_path=zip_path,
                img_folder_path_in_zip=Path(f"./{split}/{domain}/images_png"),
                target_folder_path_in_zip=Path(f"./{split}/{domain}/masks_png"),
                img_suffix=".png",
                target_suffix=".png",
                target_parser=self.target_parser,
                check_empty_targets=self.check_empty_targets,
                transforms=transforms,
            )
            for domain in self.domains
        ]

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        self.train_dataset = ConcatDataset(
            self._build_datasets("Train", transforms=self.transforms)
        )
        self.val_dataset = ConcatDataset(self._build_datasets("Val"))

        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )
