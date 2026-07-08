# COCO classes as seen in CLASS_MAPPING in datasets/coco_panoptic.py
# used for mapping into the 7 classes of LoveDA

import torch

# fmt: off
COCO_PANOPTIC_CLASSES = [
    # things (0-79)
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
    # stuff (80-132)
    "banner", "blanket", "bridge", "cardboard", "counter", "curtain",
    "door-stuff", "floor-wood", "flower", "fruit", "gravel", "house", "light",
    "mirror-stuff", "net", "pillow", "platform", "playingfield", "railroad",
    "river", "road", "roof", "sand", "sea", "shelf", "snow", "stairs", "tent",
    "towel", "wall-brick", "wall-stone", "wall-tile", "wall-wood",
    "water-other", "window-blind", "window-other", "tree-merged",
    "fence-merged", "ceiling-merged", "sky-other-merged", "cabinet-merged",
    "table-merged", "floor-other-merged", "pavement-merged",
    "mountain-merged", "grass-merged", "dirt-merged", "paper-merged",
    "food-other-merged", "building-other-merged", "rock-merged",
    "wall-other-merged", "rug-merged",
]
# fmt: on

assert len(COCO_PANOPTIC_CLASSES) == 133

# Tried our best to match classes, but things like "agriculture" (cultivated land) does not have a true
# correspondence in COCO; "grass-merged" is the least bad proxy.
# The mapping is deliberately easy to modify for sensitivity analysis.
LOVEDA_FROM_COCO_NAMES = {
    "building": ["house", "building-other-merged", "roof"],
    "road": ["road", "pavement-merged", "railroad", "bridge"],
    "water": ["river", "sea", "water-other"],
    "barren": ["sand", "dirt-merged", "gravel", "rock-merged", "mountain-merged"],
    "forest": ["tree-merged"],
    "agriculture": ["grass-merged"],
}


def build_coco_to_loveda_mapping(loveda_class_names) -> torch.Tensor:
    """Tensor [133] with the destination LoveDA index for each COCO class.

    Default: 0 (background).
    """
    mapping = torch.zeros(len(COCO_PANOPTIC_CLASSES), dtype=torch.long)

    for loveda_name, coco_names in LOVEDA_FROM_COCO_NAMES.items():
        loveda_idx = loveda_class_names.index(loveda_name)
        for coco_name in coco_names:
            coco_idx = COCO_PANOPTIC_CLASSES.index(coco_name)
            mapping[coco_idx] = loveda_idx

    return mapping
