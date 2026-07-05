# Dataset Constants
# https://github.com/Junjue-Wang/LoveDA

NUM_CLASSES = 7
IGNORE_IDX = 255

# Class after shift (valore raw - 1); raw 0 = no-data -> ignorated
CLASS_NAMES = [
    "background",
    "building",
    "road",
    "water",
    "barren",
    "forest",
    "agriculture",
]

# Palette RGB LoveDA (index = class)
PALETTE = [
    (255, 255, 255),  # background
    (255, 0, 0),      # building
    (255, 255, 0),    # road
    (0, 0, 255),      # water
    (159, 129, 183),  # barren
    (0, 255, 0),      # forest
    (255, 195, 128),  # agriculture
]

# zenodo connection
ZENODO_FILES = {
    "Train.zip": "https://zenodo.org/records/5706578/files/Train.zip?download=1",
    "Val.zip": "https://zenodo.org/records/5706578/files/Val.zip?download=1",
}

# Checkpoint EoMT-S pre-addestrato su COCO panoptic (model zoo del repo)
COCO_CKPT_URL = (
    "https://huggingface.co/tue-mps/coco_panoptic_eomt_small_640_2x"
    "/resolve/main/pytorch_model.bin"
)
COCO_CKPT_FILENAME = "coco_panoptic_eomt_small_640_2x.bin"
