# All the utils that will be used for the finetuning part of the project.

# Consists of:
#   loveda      - constants of the dataset (classes, palette, download)
#   coco_classes- COCO panoptic classes and COCO -> LoveDA mapping
#   download    - download of the dataset and the checkpoint
#   evaluator   - model construction, weight loading, mIoU evaluation
#   visualize   - qualitative visualizations (GT vs predictions)
#   semantic_module - save previews