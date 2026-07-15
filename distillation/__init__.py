# Knowledge-distillation part of the project (EoMT-S -> EoMT-Tiny on LoveDA).
#
# Consists of:
#   mask_classification_semantic_distill - the distillation LightningModule (KD loss)
#   compare_models                       - efficiency benchmark (params, FLOPs, img/s)
#   qualitative_comparison               - qualitative 3-model comparison figure
#   distill_graphs                       - result figures (trajectory, per-class, ...)
