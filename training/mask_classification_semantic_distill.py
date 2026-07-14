# ---------------------------------------------------------------
# Knowledge distillation for LoveDA semantic segmentation.
#
# Teacher: EoMT-S (ViT-Small DINOv2) fine-tuned on LoveDA.
# Student: EoMT with a lighter backbone (ViT-Tiny, ImageNet-pretrained).
#
# Method: per-pixel logit distillation. Query-level predictions of both
# models are collapsed into per-pixel semantic distributions with
# to_per_pixel_logits_semantic (sigmoid(masks) x softmax(classes)), then
# the student is trained with the standard mask-classification loss on
# ground truth PLUS a temperature-scaled KL divergence against the
# teacher distribution, applied to every deeply-supervised output.
# ---------------------------------------------------------------


from typing import List, Optional
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.mask_classification_semantic import MaskClassificationSemantic


class MaskClassificationSemanticDistill(MaskClassificationSemantic):
    def __init__(
        self,
        network: nn.Module,
        teacher_network: nn.Module,
        teacher_ckpt_path: str,
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        kd_coefficient: float = 1.0,
        kd_temperature: float = 2.0,
        ignore_idx: int = 255,
        lr: float = 1e-4,
        llrd: float = 0.8,
        llrd_l2_enabled: bool = True,
        lr_mult: float = 1.0,
        weight_decay: float = 0.05,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.8,
        ckpt_path: Optional[str] = None,
        delta_weights: bool = False,
        load_ckpt_class_head: bool = True,
    ):
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            ignore_idx=ignore_idx,
            lr=lr,
            llrd=llrd,
            llrd_l2_enabled=llrd_l2_enabled,
            lr_mult=lr_mult,
            weight_decay=weight_decay,
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            no_object_coefficient=no_object_coefficient,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            mask_thresh=mask_thresh,
            overlap_thresh=overlap_thresh,
            ckpt_path=ckpt_path,
            delta_weights=delta_weights,
            load_ckpt_class_head=load_ckpt_class_head,
        )

        self.kd_coefficient = kd_coefficient
        self.kd_temperature = kd_temperature

        teacher_network.requires_grad_(False)
        teacher_network.eval()
        self._load_teacher_ckpt(teacher_network, teacher_ckpt_path)

        # Kept in a plain list so the teacher is NOT a registered submodule:
        # no teacher weights in checkpoints, no teacher params in the optimizer,
        # and Lightning's .train()/.eval() calls never flip its mode.
        self._teacher = [teacher_network]

    @property
    def teacher(self) -> nn.Module:
        return self._teacher[0]

    @staticmethod
    def _load_teacher_ckpt(teacher: nn.Module, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        cleaned = {}
        for key, value in ckpt.items():
            key = key.replace("._orig_mod", "")
            if key.startswith("network."):
                key = key[len("network.") :]
            if key.startswith("criterion.") or key.startswith("metrics."):
                continue
            cleaned[key] = value

        target = teacher.state_dict()
        loadable, skipped = {}, []
        for key, value in cleaned.items():
            if key in target and target[key].shape == value.shape:
                loadable[key] = value
            else:
                skipped.append(key)
        missing = [key for key in target if key not in loadable]

        teacher.load_state_dict(loadable, strict=False)

        logging.info(
            f"[distill] teacher ckpt '{ckpt_path}': loaded {len(loadable)} tensors, "
            f"skipped {len(skipped)} (name/shape mismatch), missing {len(missing)}"
        )
        if skipped:
            logging.warning(
                f"[distill] teacher keys skipped (first 10): {skipped[:10]} — "
                "expected only for the placeholder COCO teacher (class_head 134 vs 8); "
                "with the real LoveDA teacher this list must be empty."
            )

    def on_fit_start(self):
        self.teacher.to(self.device)

    def _kd_loss(self, student_probs: torch.Tensor, teacher_probs: torch.Tensor):
        # Per-pixel KL divergence with temperature. The collapsed semantic maps
        # are unnormalized scores in [0, ~1]; log + softmax renormalizes them
        # into proper distributions (T=1 reduces to plain normalization).
        # Computed in float32: log of tiny probabilities underflows in fp16.
        eps = 1e-8
        temp = self.kd_temperature

        z_student = torch.log(student_probs.float().clamp_min(eps))
        z_teacher = torch.log(teacher_probs.float().clamp_min(eps))

        log_p_student = F.log_softmax(z_student / temp, dim=1)
        log_p_teacher = F.log_softmax(z_teacher / temp, dim=1)
        p_teacher = log_p_teacher.exp()

        kl = (p_teacher * (log_p_teacher - log_p_student)).sum(dim=1)

        return kl.mean() * (temp * temp)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch

        mask_logits_per_block, class_logits_per_block = self(imgs)

        with torch.no_grad():
            teacher_mask_list, teacher_class_list = self.teacher(imgs / 255.0)
            teacher_probs = self.to_per_pixel_logits_semantic(
                teacher_mask_list[-1], teacher_class_list[-1]
            )

        losses_all_blocks = {}
        kd_total = None
        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_block, class_logits_per_block))
        ):
            losses = self.criterion(
                masks_queries_logits=mask_logits,
                class_queries_logits=class_logits,
                targets=targets,
            )
            block_postfix = self.block_postfix(i)
            losses = {f"{key}{block_postfix}": value for key, value in losses.items()}
            losses_all_blocks |= losses

            student_probs = self.to_per_pixel_logits_semantic(mask_logits, class_logits)
            if student_probs.shape[-2:] != teacher_probs.shape[-2:]:
                student_probs = F.interpolate(
                    student_probs, teacher_probs.shape[-2:], mode="bilinear"
                )
            kd = self._kd_loss(student_probs, teacher_probs)

            self.log(f"losses/train_loss_kd{block_postfix}", kd, sync_dist=True)
            kd_total = kd if kd_total is None else kd_total + kd

        loss_gt = self.criterion.loss_total(losses_all_blocks, self.log)
        loss_total = loss_gt + self.kd_coefficient * kd_total

        self.log("losses/train_loss_kd_total", kd_total, sync_dist=True)
        self.log(
            "losses/train_loss_total_with_kd", loss_total, sync_dist=True, prog_bar=True
        )

        return loss_total
