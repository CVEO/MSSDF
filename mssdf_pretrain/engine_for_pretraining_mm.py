# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import sys
from typing import Iterable

import torch
import torch.nn as nn

import utils
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from loss import (
    ReconstructionLoss,
    ContrastiveAlignmentLoss,
    LinearHSICLoss,
    AuxiliaryClassificationLoss,
)


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    ema_decay=0.9999,
    log_writer=None,
    lr_scheduler=None,
    start_steps=None,
    lr_schedule_values=None,
    wd_schedule_values=None,
    loss_weights=[1.0, 0.5, 0.2, 0.1],
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "min_lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    reconstruct_loss = ReconstructionLoss()
    contrastive_alignment_loss = ContrastiveAlignmentLoss()
    linear_hsic_loss = LinearHSICLoss()
    auxiliary_classification_loss = AuxiliaryClassificationLoss()

    for step, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        dom, dsm, bool_masked_pos1, bool_masked_pos2 = batch
        dom = dom.to(device, non_blocking=True)
        dsm = dsm.to(device, non_blocking=True)
        bool_masked_pos1 = (
            bool_masked_pos1.to(device, non_blocking=True).flatten(1).to(torch.bool)
        )
        bool_masked_pos2 = (
            bool_masked_pos2.to(device, non_blocking=True).flatten(1).to(torch.bool)
        )

        # import pdb; pdb.set_trace()
        with torch.no_grad():
            dsm_labels = model.ema_teacher(dsm)
            dom_labels = model.ema_teacher(dom)

        with torch.cuda.amp.autocast():
            domtarget, dsmtarget, cls_pred, cls_target = model(
                dom, dsm, bool_masked_pos1, bool_masked_pos2
            )
            if cls_pred is not None:
                ce_loss = auxiliary_classification_loss(cls_pred, cls_target)
            else:
                ce_loss = 0

            reloss = reconstruct_loss(domtarget, dom_labels, bool_masked_pos1)
            reloss = reloss + reconstruct_loss(dsmtarget, dsm_labels, bool_masked_pos2)

            align_loss = contrastive_alignment_loss(domtarget, dsmtarget)
            hsic_loss = linear_hsic_loss(domtarget, dsmtarget)

            loss = (
                loss_weights[0] * reloss
                + loss_weights[1] * align_loss
                + loss_weights[2] * hsic_loss
                + loss_weights[3] * ce_loss
            )

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )
        grad_norm = loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=is_second_order,
        )
        loss_scale_value = loss_scaler.state_dict()["scale"]
        model.update_ema(ema_decay)
        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
