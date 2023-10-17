# Copyright (c) OpenMMLab. All rights reserved.
# This is a BETA new format config file, and the usage may change recently.
from mmengine.config import read_base

with read_base():
    from .._base_.models.mae_vit_base_p16_unions import *
    from .._base_.datasets.unions_mae import *
    from .._base_.default_runtime import *

from mmengine.hooks.checkpoint_hook import CheckpointHook
from mmengine.optim.optimizer.amp_optimizer_wrapper import AmpOptimWrapper
from mmengine.optim.scheduler.lr_scheduler import CosineAnnealingLR, LinearLR
from mmengine.runner.loops import EpochBasedTrainLoop
from torch.optim.adamw import AdamW

# optimizer wrapper
optim_wrapper = dict(
    type='OptimWrapper',
    #loss_scale='dynamic',
    optimizer=dict(
        type=AdamW,
        lr=1.5e-5 * 32 / 256, 
        betas=(0.9, 0.95),
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'ln': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.),
            'mask_token': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.)
        }))

# learning rate scheduler
param_scheduler = [ # base lr is increasing when I thought it should be decreasing, this may cause that
    dict(
        type=LinearLR,
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=20,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        T_max=360,
        by_epoch=True,
        begin=20,
        end=400,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=100)
# only keeps the latest 3 checkpoints
default_hooks.checkpoint = dict(
    type=CheckpointHook, interval=1, max_keep_ckpts=3)

randomness.update(seed=0, diff_rank_seed=True)

# auto resume
resume = True

work_dir = '/arc/projects/unions/ssl/results/training_runs/MAEViT_Oct16_8'

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=32)