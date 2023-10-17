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

from mmpretrain.datasets import MultiHDF5Dataset, PackInputs, RandomResizedCrop
dataset_type = MultiHDF5Dataset 
data_root = '/arc/projects/unions/ssl/data/processed/unions-cutouts/ugriz_lsb/10k_per_h5/valid2'

from mmcv.transforms import RandomFlip
from mmpretrain.evaluation import Accuracy

train_pipeline = [
    dict(
        type=RandomResizedCrop,
        scale=32,
        crop_ratio_range=(0.2, 1.0),
        interpolation='bicubic'),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackInputs) 
]

pipeline = dict(batch_size=128, dataset=dict(type=dataset_type, # pipelines to be the same?
        data_root=data_root,
        pipeline=train_pipeline))
val_dataloader = dict(batch_size=128, dataset=dict(type=dataset_type,
        data_root=data_root,
        pipeline=train_pipeline))
test_dataloader = val_dataloader
train_dataloader = None

test_cfg = dict(type='TestLoop') 
val_cfg = dict(type='ValLoop')

test_evaluator = dict(type=Accuracy) # need to change to loss
val_evaluator = test_evaluator