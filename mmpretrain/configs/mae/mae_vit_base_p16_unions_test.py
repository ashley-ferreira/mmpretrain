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
data_root_val = '/arc/projects/unions/ssl/data/processed/unions-cutouts/ugriz_lsb/10k_per_h5/val'


from mmcv.transforms import RandomFlip
from mmpretrain.evaluation import Accuracy

train_pipeline = [
    dict(type=CenterCrop, crop_size=32),
    dict(type=PackInputs) 
]

train_dataloader = dict(batch_size=32*4, dataset=dict(type=dataset_type, 
        data_root=data_root,
        pipeline=train_pipeline))

val_dataloader = dict(batch_size=32*4, dataset=dict(type=dataset_type,
        data_root=data_root_val,
        pipeline=train_pipeline))

test_dataloader = val_dataloader

test_cfg = dict(type='TestLoop') 
val_cfg = dict(type='ValLoop')

test_evaluator = dict(type=PixelReconstructionLoss, criterion='L2')
val_evaluator = test_evaluator