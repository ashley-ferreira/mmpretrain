from mmpretrain.datasets import MultiHDF5Dataset, PackInputs, RandomResizedCrop
from mmcv.transforms import RandomFlip

# dataset settings
dataset_type = MultiHDF5Dataset
data_root = '/arc/projects/unions/ssl/data/processed/unions-cutouts/ugriz_lsb/10k_per_h5/valid'

train_pipeline = [
    dict(
        type=RandomResizedCrop,
        scale=144,
        crop_ratio_range=(0.2, 1.0),
        interpolation='bicubic'),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackInputs)
]

train_dataloader = dict(
    batch_size=4,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=train_pipeline))