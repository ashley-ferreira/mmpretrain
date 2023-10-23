from mmpretrain.datasets import MultiHDF5Dataset, PackInputs, RandomResizedCrop, BEiTMaskGenerator
from mmcv.transforms import RandomFlip, CenterCrop

# dataset settings
dataset_type = MultiHDF5Dataset
data_root = '/arc/projects/unions/ssl/data/processed/unions-cutouts/ugriz_lsb/10k_per_h5/valid2'

train_pipeline = [
    dict(type=CenterCrop, crop_size=32),
    dict(type=PackInputs) 
]

train_dataloader = dict(
    batch_size=32*4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=train_pipeline))