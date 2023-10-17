# Copyright (c) OpenMMLab. All rights reserved.
# This is a BETA new format config file, and the usage may change recently.
from mmpretrain.models import (MAE, MAEPretrainDecoder, MAEPretrainHead,
                               MAEViT, PixelReconstructionLoss)

img_size = 32 # needs to match the crop size defined in the training pipeline
patch_size = 4  # img_size needs to be divisible by patch_size
in_chans = 5
num_patches_x = img_size / patch_size
num_patches_y = img_size / patch_size
num_patches = int(num_patches_x * num_patches_y)

# model settings
model = dict(
    type=MAE, # we probably want a bit lower of a mask ratio in our case, with smaller patches
    backbone=dict(type=MAEViT, arch='b', patch_size=patch_size, mask_ratio=0.5, in_channels=in_chans, img_size=img_size),
    neck=dict(
        type=MAEPretrainDecoder,
        patch_size=patch_size,
        num_patches=num_patches,
        in_chans=in_chans,
        embed_dim=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
    ),
    head=dict(
        type=MAEPretrainHead,
        norm_pix=True,
        in_chans=in_chans,
        patch_size=patch_size,
        loss=dict(type=PixelReconstructionLoss, criterion='L2')),
    init_cfg=[
        dict(type='Xavier', layer='Linear', distribution='uniform'),
        dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
    ])