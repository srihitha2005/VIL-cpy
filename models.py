# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for Joint-FT
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
import torch.nn as nn
from timm.models._registry import register_model
from vision_transformer import _create_vision_transformer
from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
import torch
import timm

__all__ = [
    'vit_base_patch16_224',
    'vit_base_patch16_clip_224.openai',
    'vit_b16_in21k'
]

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

@register_model
def vit_base_patch16_224_biovit(pretrained=True, **kwargs):
    """ ViT-Base (ViT-B/16) pretrained on chest X-rays (BioViT / ViT-MAE). """
    
    default_cfg = _cfg(
        url=None,  # Don't use URL for local load
        custom_load=False,  # Disable auto download
    )
    kwargs.update(pretrained_cfg=default_cfg)

    model_args = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        **kwargs
    )

    # Create the architecture
    model = _create_vision_transformer(
        'vit_base_patch16_224', pretrained=False, **dict(model_args, **kwargs)
    )

    # ✅ Manually load your local .pth file here
    checkpoint_path = "biovil_vit_b16_timm.pth"  # Your local path
    state_dict = torch.load(checkpoint_path, map_location='cuda')
    model.load_state_dict(state_dict, strict=False)

    return model

@register_model
def vit_base_patch16_224_in1k(pretrained=True, **kwargs):
    """ ViT-Base (ViT-B/16) pretrained on ImageNet-1k """

    default_cfg = _cfg(url=None, custom_load=False)
    kwargs.update(pretrained_cfg=default_cfg)

    model_args = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        **kwargs
    )

    model = _create_vision_transformer(
        'vit_base_patch16_224', pretrained=False, **dict(model_args, **kwargs)
    )

    if pretrained:
        # Load pretrained model with default head (1000 classes)
        pretrained_model = timm.create_model('vit_base_patch16_224', pretrained=True)

        # Filter out the classifier head
        pretrained_dict = pretrained_model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('head')}

        # Load only matching keys
        model.load_state_dict(pretrained_dict, strict=False)

    return model

@register_model
def vit_base_patch16_224_in21k(pretrained=True, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    default_cfg = _cfg(
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz',
        custom_load=True,)
    kwargs.update(pretrained_cfg=default_cfg)

    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)

    model = _create_vision_transformer(
        'vit_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_base_patch16_224(pretrained=True, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    default_cfg = _cfg(
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz',
        custom_load=True,)
    kwargs.update(pretrained_cfg=default_cfg)

    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)

    model = _create_vision_transformer(
        'vit_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_base_patch16_clip_224_openai(pretrained=True, **kwargs):
    """ ViT-B/16 CLIP image tower, OpenAI original weights
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, pre_norm=True, norm_layer=nn.LayerNorm)
    
    model = _create_vision_transformer(
        'vit_base_patch16_clip_224.openai', pretrained=pretrained, **dict(model_args, **kwargs))
    return model 
