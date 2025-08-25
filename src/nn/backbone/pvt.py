'''
Pyramid Vision Transformer (PVT) backbone implementation
Based on: https://arxiv.org/abs/2102.12122
by lyuwenyu
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from collections import OrderedDict
import math

from .common import get_activation
from src.core import register

__all__ = ['PVT']


# PVT model configurations
PVT_CONFIGS = {
    'pvt_tiny': {
        'patch_size': 4, 'embed_dims': [64, 128, 320, 512], 'num_heads': [1, 2, 5, 8],
        'mlp_ratios': [8, 8, 4, 4], 'qkv_bias': True, 'norm_layer': partial(nn.LayerNorm, eps=1e-6),
        'depths': [2, 2, 2, 2], 'sr_ratios': [8, 4, 2, 1], 'drop_rate': 0.0, 'drop_path_rate': 0.1
    },
    'pvt_small': {
        'patch_size': 4, 'embed_dims': [64, 128, 320, 512], 'num_heads': [1, 2, 5, 8],
        'mlp_ratios': [8, 8, 4, 4], 'qkv_bias': True, 'norm_layer': partial(nn.LayerNorm, eps=1e-6),
        'depths': [3, 4, 6, 3], 'sr_ratios': [8, 4, 2, 1], 'drop_rate': 0.0, 'drop_path_rate': 0.1
    },
    'pvt_medium': {
        'patch_size': 4, 'embed_dims': [64, 128, 320, 512], 'num_heads': [1, 2, 5, 8],
        'mlp_ratios': [8, 8, 4, 4], 'qkv_bias': True, 'norm_layer': partial(nn.LayerNorm, eps=1e-6),
        'depths': [3, 4, 18, 3], 'sr_ratios': [8, 4, 2, 1], 'drop_rate': 0.0, 'drop_path_rate': 0.1
    },
    'pvt_large': {
        'patch_size': 4, 'embed_dims': [64, 128, 320, 512], 'num_heads': [1, 2, 5, 8],
        'mlp_ratios': [8, 8, 4, 4], 'qkv_bias': True, 'norm_layer': partial(nn.LayerNorm, eps=1e-6),
        'depths': [3, 8, 27, 3], 'sr_ratios': [8, 4, 2, 1], 'drop_rate': 0.0, 'drop_path_rate': 0.1
    }
}


download_url = {
    'pvt_tiny': 'https://github.com/whai362/PVT/releases/download/v2/pvt_tiny.pth',  # 用户填写
    'pvt_small': 'https://github.com/whai362/PVT/releases/download/v2/pvt_small.pth',  # 用户填写
    'pvt_medium': 'https://github.com/whai362/PVT/releases/download/v2/pvt_medium.pth',  # 用户填写
    'pvt_large': 'https://github.com/whai362/PVT/releases/download/v2/pvt_large.pth',  # 用户填写
}


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        
        # Fix padding calculation to avoid size mismatch
        # For proper stride alignment, use padding that ensures output size = input_size // stride
        if patch_size[0] == stride and patch_size[1] == stride:
            # When patch_size == stride, no padding needed for exact division
            padding = (0, 0)
        else:
            # Use minimal padding for overlapping patches
            padding = ((patch_size[0] - stride) // 2, (patch_size[1] - stride) // 2)
            
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


@register
class PVT(nn.Module):
    def __init__(self, variant='pvt_small', img_size=224, patch_size=4, in_chans=3, num_classes=1000, 
                 embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], 
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], 
                 num_stages=4, return_idx=[0, 1, 2, 3], pretrained=False, freeze_at=-1, freeze_norm=True):
        super().__init__()
        self.variant = variant
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.return_idx = return_idx

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        # Build stages
        self.patch_embeds = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_stages):
            # Patch embedding
            if i == 0:
                patch_embed = OverlapPatchEmbed(img_size=img_size, patch_size=4, stride=4, in_chans=in_chans,
                                                embed_dim=embed_dims[i])
            else:
                # Fix: Use dynamic img_size based on previous stage output
                # For stage i, the input size is img_size // (4 * 2^(i-1))
                current_size = img_size // (4 * (2 ** (i - 1)))
                patch_embed = OverlapPatchEmbed(img_size=current_size, patch_size=2, stride=2,
                                                in_chans=embed_dims[i - 1], embed_dim=embed_dims[i])
            self.patch_embeds.append(patch_embed)

            # Transformer blocks
            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                norm_layer=norm_layer, sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            self.blocks.append(block)
            self.norms.append(norm_layer(embed_dims[i]))
            cur += depths[i]

        # Set output channels based on embed_dims
        self.out_channels = [embed_dims[i] for i in return_idx]
        self.out_strides = [4 * (2 ** i) for i in return_idx]

        # Freeze parameters if needed
        if freeze_at >= 0:
            for i in range(min(freeze_at + 1, num_stages)):
                self._freeze_parameters(self.patch_embeds[i])
                self._freeze_parameters(self.blocks[i])
                self._freeze_parameters(self.norms[i])

        if freeze_norm:
            self._freeze_norm(self)

        if pretrained:
            if download_url[variant]:
                state = torch.hub.load_state_dict_from_url(download_url[variant])
                self.load_state_dict(state, strict=False)
                print(f'Load PVT-{variant} state_dict')
            else:
                print(f'Warning: No download URL provided for {variant}')

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
        else:
            for child in m.children():
                self._freeze_norm(child)



    def forward_features(self, x):
        """Forward features extraction"""
        outs = []
        B = x.shape[0]

        for i in range(self.num_stages):
            # Patch embedding
            x, H, W = self.patch_embeds[i](x)
            
            # Transformer blocks
            for blk in self.blocks[i]:
                x = blk(x, H, W)
            x = self.norms[i](x)
            
            # Reshape to feature map format
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            
            # Collect output if this stage is in return_idx
            if i in self.return_idx:
                outs.append(x)

        return outs

    def forward(self, x):
        """Forward pass"""
        return self.forward_features(x)


def build_pvt(variant='pvt_small', **kwargs):
    """Build PVT model with specific configuration"""
    if variant not in PVT_CONFIGS:
        raise ValueError(f"Unknown PVT variant: {variant}. Available: {list(PVT_CONFIGS.keys())}")
    
    config = PVT_CONFIGS[variant].copy()
    config.update(kwargs)
    config['variant'] = variant
    return PVT(**config)


# Register PVT models
@register
def pvt_tiny(pretrained=False, **kwargs):
    """PVT-Tiny model"""
    return build_pvt('pvt_tiny', pretrained=pretrained, **kwargs)

@register
def pvt_small(pretrained=False, **kwargs):
    """PVT-Small model"""
    return build_pvt('pvt_small', pretrained=pretrained, **kwargs)

@register
def pvt_medium(pretrained=False, **kwargs):
    """PVT-Medium model"""
    return build_pvt('pvt_medium', pretrained=pretrained, **kwargs)

@register
def pvt_large(pretrained=False, **kwargs):
    """PVT-Large model"""
    return build_pvt('pvt_large', pretrained=pretrained, **kwargs)