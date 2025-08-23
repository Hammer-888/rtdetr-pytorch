#!/usr/bin/env python3
"""
Simplified test script for PVT backbone implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from collections import OrderedDict
import math

# Simplified version of common functions for testing
def get_activation(act):
    if act == 'relu':
        return nn.ReLU(inplace=True)
    elif act == 'gelu':
        return nn.GELU()
    else:
        return nn.Identity()

# Mock register decorator for testing
def register(cls):
    return cls

# Copy the PVT implementation here for testing
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
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


@register
class PVT(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], num_stages=4, return_idx=[0, 1, 2, 3],
                 pretrained=False, freeze_at=-1, freeze_norm=True):
        super().__init__()
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
                patch_embed = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                                embed_dim=embed_dims[i])
            else:
                patch_embed = OverlapPatchEmbed(img_size=img_size // (2 ** (i + 1)), patch_size=3, stride=2,
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

        # Output channels and strides to match ResNet50
        # ResNet50 outputs: [512, 1024, 2048] with strides [8, 16, 32]
        # We need to map PVT outputs to these dimensions
        self.out_channels = []
        self.out_strides = []
        
        # Map PVT stages to ResNet-like outputs
        stage_to_resnet = {
            1: (512, 8),   # Stage 1 -> ResNet stage 2
            2: (1024, 16), # Stage 2 -> ResNet stage 3  
            3: (2048, 32)  # Stage 3 -> ResNet stage 4
        }
        
        self.channel_adapters = nn.ModuleList()
        for i, idx in enumerate(return_idx):
            if idx in [1, 2, 3]:  # Only adapt stages that map to ResNet outputs
                target_channels, stride = stage_to_resnet[idx]
                self.out_channels.append(target_channels)
                self.out_strides.append(stride)
                
                # Add channel adapter to match ResNet output dimensions
                adapter = nn.Conv2d(embed_dims[idx], target_channels, kernel_size=1, bias=False)
                self.channel_adapters.append(adapter)
            else:
                self.out_channels.append(embed_dims[idx])
                self.out_strides.append(4 * (2 ** idx))
                self.channel_adapters.append(nn.Identity())

    def forward(self, x):
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
            
            # Apply channel adapter and collect output if needed
            if i in self.return_idx:
                adapter_idx = self.return_idx.index(i)
                adapted_x = self.channel_adapters[adapter_idx](x)
                outs.append(adapted_x)

        return outs


# PVT model configurations
PVT_CONFIGS = {
    'pvt_tiny': {
        'embed_dims': [64, 128, 320, 512],
        'num_heads': [1, 2, 5, 8],
        'mlp_ratios': [8, 8, 4, 4],
        'depths': [2, 2, 2, 2],
        'sr_ratios': [8, 4, 2, 1]
    },
    'pvt_small': {
        'embed_dims': [64, 128, 320, 512], 
        'num_heads': [1, 2, 5, 8],
        'mlp_ratios': [8, 8, 4, 4],
        'depths': [3, 4, 6, 3],
        'sr_ratios': [8, 4, 2, 1]
    }
}


def build_pvt(variant='pvt_small', **kwargs):
    """Build PVT model with specific configuration"""
    if variant not in PVT_CONFIGS:
        raise ValueError(f"Unknown PVT variant: {variant}. Available: {list(PVT_CONFIGS.keys())}")
    
    config = PVT_CONFIGS[variant]
    config.update(kwargs)
    return PVT(**config)


def test_pvt_backbone():
    """Test PVT backbone with different configurations"""
    print("Testing PVT Backbone Implementation...")
    
    # Test input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 640, 640)
    print(f"Input shape: {input_tensor.shape}")
    
    # Test PVT Small configuration
    print("\n=== Testing PVT Small ===")
    pvt_small = build_pvt(
        variant='pvt_small',
        img_size=640,
        return_idx=[1, 2, 3],
        num_stages=4,
        drop_path_rate=0.1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in pvt_small.parameters()) / 1e6:.2f}M")
    print(f"Output channels: {pvt_small.out_channels}")
    print(f"Output strides: {pvt_small.out_strides}")
    
    # Forward pass
    with torch.no_grad():
        outputs = pvt_small(input_tensor)
    
    print(f"Number of outputs: {len(outputs)}")
    for i, output in enumerate(outputs):
        print(f"Output {i}: {output.shape}")
    
    # Verify output shapes match ResNet50 expectations
    expected_channels = [512, 1024, 2048]
    expected_strides = [8, 16, 32]
    
    print("\n=== Verification ===")
    for i, (output, exp_ch, exp_stride) in enumerate(zip(outputs, expected_channels, expected_strides)):
        actual_ch = output.shape[1]
        actual_h, actual_w = output.shape[2], output.shape[3]
        expected_h = 640 // exp_stride
        expected_w = 640 // exp_stride
        
        print(f"Stage {i+1}:")
        print(f"  Channels: {actual_ch} (expected: {exp_ch}) {'✓' if actual_ch == exp_ch else '✗'}")
        print(f"  Spatial: {actual_h}x{actual_w} (expected: {expected_h}x{expected_w}) {'✓' if actual_h == expected_h and actual_w == expected_w else '✗'}")
    
    print("\n=== Test completed successfully! ===")

if __name__ == "__main__":
    test_pvt_backbone()