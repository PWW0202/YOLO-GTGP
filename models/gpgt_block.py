# ------------------------------------------------------------------------------
# 🚀 Hybrid Detection Blocks Toolkit for PyTorch & Ultralytics YOLO
# ------------------------------------------------------------------------------
# This module integrates two high-performance feature extraction components
# designed for Tunnel Geological Prediction using GPR (YOLO-GTGP):
# 1. GhostRepLite (GRL): A lightweight re-parameterized feature extraction block.
# 2. DualPath_CNNTransformer (DPCT): A hybrid CNN-Transformer block.
#
# Optimized for real-time object detection models (e.g., YOLOv8 / YOLO11).
# ------------------------------------------------------------------------------

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import Bottleneck

__all__ = [
    'RepConvN', 'GhostRepLite', 
    'DropPath', 'LayerNorm2d', 'ConvolutionalGLU', 'PSA_Attention', 
    'MHSA_CGLU', 'PartiallyTransformerBlock', 'DualPath_CNNTransformer'
]

# ==============================================================================
# SECTION 1: GhostRepLite Components (Reparameterized Lightweight Block)
# ==============================================================================

class RepConvN(nn.Module):
    """
    Structural Re-parameterization Convolution Block.
    
    Employs a multi-branch design (3x3 conv + 1x1 conv + identity) during training 
    to enrich gradient diversity and feature representation. Fuses branches into 
    a single 3x3 convolution during deployment for hardware-native acceleration.
    """
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1, "RepConvN strictly supports 3x3 kernels with padding 1"
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = None
        # Multi-branch training status
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Inference forward pass using the fused single convolution."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward pass adaptively accommodating training and fused statuses."""
        if hasattr(self, 'conv'):
            return self.forward_fuse(x)
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Derives equivalent 3x3 kernel weights and bias tensors by fusing all branches."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 kernel tensor symmetrically into a 3x3 kernel spatial layout."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Fuses BatchNorm running statistics into convolutional weight and bias tensors."""
        if branch is None:
            return 0, 0 
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        """
        Transforms the multi-branch graph into a clean, unified standard Conv2d layer.
        Essential procedure prior to ONNX/TensorRT model export.
        """
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                              out_channels=self.conv1.conv.out_channels,
                              kernel_size=self.conv1.conv.kernel_size,
                              stride=self.conv1.conv.stride,
                              padding=self.conv1.conv.padding,
                              dilation=self.conv1.conv.dilation,
                              groups=self.conv1.conv.groups,
                              bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')


class GhostRepLite(nn.Module):
    """
    GhostRepLite (GRL): A lightweight, re-parameterized feature extraction block.
    
    Synthesizes CSP gradient splitting, ELAN feature aggregation multi-paths, 
    and RepConvN structural re-parameterization to establish excellent 
    accuracy-to-speed tradeoffs, specifically optimized for GPR anomaly detection.
    """

    def __init__(self, c1, c2, n=1, scale=0.5, e=0.5):
        super(GhostRepLite, self).__init__()
        
        self.c = int(c2 * e)  # Hidden channels
        self.mid = int(self.c * scale)
        
        # CSP Stage: channel reduction and splitting
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        # Final channel fusion layer
        self.cv2 = Conv(self.c + self.mid * (n + 1), c2, 1)
        
        # Structural Re-parameterization backbone branch
        self.cv3 = RepConvN(self.c, self.mid, 3)
        # Dense residual-like aggregation paths
        self.m = nn.ModuleList(Conv(self.mid, self.mid, 3) for _ in range(n - 1))
        # Internal transition bottleneck
        self.cv4 = Conv(self.mid, self.mid, 1)

    def forward(self, x):
        """Standard execution flow utilizing torch.chunk for feature partitioning."""
        y = list(self.cv1(x).chunk(2, 1))
        y[-1] = self.cv3(y[-1])
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.cv4(y[-1]))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Hardware-friendly execution flow utilizing torch.split for optimized deployment."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y[-1] = self.cv3(y[-1])
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.cv4(y[-1]))
        return self.cv2(torch.cat(y, 1))


# ==============================================================================
# SECTION 2: DualPath_CNNTransformer Components (Hybrid CNN-Transformer Block)
# ==============================================================================

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample for regularization.
    Safe for spatial feature tensors.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # Binarize tensor map
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Stochastic depth regularization module for main residual pathways."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm2d(nn.LayerNorm):
    """
    2D spatial Layer Normalization variant tailored for (B, C, H, W) layout.
    """
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class ConvolutionalGLU(nn.Module):
    """
    Convolutional Gated Linear Unit (CGLU).
    Combines channel gating with a 3x3 depthwise convolution to enforce spatial bias.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True, groups=hidden_features),
            act_layer()
        )
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x_shortcut = x
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.dwconv(x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x_shortcut + x


class PSA_Attention(nn.Module):
    """
    Pyramid Spatial / Position-Sensitive Attention Block.
    Implements multi-head global attention augmented with positional encoding.
    """
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)  # Local positional encoding

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class MHSA_CGLU(nn.Module):
    """
    Global Context Transformer Branch combining PSA Attention and Convolutional GLU.
    """
    def __init__(self, inc, drop_path=0.1):
        super().__init__()
        self.norm1 = LayerNorm2d(inc)
        self.norm2 = LayerNorm2d(inc)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = ConvolutionalGLU(inc)
        self.mhsa = PSA_Attention(inc, num_heads=8)

    def forward(self, x):
        shortcut = x
        x = self.drop_path(self.mhsa(self.norm1(x))) + shortcut
        x = self.drop_path(self.mlp(self.norm2(x))) + x
        return x


class PartiallyTransformerBlock(nn.Module):
    """
    Partially Transformer Block (PTB).
    Splits the channels concurrently into local standard CNN paths and global Attention paths.
    """
    def __init__(self, c, tcr, shortcut=True):
        super().__init__()
        self.t_ch = int(c * tcr)       # Channels for Transformer path
        self.c_ch = c - self.t_ch      # Channels for CNN path
        
        self.c_b = Bottleneck(self.c_ch, self.c_ch, shortcut=shortcut)
        self.t_b = MHSA_CGLU(self.t_ch)
        self.conv_fuse = Conv(c, c)
    
    def forward(self, x):
        cnn_branch, transformer_branch = x.split((self.c_ch, self.t_ch), dim=1)
        cnn_branch = self.c_b(cnn_branch)
        transformer_branch = self.t_b(transformer_branch)
        return self.conv_fuse(torch.cat([cnn_branch, transformer_branch], dim=1))


class DualPath_CNNTransformer(nn.Module):
    """
    DualPath-CNNTransformer (DPCT): A hybrid architecture tailored for complex geological anomalies.
    Wraps the CNN-Transformer dual-pathway block in a CSP flow to optimize feature 
    allocation via the channel adjustment factor α.
    """
    def __init__(self, c1, c2, n=1, tcr=0.25, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # Hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(PartiallyTransformerBlock(self.c, tcr, shortcut=shortcut) for _ in range(n))

    def forward(self, x):
        """Forward pass using chunk partitioning."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split partitioning for ONNX optimization."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))