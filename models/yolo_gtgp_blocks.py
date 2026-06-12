import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv, RepConv
from ultralytics.nn.modules.block import Bottleneck

# ==============================================================================
# Utility Modules: Regularization & Normalization
# ==============================================================================

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) regularization per sample.
    Prevents co-adaptation of parallel paths in deep neural networks.
    """
    if drop_prob == 0. or not training: 
        return x
    keep_prob = 1 - drop_prob
    # Handle tensors with different dimensions, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # Binarize mask
    return x.div(keep_prob) * random_tensor

class DropPath(nn.Module):
    """Stochastic Depth regularization layer for Transformer blocks."""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x): 
        return drop_path(x, self.drop_prob, self.training)

class LayerNorm2d(nn.Module):
    """
    2D Layer Normalization optimized for channel-first image tensors (B, C, H, W).
    Stabilizes training by normalizing over the channel dimension.
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
    def forward(self, x): 
        # Permute to (B, H, W, C) for LayerNorm, then permute back to (B, C, H, W)
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

# ==============================================================================
# YOLO-GTGP Core Module 1: GhostRepLite (GRL)
# ==============================================================================

class GhostRepLite(nn.Module):
    """
    GhostRepLite (GRL) module for YOLO-GTGP.
    Objective: Efficient local feature extraction and multi-scale fusion for GPR 
    images. Utilizes structural re-parameterization (RepConv) and channel splitting
    to minimize computational overhead while preserving spatial details.
    """
    def __init__(self, c1, c2, n=1, scale=0.5, e=0.5):
        super().__init__()
        self.c = int(c2 * e)            # Hidden channel dimension
        self.mid = int(self.c * scale)  # Compressed channel width for lightweight branches
        
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(self.c + self.mid * (n + 1), c2, 1)
        self.cv3 = RepConv(self.c, self.mid, 3) # Re-parameterized convolution
        self.m = nn.ModuleList(Conv(self.mid, self.mid, 3) for _ in range(n - 1))
        self.cv4 = Conv(self.mid, self.mid, 1)

    def forward(self, x):
        # Split input features into two parallel branches
        y = list(self.cv1(x).chunk(2, 1))
        # Apply RepConv to the second branch
        y[-1] = self.cv3(y[-1])
        # Pass through sequential convolutional layers and collect hierarchical features
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.cv4(y[-1]))
        # Concatenate all branch outputs and project to target channel size
        return self.cv2(torch.cat(y, 1))

# ==============================================================================
# YOLO-GTGP Core Module 2: DualPath-CNNTransformer (DPCT) Components
# ==============================================================================

class PSA_Attention(nn.Module):
    """
    Polarized Self-Attention block.
    Captures long-range spatial and channel-wise dependencies, essential for 
    identifying global hyperbola signatures of geological anomalies in GPR B-scans.
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
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False) # Position Embedding

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        
        # Split into Query, Key, and Value matrices
        q, k, v = qkv.view(B, self.num_heads, self.key_dim*2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        
        # Calculate attention map and apply softmax scaling
        attn = ((q.transpose(-2, -1) @ k) * self.scale).softmax(dim=-1)
        
        # Aggregate features and add convolutional position embedding
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        return self.proj(x)

class ConvolutionalGLU(nn.Module):
    """
    Convolutional Gated Linear Unit (C-GLU).
    Acts as a Feed-Forward Network (FFN) utilizing depth-wise convolutions to 
    enhance localized texturing and non-linear feature representations.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features), 
            act_layer()
        )
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x_shortcut = x
        # Split into feature processing and gating pathways
        x, v = self.fc1(x).chunk(2, dim=1)
        # Apply gating mechanism
        x = self.dwconv(x) * v
        # Apply dropout and residual connection
        return x_shortcut + self.drop(self.fc2(self.drop(x)))

class MHSA_CGLU(nn.Module):
    """
    Metaformer-style block combining Multi-Head Self-Attention (MHSA) 
    and Convolutional GLU. Forms the core of the Transformer branch.
    """
    def __init__(self, inc, drop_path=0.1):
        super().__init__()
        self.norm1, self.norm2 = LayerNorm2d(inc), LayerNorm2d(inc)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp, self.mhsa = ConvolutionalGLU(inc), PSA_Attention(inc, num_heads=8)

    def forward(self, x):
        # Attention block with residual shortcut
        x = self.drop_path(self.mhsa(self.norm1(x))) + x
        # MLP block with residual shortcut
        return self.drop_path(self.mlp(self.norm2(x))) + x

class PartiallyTransformerBlock(nn.Module):
    """
    Partially Transformer Block (PTB).
    Splits feature channels into two parallel paths: a CNN path for local features 
    and a Transformer path for global semantics, significantly reducing computation 
    while maximizing representation capability.
    """
    def __init__(self, c, tcr, shortcut=True) -> None:
        super().__init__()
        self.t_ch = int(c * tcr)    # Channels allocated to the Transformer branch
        self.c_ch = c - self.t_ch   # Channels allocated to the CNN branch
        
        self.c_b = Bottleneck(self.c_ch, self.c_ch, shortcut=shortcut)
        self.t_b = MHSA_CGLU(self.t_ch)
        self.conv_fuse = Conv(c, c) # Fusion layer

    def forward(self, x):
        # Split features along the channel dimension
        cnn_branch, transformer_branch = x.split((self.c_ch, self.t_ch), 1)
        # Process and concatenate both representations
        return self.conv_fuse(torch.cat([self.c_b(cnn_branch), self.t_b(transformer_branch)], dim=1))

class DualPath_CNNTransformer(nn.Module):
    """
    DualPath-CNNTransformer (DPCT) module for YOLO-GTGP.
    Objective: Serves as the primary deep-stage feature extractor. Maintains a split 
    topology to simultaneously process local spatial details (via CNN) and global 
    geological contexts (via Transformers), crucial for anomaly detection.
    """
    def __init__(self, c1, c2, n=1, tcr=0.25, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e) 
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(PartiallyTransformerBlock(self.c, tcr, shortcut=shortcut) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        # Sequentially pass through Partially Transformer Blocks
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))