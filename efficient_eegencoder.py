"""
=============================================================================
 EfficientEEGEncoder — Proposed Modifications for Thesis
=============================================================================
 AUTHOR:  [Your Name]
 PURPOSE: An efficient variant of EEGEncoder that reduces memory and compute
          while targeting maintained/improved performance and subject independence.
          
 PROFESSOR'S DIRECTION:
   "Focus on building/proposing an efficient (less memory and compute) 
    transformer based model, while trying to improve the performance. 
    Remember, another target is subject independence."
   Reference: https://dl.acm.org/doi/full/10.1145/3530811

 THIS FILE IS ISOLATED FROM THE ORIGINAL CODEBASE.
 All modifications are marked with "# MODIFICATION:" comments.

 KEY CHANGES SUMMARY:
   1. Linear Attention      — O(n) instead of O(n²) attention complexity
   2. Reduced Branches      — 5 → 3 parallel branches (40% fewer params)
   3. Depthwise Separable   — Lighter convolutions in the feature extractor
   4. Shared Transformer    — 1 transformer shared across all branches
   5. Gradient Checkpointing — Lower GPU memory usage
=============================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# =====================================================================
# MODIFICATION 1: Linear Attention Mechanism
# =====================================================================
# RATIONALE: Standard self-attention has O(n²) time and memory complexity
# in sequence length. For EEG with long sequences, this is wasteful.
# Linear attention uses kernel feature maps φ(Q) and φ(K) to compute
# attention in O(n) time: Attn = φ(Q)(φ(K)ᵀV) instead of softmax(QKᵀ)V
#
# Reference: "Transformers are RNNs" (Katharopoulos et al., 2020)
# and "Efficient Transformers: A Survey" (Tay et al., 2022)
# =====================================================================

class LinearAttention(nn.Module):
    """
    Linear attention with ELU+1 feature map.
    Complexity: O(n·d²) instead of O(n²·d) — much faster for long sequences.
    
    Compared to original LlamaAttention which uses full O(n²) attention.
    """
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, \
            "hidden_size must be divisible by num_heads"
        
        # MODIFICATION: Using standard Linear instead of LinearL2 for cleaner code
        # You can swap back to LinearL2 if L2 regularization is desired
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def _elu_feature_map(self, x):
        """ELU+1 feature map: φ(x) = elu(x) + 1, ensures non-negative values."""
        # MODIFICATION: This is the kernel trick that makes attention linear.
        # Instead of softmax(QK^T), we compute φ(Q)·(φ(K)^T · V)
        return F.elu(x) + 1
    
    def forward(self, hidden_states):
        bsz, seq_len, _ = hidden_states.size()
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for multi-head: (bsz, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # MODIFICATION: Apply feature map instead of computing full attention matrix
        q = self._elu_feature_map(q)
        k = self._elu_feature_map(k)
        
        # Linear attention: O(n·d²) instead of O(n²·d)
        # Compute K^T V first: (bsz, heads, head_dim, head_dim)
        kv = torch.matmul(k.transpose(-2, -1), v)
        # Then Q(K^T V): (bsz, heads, seq_len, head_dim)
        qkv = torch.matmul(q, kv)
        
        # Normalization (equivalent to softmax denominator)
        z = torch.matmul(q, k.sum(dim=-2, keepdim=True).transpose(-2, -1))
        z = z.clamp(min=1e-6)  # Numerical stability
        
        attn_output = qkv / z
        attn_output = self.dropout(attn_output)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output


# =====================================================================
# RMSNorm (kept from original — it's already efficient)
# =====================================================================
class RMSNorm(nn.Module):
    """RMSNorm — same as original LlamaRMSNorm. Already efficient."""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(x.dtype)


# =====================================================================
# SwiGLU Feed-Forward (kept from original — it's already efficient)
# =====================================================================
class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network — same concept as original LlamaMLP.
    SwiGLU(x) = Swish(xW_gate) ⊙ (xW_up), then projected down.
    """
    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x))) * self.up_proj(x))


# =====================================================================
# Efficient Transformer Block (replaces LlamaDecoderLayer)
# =====================================================================
class EfficientTransformerBlock(nn.Module):
    """
    MODIFICATION: Replaces original LlamaDecoderLayer.
    
    Changes vs original:
    - Linear attention instead of full attention (O(n) vs O(n²))
    - Keeps Pre-Norm + RMSNorm + SwiGLU (these are already efficient)
    - No rotary embeddings (not needed for fixed-length EEG segments)
    - No causal masking (EEG classification doesn't need it)
    """
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.1):
        super().__init__()
        # MODIFICATION: Linear attention instead of full LlamaAttention
        self.self_attn = LinearAttention(hidden_size, num_heads, dropout)
        self.mlp = SwiGLUFFN(hidden_size, intermediate_size, dropout)
        # Pre-Norm (same as original)
        self.input_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)

    def forward(self, hidden_states):
        # Pre-Norm + Attention + Residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states
        
        # Pre-Norm + FFN + Residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


# =====================================================================
# Efficient Transformer (replaces LlamaForCausalLM)
# =====================================================================
class EfficientTransformer(nn.Module):
    """
    MODIFICATION: Lightweight transformer replacing LlamaForCausalLM.
    
    Changes vs original:
    - No vocabulary embedding (we pass continuous EEG features directly)
    - No language model head (we don't need token prediction)
    - Linear attention layers
    - Supports gradient checkpointing for memory savings
    """
    def __init__(self, hidden_size=32, num_heads=2, num_layers=2,
                 intermediate_size=32, dropout=0.1, use_gradient_ckpt=False):
        super().__init__()
        self.use_gradient_ckpt = use_gradient_ckpt
        self.layers = nn.ModuleList([
            EfficientTransformerBlock(hidden_size, num_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, hidden_size) — continuous EEG features
        Returns:
            (batch_size, seq_len, hidden_size) — transformed features
        """
        hidden_states = x
        for layer in self.layers:
            # MODIFICATION 5: Gradient checkpointing to reduce memory
            if self.use_gradient_ckpt and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, use_reentrant=False)
            else:
                hidden_states = layer(hidden_states)
        
        hidden_states = self.norm(hidden_states)
        return hidden_states


# =====================================================================
# MODIFICATION 3: Depthwise Separable Convolutions
# =====================================================================
# RATIONALE: Standard Conv2d with F1→F2 channels has F1×F2×K×1 params.
# Depthwise separable splits this into:
#   1. Depthwise conv: F1 groups, F1×1×K×1 params (spatial filtering)
#   2. Pointwise conv: F1×F2×1×1 params (channel mixing)
# Total: F1(K+F2) vs F1×F2×K — significantly fewer parameters
# =====================================================================

class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise separable convolution — fewer params than standard Conv2d."""
    def __init__(self, in_channels, out_channels, kernel_size, padding='same', bias=False):
        super().__init__()
        # MODIFICATION: Split into depthwise + pointwise
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                    padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
    
    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class EfficientConvBlock(nn.Module):
    """
    MODIFICATION: Lighter Downsampling Projector using depthwise separable convolutions.
    
    Changes vs original ConvBlock:
    - conv1 uses depthwise separable instead of standard Conv2d
    - conv2 uses depthwise separable instead of standard Conv2d
    - Same overall structure: Conv → BN → ELU → Pool → Dropout
    """
    def __init__(self, F1=16, kernLength=64, poolSize=7, D=2, in_chans=22, dropout=0.3):
        super().__init__()
        F2 = F1 * D
        # MODIFICATION: Depthwise separable for first conv (biggest kernel)
        self.conv1 = DepthwiseSeparableConv2d(1, F1, (kernLength, 1), padding='same')
        self.batchnorm1 = nn.BatchNorm2d(F1)
        # Spatial depthwise (same as original — already efficient)
        self.depthwise = nn.Conv2d(F1, F2, (1, in_chans), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F2)
        self.activation = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((8, 1))
        self.dropout1 = nn.Dropout(dropout)
        # MODIFICATION: Depthwise separable for second conv
        self.conv2 = DepthwiseSeparableConv2d(F2, F2, (16, 1), padding='same')
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d((poolSize, 1))
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        return x


# =====================================================================
# TCN Block (kept mostly the same — already efficient)
# =====================================================================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TCNBlock(nn.Module):
    """TCN Block — same as original TCNBlock_ (temporal convolutions are already efficient)."""
    def __init__(self, input_dim, depth, kernel_size, filters, dropout, activation='elu'):
        super().__init__()
        self.depth = depth
        self.activation_fn = getattr(F, activation)
        self.blocks = nn.ModuleList()
        self.downsample = nn.Conv1d(input_dim, filters, 1) if input_dim != filters else None
        self.cn1 = nn.Sequential(
            nn.Conv1d(input_dim, filters, kernel_size),
            nn.BatchNorm1d(filters), nn.SiLU(), nn.Dropout(0.3))
        self.cn2 = nn.Sequential(
            nn.Conv1d(filters, filters, kernel_size),
            nn.BatchNorm1d(filters), nn.SiLU(), nn.Dropout(0.3))

        for i in range(depth - 1):
            d = 2 ** (i + 1)
            pad = (kernel_size - 1) * d
            in_ch = filters if i > 0 else input_dim
            self.blocks.append(nn.Sequential(
                nn.Conv1d(in_ch, filters, kernel_size, padding=pad, dilation=d),
                Chomp1d(pad), nn.BatchNorm1d(filters), nn.ReLU(), nn.Dropout(dropout),
                nn.Conv1d(filters, filters, kernel_size, padding=pad, dilation=d),
                Chomp1d(pad), nn.BatchNorm1d(filters), nn.ReLU(), nn.Dropout(dropout)))

    def forward(self, x):
        out = x.transpose(1, 2)
        out = self.cn1(out)
        out = self.cn2(out)
        res = self.downsample(out) if self.downsample is not None else out
        for i, block in enumerate(self.blocks):
            out = block(out) + (res if i == 0 else self.blocks[i-1](res))
            out = self.activation_fn(out)
        return out.transpose(1, 2)


# =====================================================================
# MAIN MODEL: EfficientEEGEncoder
# =====================================================================

class EfficientEEGEncoder(nn.Module):
    """
    EfficientEEGEncoder — A lighter, more efficient variant of EEGEncoder.
    
    MODIFICATIONS vs Original EEGEncoder:
    ┌──────────────────────┬──────────────────────┬──────────────────────────────┐
    │ Component            │ Original             │ Efficient (Ours)             │
    ├──────────────────────┼──────────────────────┼──────────────────────────────┤
    │ Attention            │ Full O(n²) Llama     │ Linear O(n) attention        │
    │ Parallel Branches    │ 5                    │ 3 (configurable)             │
    │ Transformer per      │ 5 separate copies    │ 1 shared transformer         │
    │ branch               │                      │                              │
    │ ConvBlock            │ Standard Conv2d      │ Depthwise Separable Conv2d   │
    │ Rotary Embeddings    │ Yes (from Llama)     │ No (fixed-length EEG)        │
    │ Causal Mask          │ Yes (from Llama)     │ No (classification task)     │
    │ Grad Checkpoint      │ No                   │ Yes (optional)               │
    │ LM Head              │ Yes (unused)         │ No                           │
    └──────────────────────┴──────────────────────┴──────────────────────────────┘
    
    Args:
        n_classes: Number of MI classes (default: 4)
        n_branches: Number of parallel DSTS branches (default: 3)
        hidden_size: Transformer hidden size (default: 32)
        share_transformer: If True, all branches share one transformer (default: True)
        use_gradient_ckpt: If True, use gradient checkpointing (default: False)
    """
    def __init__(self, n_classes=4, in_chans=22, in_samples=1125,
                 # MODIFICATION 2: Reduced from 5 to 3 branches
                 n_branches=3,
                 hidden_size=32,
                 eegn_F1=16, eegn_D=2, eegn_kernelSize=64,
                 eegn_poolSize=7, eegn_dropout=0.3,
                 tcn_depth=2, tcn_kernelSize=4, tcn_filters=32,
                 tcn_dropout=0.3, tcn_activation='elu',
                 # MODIFICATION 4: Share transformer across branches
                 share_transformer=True,
                 # MODIFICATION 5: Gradient checkpointing
                 use_gradient_ckpt=False,
                 fuse='average'):
        super().__init__()
        self.n_branches = n_branches
        self.fuse = fuse
        self.share_transformer = share_transformer
        
        F2 = eegn_F1 * eegn_D  # = 32
        
        # MODIFICATION 3: Depthwise separable conv block
        self.conv_block = EfficientConvBlock(
            F1=eegn_F1, kernLength=eegn_kernelSize,
            poolSize=eegn_poolSize, D=eegn_D,
            in_chans=in_chans, dropout=eegn_dropout)
        
        # TCN blocks (one per branch — same as original)
        self.tcn_blocks = nn.ModuleList([
            TCNBlock(F2, tcn_depth, tcn_kernelSize, tcn_filters,
                     tcn_dropout, tcn_activation)
            for _ in range(n_branches)])
        
        # Dense layers (one per branch — same as original)
        self.dense_layers = nn.ModuleList([
            nn.Linear(tcn_filters, n_classes) for _ in range(n_branches)])
        
        self.branch_dropout = nn.Dropout(0.3)
        
        # MODIFICATION 4: Shared transformer — only ONE instead of n_branches copies
        if share_transformer:
            self.transformer = EfficientTransformer(
                hidden_size=F2, num_heads=2, num_layers=2,
                intermediate_size=F2, dropout=0.3,
                use_gradient_ckpt=use_gradient_ckpt)
        else:
            self.transformers = nn.ModuleList([
                EfficientTransformer(
                    hidden_size=F2, num_heads=2, num_layers=2,
                    intermediate_size=F2, dropout=0.3,
                    use_gradient_ckpt=use_gradient_ckpt)
                for _ in range(n_branches)])
        
        if fuse == 'concat':
            self.final_dense = nn.Linear(n_classes * n_branches, n_classes)
    
    def forward(self, x):
        # Downsampling projector
        x = self.conv_block(x)
        x = x[:, :, :, 0].permute(0, 2, 1)  # (batch, seq_len, hidden_size)
        
        branch_outputs = []
        for i in range(self.n_branches):
            # Apply branch dropout (ensemble effect)
            branch_input = self.branch_dropout(x)
            
            # TCN pathway → temporal features
            tcn_out = self.tcn_blocks[i](branch_input)
            tcn_out = tcn_out[:, -1, :]  # Take last timestep
            
            # MODIFICATION 4: Transformer pathway (shared or per-branch)
            if self.share_transformer:
                trm_out = self.transformer(branch_input).mean(1)
            else:
                trm_out = self.transformers[i](branch_input).mean(1)
            
            # Feature fusion: temporal + spatial
            fused = tcn_out + F.dropout(trm_out, 0.3, training=self.training)
            
            # Classification
            out = self.dense_layers[i](fused)
            branch_outputs.append(out)
        
        # Aggregate branches
        if self.fuse == 'average':
            out = torch.mean(torch.stack(branch_outputs, dim=0), dim=0)
        elif self.fuse == 'concat':
            out = torch.cat(branch_outputs, dim=1)
            out = self.final_dense(out)
        
        out = F.softmax(out, dim=1)
        return out


# =====================================================================
# UTILITY: Compare model sizes
# =====================================================================
def compare_models():
    """Print parameter counts for baseline vs efficient model."""
    from colab_train_baseline import EEGEncoder as BaselineModel
    
    baseline = BaselineModel()
    efficient = EfficientEEGEncoder()
    
    baseline_params = sum(p.numel() for p in baseline.parameters())
    efficient_params = sum(p.numel() for p in efficient.parameters())
    reduction = (1 - efficient_params / baseline_params) * 100
    
    print(f"{'='*50}")
    print(f"  MODEL COMPARISON")
    print(f"{'='*50}")
    print(f"  Baseline EEGEncoder:    {baseline_params:>10,} params")
    print(f"  Efficient EEGEncoder:   {efficient_params:>10,} params")
    print(f"  Reduction:              {reduction:>9.1f}%")
    print(f"{'='*50}")
    
    # Test inference speed
    import time
    x = torch.randn(16, 1, 22, 1125)
    
    baseline.eval()
    efficient.eval()
    
    with torch.no_grad():
        # Warmup
        _ = baseline(x)
        _ = efficient(x)
        
        # Baseline timing
        start = time.time()
        for _ in range(10):
            _ = baseline(x)
        baseline_time = (time.time() - start) / 10
        
        # Efficient timing
        start = time.time()
        for _ in range(10):
            _ = efficient(x)
        efficient_time = (time.time() - start) / 10
    
    speedup = baseline_time / efficient_time
    print(f"  Baseline inference:     {baseline_time*1000:>9.1f} ms/batch")
    print(f"  Efficient inference:    {efficient_time*1000:>9.1f} ms/batch")
    print(f"  Speedup:                {speedup:>9.2f}x")
    print(f"{'='*50}")


if __name__ == '__main__':
    # Quick sanity check
    model = EfficientEEGEncoder()
    params = sum(p.numel() for p in model.parameters())
    print(f"EfficientEEGEncoder parameters: {params:,}")
    
    # Test forward pass
    x = torch.randn(4, 1, 22, 1125)  # (batch, 1, channels, timepoints)
    out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")  # Should be (4, 4)
    print(f"Output sum:   {out.sum(dim=1)}")  # Should be ~1.0 (softmax)
    print("✅ Forward pass successful!")
