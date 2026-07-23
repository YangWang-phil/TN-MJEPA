"""Transformer, pooling, and continuous Fourier position primitives."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


def build_mlp(dim: int, hidden_dim: int, out_dim: Optional[int] = None, dropout: float = 0.0) -> nn.Sequential:
    out_dim = dim if out_dim is None else out_dim
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, length, dim = x.shape
        qkv = self.qkv(x).reshape(bsz, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_bias is not None:
            attn = attn + attn_bias
        if key_padding_mask is not None:
            mask = ~key_padding_mask[:, None, None, :]
            attn = attn.masked_fill(mask, float("-inf"))
        attn_prob = F.softmax(attn, dim=-1)
        attn_prob = self.dropout(attn_prob)
        out = torch.matmul(attn_prob, v).transpose(1, 2).reshape(bsz, length, dim)
        out = self.proj(out)
        return out, attn_prob


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = build_mlp(dim, hidden_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        feature_modulation: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.norm1(x)
        if feature_modulation is not None:
            gamma, beta = feature_modulation
            y = gamma * y + beta
        attn_out, attn_prob = self.attn(y, key_padding_mask=key_padding_mask, attn_bias=attn_bias)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x, attn_prob


def apply_transformer_blocks(
    blocks: nn.ModuleList,
    tokens: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor] = None,
    attn_bias: Optional[torch.Tensor] = None,
    feature_modulation: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_checkpoint: bool = False,
    collect_attn: bool = False,
):
    """Runs a stack of TransformerBlocks, optionally with gradient checkpointing.

    Checkpointing trades recompute for activation memory: with real sequences
    still reaching hundreds of tokens per modality (and thousands more once
    concatenated in fusion), keeping every layer's attention activations
    around for backward multiplies memory by depth. Only applied in training
    mode (checkpoint is a no-op / wasteful under no_grad, e.g. target encoders).

    ``collect_attn=True`` additionally returns the per-layer attention
    probability maps (for visualization/analysis only -- incompatible with
    checkpointing, so intended for eval-mode forward passes).
    """
    attn_maps = [] if collect_attn else None
    for block in blocks:
        if use_checkpoint and torch.is_grad_enabled() and tokens.requires_grad:
            tokens, _ = grad_checkpoint(block, tokens, key_padding_mask, attn_bias, feature_modulation, use_reentrant=False)
        else:
            tokens, attn_prob = block(tokens, key_padding_mask=key_padding_mask, attn_bias=attn_bias, feature_modulation=feature_modulation)
            if collect_attn:
                attn_maps.append(attn_prob)
    if collect_attn:
        return tokens, attn_maps
    return tokens


class AttentionPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, return_weights: bool = False):
        bsz = x.shape[0]
        query = self.query.expand(bsz, -1, -1)
        attn = torch.matmul(query, x.transpose(-2, -1)) / math.sqrt(x.shape[-1])
        if mask is not None:
            attn = attn.masked_fill(~mask[:, None, :], float("-inf"))
        weight = torch.softmax(attn, dim=-1)
        pooled = torch.matmul(weight, x).squeeze(1)
        pooled = self.proj(pooled)
        if return_weights:
            return pooled, weight.squeeze(1)  # B, N
        return pooled


def continuous_fourier_features(values: torch.Tensor, num_bands: int) -> torch.Tensor:
    values = values.unsqueeze(-1)
    freqs = torch.arange(1, num_bands + 1, device=values.device, dtype=values.dtype)
    angles = 2 * math.pi * values * freqs
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class ContinuousPositionProjector(nn.Module):
    def __init__(self, dim: int, num_bands: int) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.proj = nn.Linear(num_bands * 4, dim)

    def forward(self, time_axis: torch.Tensor, freq_axis: Optional[torch.Tensor] = None) -> torch.Tensor:
        time_feat = continuous_fourier_features(time_axis, self.num_bands)
        if freq_axis is None:
            freq_feat = torch.zeros_like(time_feat)
        else:
            freq_feat = continuous_fourier_features(freq_axis, self.num_bands)
        feat = torch.cat([time_feat, freq_feat], dim=-1)
        return self.proj(feat)
