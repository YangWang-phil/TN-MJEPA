"""Velocity-conditioned FiLM, attention bias, and frequency gating."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class SpeedGuidedAttention(nn.Module):
    def __init__(
        self,
        vel_dim: int,
        feat_dim: int,
        num_heads: int,
        num_freq_groups: int,
        use_film: bool = True,
        use_attn_bias: bool = True,
        use_freq_gate: bool = True,
    ) -> None:
        super().__init__()
        self.use_film = use_film
        self.use_attn_bias = use_attn_bias
        self.use_freq_gate = use_freq_gate
        self.num_heads = num_heads
        self.num_freq_groups = num_freq_groups

        self.default_speed = nn.Parameter(torch.zeros(1, 1, vel_dim))
        self.film = nn.Linear(vel_dim, feat_dim * 2) if use_film else None
        self.attn_proj = nn.Linear(vel_dim, num_heads) if use_attn_bias else None
        self.freq_gate = nn.Linear(vel_dim, num_freq_groups) if use_freq_gate else None

    def _with_default(self, speed_hidden: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if valid_mask is None:
            return speed_hidden
        default = self.default_speed.expand(speed_hidden.shape[0], speed_hidden.shape[1], -1)
        return torch.where(valid_mask.unsqueeze(-1), speed_hidden, default)

    def feature_modulation(
        self,
        speed_hidden: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not self.use_film or self.film is None:
            return None
        speed_hidden = self._with_default(speed_hidden, valid_mask)
        gamma_beta = self.film(speed_hidden)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = 1.0 + gamma
        return gamma, beta

    def attention_bias(self, speed_hidden: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        if not self.use_attn_bias or self.attn_proj is None:
            return None
        speed_hidden = self._with_default(speed_hidden, valid_mask)
        per_head = self.attn_proj(speed_hidden).transpose(1, 2)  # B, H, T
        return per_head.unsqueeze(-1) + per_head.unsqueeze(-2)

    def frequency_gate(
        self,
        speed_hidden: torch.Tensor,
        num_freq_groups: int,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if not self.use_freq_gate or self.freq_gate is None:
            return None
        speed_hidden = self._with_default(speed_hidden, valid_mask)
        logits = self.freq_gate(speed_hidden)
        if logits.shape[-1] != num_freq_groups:
            raise ValueError(f"Expected {num_freq_groups=} but got {logits.shape[-1]}")
        return torch.softmax(logits, dim=-1)
