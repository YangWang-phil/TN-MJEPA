"""JEPA predictor with modality-specific continuous positions."""

from __future__ import annotations


import torch
import torch.nn as nn

from pretrain.configs import ModelConfig
from pretrain.models.common import ContinuousPositionProjector, TransformerBlock, apply_transformer_blocks


class JEPAPredictor(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.mask_token = nn.Parameter(torch.randn(1, 1, cfg.embed_dim) * 0.02)
        self.pos_proj = ContinuousPositionProjector(cfg.embed_dim, cfg.max_fourier_bands) if cfg.predictor_use_position else None
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.predictor_depth)]
        )
        self.norm = nn.LayerNorm(cfg.embed_dim)

    def forward(
        self,
        context_tokens: torch.Tensor,
        visible_mask: torch.Tensor,
        target_mask: torch.Tensor,
        time_pos: torch.Tensor,
    ) -> torch.Tensor:
        mask_token = self.mask_token.expand(context_tokens.shape[0], context_tokens.shape[1], -1)
        tokens = torch.where(target_mask.unsqueeze(-1), mask_token, context_tokens)
        if self.pos_proj is not None:
            tokens = tokens + self.pos_proj(time_pos, None)
        key_padding_mask = visible_mask | target_mask
        tokens = apply_transformer_blocks(self.blocks, tokens, key_padding_mask=key_padding_mask, use_checkpoint=self.cfg.use_grad_checkpoint)
        return self.norm(tokens)
