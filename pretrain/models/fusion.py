"""Cross-modal Transformer fusion and attention pooling."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from pretrain.configs import ModelConfig
from pretrain.models.common import AttentionPool, TransformerBlock, apply_transformer_blocks


class MultimodalFusionTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.modality_embeddings = nn.Embedding(4, cfg.embed_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.fusion_depth)]
        )
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.pool = AttentionPool(cfg.embed_dim)

    def forward(
        self,
        tf_tokens: torch.Tensor,
        tf_mask: torch.Tensor,
        wave_tokens: torch.Tensor,
        wave_mask: torch.Tensor,
        vel_tokens: torch.Tensor,
        vel_mask: torch.Tensor,
        meta_token: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        meta_mask = torch.ones(meta_token.shape[:2], dtype=torch.bool, device=meta_token.device)
        tokens = torch.cat(
            [
                tf_tokens + self.modality_embeddings.weight[0],
                wave_tokens + self.modality_embeddings.weight[1],
                vel_tokens + self.modality_embeddings.weight[2],
                meta_token + self.modality_embeddings.weight[3],
            ],
            dim=1,
        )
        mask = torch.cat([tf_mask, wave_mask, vel_mask, meta_mask], dim=1)
        attn_maps = None
        if return_attention:
            # Visualization path (eval only): collect per-layer attention maps and the
            # pooling weights so cross-modal attention flow can be plotted for the paper.
            tokens, attn_maps = apply_transformer_blocks(
                self.blocks, tokens, key_padding_mask=mask, use_checkpoint=False, collect_attn=True,
            )
        else:
            tokens = apply_transformer_blocks(self.blocks, tokens, key_padding_mask=mask, use_checkpoint=self.cfg.use_grad_checkpoint)
        tokens = self.norm(tokens)
        lengths = {
            "tf": tf_tokens.shape[1],
            "wave": wave_tokens.shape[1],
            "vel": vel_tokens.shape[1],
            "meta": meta_token.shape[1],
        }
        out: Dict[str, torch.Tensor] = {"mask": mask, "lengths": lengths}
        if return_attention:
            pooled, pool_weights = self.pool(tokens, mask, return_weights=True)
            out["attn_maps"] = attn_maps
            out["pool_weights"] = pool_weights
        else:
            pooled = self.pool(tokens, mask)
        out["tokens"] = tokens
        out["pooled"] = pooled
        return out


def split_modalities(tokens: torch.Tensor, lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:
    tf_end = lengths["tf"]
    wave_end = tf_end + lengths["wave"]
    vel_end = wave_end + lengths["vel"]
    return {
        "tf": tokens[:, :tf_end],
        "wave": tokens[:, tf_end:wave_end],
        "vel": tokens[:, wave_end:vel_end],
        "meta": tokens[:, vel_end:],
    }
