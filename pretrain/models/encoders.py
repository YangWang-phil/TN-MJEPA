"""TF, aligned pressure, velocity, and metadata encoders."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pretrain.configs import ModelConfig
from pretrain.metadata import MODEL_METADATA_SLICE
from pretrain.models.common import (
    AttentionPool,
    ContinuousPositionProjector,
    TransformerBlock,
    apply_transformer_blocks,
)
from pretrain.models.sga import SpeedGuidedAttention


def patch_average_1d(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    bsz, length = x.shape
    pad = (-length) % patch_size
    if pad:
        x = F.pad(x, (0, pad), value=0.0)
    x = x.reshape(bsz, -1, patch_size)
    return x.mean(dim=-1)


def patch_average_mask(mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    pooled = patch_average_1d(mask.float(), patch_size)
    return pooled > 0.5


class TFEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig, sga: Optional[SpeedGuidedAttention] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.sga = sga
        self.patch_embed = nn.Conv2d(1, cfg.embed_dim, kernel_size=(cfg.tf_patch_freq, cfg.tf_patch_time), stride=(cfg.tf_patch_freq, cfg.tf_patch_time))
        self.pos_proj = ContinuousPositionProjector(cfg.embed_dim, cfg.max_fourier_bands)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.tf_depth)]
        )
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.pool = AttentionPool(cfg.embed_dim)

    def forward(
        self,
        tf_log: torch.Tensor,
        freq_axis: torch.Tensor,
        time_axis: torch.Tensor,
        time_mask: torch.Tensor,
        speed_hidden: Optional[torch.Tensor] = None,
        speed_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        pad_time = (-tf_log.shape[-1]) % self.cfg.tf_patch_time
        if pad_time:
            tf_log = F.pad(tf_log, (0, pad_time), value=0.0)
        x = self.patch_embed(tf_log.unsqueeze(1))  # B, D, Fp, Tp
        bsz, dim, fp, tp = x.shape
        tokens = x.flatten(2).transpose(1, 2)

        freq_pooled = patch_average_1d(freq_axis, self.cfg.tf_patch_freq)
        time_pooled = patch_average_1d(time_axis, self.cfg.tf_patch_time)
        time_mask_pooled = patch_average_mask(time_mask, self.cfg.tf_patch_time)

        token_time = time_pooled[:, None, :].expand(-1, fp, -1).reshape(bsz, fp * tp)
        token_freq = freq_pooled[:, :, None].expand(-1, -1, tp).reshape(bsz, fp * tp)
        token_mask = time_mask_pooled[:, None, :].expand(-1, fp, -1).reshape(bsz, fp * tp)
        tokens = tokens + self.pos_proj(token_time, token_freq)

        attn_bias = None
        feature_modulation = None
        if self.sga is not None and speed_hidden is not None:
            speed_mask_src = (speed_valid_mask if speed_valid_mask is not None else time_mask).float().unsqueeze(1)
            speed_token_tp = F.interpolate(speed_hidden.transpose(1, 2), size=tp, mode="linear", align_corners=False).transpose(1, 2)
            speed_mask_tp = F.interpolate(speed_mask_src, size=tp, mode="nearest").squeeze(1) > 0.5

            speed_token = speed_token_tp[:, None, :, :].expand(-1, fp, -1, -1).reshape(bsz, fp * tp, -1)
            speed_mask = speed_mask_tp[:, None, :].expand(-1, fp, -1).reshape(bsz, fp * tp)
            feature_modulation = self.sga.feature_modulation(speed_token, speed_mask)
            attn_bias = self.sga.attention_bias(speed_token, speed_mask)

            freq_gate = self.sga.frequency_gate(speed_token_tp, self.sga.num_freq_groups, speed_mask_tp)
            if freq_gate is not None:
                freq_group = torch.linspace(0, self.sga.num_freq_groups - 1, steps=fp, device=tokens.device).long()
                gate = freq_gate[:, None, :, :].expand(-1, fp, -1, -1)
                gate = gate.gather(-1, freq_group[None, :, None, None].expand(bsz, fp, tp, 1)).squeeze(-1)
                gate = gate.reshape(bsz, fp * tp).unsqueeze(-1)
                tokens = tokens * gate

        tokens = apply_transformer_blocks(
            self.blocks, tokens, key_padding_mask=token_mask, attn_bias=attn_bias,
            feature_modulation=feature_modulation, use_checkpoint=self.cfg.use_grad_checkpoint,
        )
        tokens = self.norm(tokens)
        pooled = self.pool(tokens, token_mask)
        return {
            "tokens": tokens,
            "mask": token_mask,
            "pooled": pooled,
            "time_pos": token_time,
            "freq_pos": token_freq,
        }


def build_conv1d_stem(strides: list, out_dim: int) -> nn.Sequential:
    layers = []
    in_ch = 1
    for i, stride in enumerate(strides):
        out_ch = out_dim if i == len(strides) - 1 else max(out_dim // 2, 1)
        layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=9, stride=stride, padding=4))
        layers.append(nn.GELU())
        in_ch = out_ch
    return nn.Sequential(*layers)


class WaveEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig, sga: Optional[SpeedGuidedAttention] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.sga = sga
        self.stem = build_conv1d_stem(cfg.wave_stem_strides, cfg.embed_dim)
        self.pos_proj = ContinuousPositionProjector(cfg.embed_dim, cfg.max_fourier_bands)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.wave_depth)]
        )
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.pool = AttentionPool(cfg.embed_dim)

    def forward(
        self,
        pa_raw: torch.Tensor,
        time_axis: torch.Tensor,
        time_mask: torch.Tensor,
        speed_hidden: Optional[torch.Tensor] = None,
        speed_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x = self.stem(pa_raw.unsqueeze(1))
        tokens = x.transpose(1, 2)
        token_len = tokens.shape[1]
        time_axis = F.interpolate(time_axis.unsqueeze(1), size=token_len, mode="linear", align_corners=False).squeeze(1)
        mask = F.interpolate(time_mask.float().unsqueeze(1), size=token_len, mode="nearest").squeeze(1) > 0.5
        tokens = tokens + self.pos_proj(time_axis, None)

        attn_bias = None
        feature_modulation = None
        if self.sga is not None and speed_hidden is not None:
            speed_token = F.interpolate(speed_hidden.transpose(1, 2), size=token_len, mode="linear", align_corners=False).transpose(1, 2)
            speed_mask = F.interpolate((speed_valid_mask if speed_valid_mask is not None else time_mask).float().unsqueeze(1), size=token_len, mode="nearest").squeeze(1) > 0.5
            feature_modulation = self.sga.feature_modulation(speed_token, speed_mask)
            attn_bias = self.sga.attention_bias(speed_token, speed_mask)

        tokens = apply_transformer_blocks(
            self.blocks, tokens, key_padding_mask=mask, attn_bias=attn_bias,
            feature_modulation=feature_modulation, use_checkpoint=self.cfg.use_grad_checkpoint,
        )
        tokens = self.norm(tokens)
        pooled = self.pool(tokens, mask)
        return {"tokens": tokens, "mask": mask, "pooled": pooled, "time_pos": time_axis}


class VelEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.stem = build_conv1d_stem(cfg.vel_stem_strides, cfg.vel_hidden_dim)
        self.hidden_proj = nn.Linear(cfg.vel_hidden_dim, cfg.embed_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg.vel_hidden_dim, cfg.vel_num_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.vel_depth)]
        )
        self.norm = nn.LayerNorm(cfg.vel_hidden_dim)
        self.tokens_norm = nn.LayerNorm(cfg.embed_dim)
        self.pool = AttentionPool(cfg.embed_dim)
        self.pos_proj = ContinuousPositionProjector(cfg.vel_hidden_dim, cfg.max_fourier_bands)

    def forward(self, vel: torch.Tensor, time_axis: torch.Tensor, time_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(vel.unsqueeze(1)).transpose(1, 2)
        token_len = x.shape[1]
        time_axis_ds = F.interpolate(time_axis.unsqueeze(1), size=token_len, mode="linear", align_corners=False).squeeze(1)
        mask_ds = F.interpolate(time_mask.float().unsqueeze(1), size=token_len, mode="nearest").squeeze(1) > 0.5
        x = x + self.pos_proj(time_axis_ds, None)
        x = apply_transformer_blocks(self.blocks, x, key_padding_mask=mask_ds, use_checkpoint=self.cfg.use_grad_checkpoint)
        hidden = self.norm(x)
        tokens = self.tokens_norm(self.hidden_proj(hidden))
        pooled = self.pool(tokens, mask_ds)
        return {"hidden": hidden, "tokens": tokens, "mask": mask_ds, "pooled": pooled, "time_pos": time_axis_ds}


class MetaMLP(nn.Module):
    """Encode only sample duration and slice type for fusion."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, cfg.meta_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.meta_hidden_dim, cfg.embed_dim),
        )

    def forward(self, meta: torch.Tensor) -> torch.Tensor:
        return self.net(meta[:, MODEL_METADATA_SLICE]).unsqueeze(1)
