"""Tube-mask sampling, token alignment, and modality dropout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

from pretrain.configs import ModelConfig


@dataclass
class MaskOutput:
    visible_mask: torch.Tensor
    target_mask: torch.Tensor
    modality_keep: Dict[str, torch.Tensor]


def sample_time_tube_masks(time_mask: torch.Tensor, cfg: ModelConfig) -> torch.Tensor:
    bsz, length = time_mask.shape
    target = torch.zeros_like(time_mask)
    for b in range(bsz):
        valid_idx = torch.where(time_mask[b])[0]
        if valid_idx.numel() <= 1:
            continue
        valid_len = int(valid_idx.numel())
        occupied = torch.zeros(valid_len, dtype=torch.bool, device=time_mask.device)
        for _ in range(cfg.jepa_num_tube_masks):
            ratio = torch.empty(1, device=time_mask.device).uniform_(cfg.jepa_tube_ratio_min, cfg.jepa_tube_ratio_max).item()
            tube_len = max(1, int(round(valid_len * ratio)))
            if tube_len >= valid_len:
                occupied[:] = True
                break
            start = int(torch.randint(0, valid_len - tube_len + 1, (1,), device=time_mask.device).item())
            occupied[start : start + tube_len] = True
        target[b, valid_idx] = occupied
    return target


def modality_dropout(cfg: ModelConfig, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for name in ["tf", "wave", "vel"]:
        keep = torch.rand(batch_size, device=device) > cfg.modality_dropout_prob
        out[name] = keep
    return out


def expand_time_mask_to_tokens(time_target_mask: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    """Realigns a raw-time-frame target mask onto a modality's (possibly downsampled) token grid.

    The TF/Wave/Vel encoders now aggressively downsample the raw time axis (real T reaches
    tens of thousands of frames), so `token_mask` is almost always coarser than
    `time_target_mask`, not finer -- the reverse of what this function originally assumed
    (it used to only handle repeat_interleave upsampling). A token is treated as "targeted"
    if any of the raw frames it covers were targeted (max-pool == "any overlap").
    """
    raw_len = time_target_mask.shape[1]
    tok_len = token_mask.shape[1]
    if tok_len == raw_len:
        return time_target_mask & token_mask
    if tok_len < raw_len:
        pooled = F.adaptive_max_pool1d(time_target_mask.float().unsqueeze(1), tok_len).squeeze(1) > 0.5
        return pooled & token_mask
    scale = max(tok_len // max(raw_len, 1), 1)
    expanded = torch.repeat_interleave(time_target_mask, repeats=scale, dim=1)
    if expanded.shape[1] < tok_len:
        pad = tok_len - expanded.shape[1]
        expanded = F.pad(expanded, (0, pad), value=False)
    return expanded[:, :tok_len] & token_mask


def build_masks(
    time_mask: torch.Tensor,
    token_masks: Dict[str, torch.Tensor],
    cfg: ModelConfig,
) -> MaskOutput:
    time_target_mask = sample_time_tube_masks(time_mask, cfg)
    visible_mask = {}
    target_mask = {}
    for name, token_mask in token_masks.items():
        tgt = expand_time_mask_to_tokens(time_target_mask, token_mask)
        target_mask[name] = tgt
        visible_mask[name] = token_mask & ~tgt
    keep = modality_dropout(cfg, batch_size=time_mask.shape[0], device=time_mask.device)
    for name, keep_mask in keep.items():
        visible_mask[name] = visible_mask[name] & keep_mask[:, None]
    return MaskOutput(visible_mask=visible_mask, target_mask=target_mask, modality_keep=keep)
