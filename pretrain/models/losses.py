"""JEPA, multi-positive contrastive, auxiliary, and VICReg losses."""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn.functional as F

from pretrain.configs import LossConfig


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return pred.new_tensor(0.0)
    return F.smooth_l1_loss(pred[mask], target[mask])


def multi_positive_info_nce(
    embeddings: torch.Tensor,
    positive_sets: Sequence[Sequence[int]],
    temperature: float,
) -> torch.Tensor:
    emb = F.normalize(embeddings, dim=-1)
    sim = torch.matmul(emb, emb.transpose(0, 1)) / temperature
    total = emb.new_tensor(0.0)
    count = 0
    for i, positives in enumerate(positive_sets):
        positives = [p for p in positives if p != i]
        if not positives:
            continue
        logits = sim[i]
        denom = torch.logsumexp(logits[torch.arange(logits.shape[0], device=logits.device) != i], dim=0)
        pos_logits = torch.stack([logits[p] for p in positives], dim=0)
        total = total + (denom - torch.logsumexp(pos_logits, dim=0))
        count += 1
    return total / max(count, 1)


def build_positive_sets(records: Sequence[object]) -> List[List[int]]:
    n = len(records)
    positives: List[List[int]] = [[] for _ in range(n * 2)]
    for i in range(n):
        positives[i].append(i + n)
        positives[i + n].append(i)
    for i, record_i in enumerate(records):
        source_i = getattr(record_i, "source_file", None)
        start_i = getattr(record_i, "mileage_start_km", float("nan"))
        end_i = getattr(record_i, "mileage_end_km", float("nan"))
        slice_type_i = getattr(record_i, "slice_type", "unknown")
        if slice_type_i != "mileage":
            continue
        for j, record_j in enumerate(records):
            if i == j:
                continue
            if getattr(record_j, "slice_type", "unknown") != "mileage":
                continue
            if getattr(record_j, "source_file", None) != source_i:
                continue
            start_j = getattr(record_j, "mileage_start_km", float("nan"))
            end_j = getattr(record_j, "mileage_end_km", float("nan"))
            overlap = max(0.0, min(end_i, end_j) - max(start_i, start_j))
            if overlap > 0:
                positives[i].append(j)
                positives[i + n].append(j + n)
    return positives


def aux_regression_loss(pred_meta: torch.Tensor, target_meta: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_meta, target_meta)


def vicreg_loss(x: torch.Tensor, y: torch.Tensor, cfg: LossConfig) -> torch.Tensor:
    sim_loss = F.mse_loss(x, y)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    # unbiased=False: default unbiased variance divides by (N-1), which is 0/0=NaN
    # whenever a tail batch (bucket batching / non-drop-last val loader) has exactly
    # 1 sample. Population variance keeps this well-defined (0) instead.
    std_x = torch.sqrt(x.var(dim=0, unbiased=False) + 1e-4)
    std_y = torch.sqrt(y.var(dim=0, unbiased=False) + 1e-4)
    var_loss = torch.mean(F.relu(1.0 - std_x)) + torch.mean(F.relu(1.0 - std_y))
    cov_x = (x.T @ x) / max(x.shape[0] - 1, 1)
    cov_y = (y.T @ y) / max(y.shape[0] - 1, 1)
    cov_loss = off_diagonal(cov_x).pow(2).mean() + off_diagonal(cov_y).pow(2).mean()
    return cfg.vicreg_sim_weight * sim_loss + cfg.vicreg_var_weight * var_loss + cfg.vicreg_cov_weight * cov_loss


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n != m:
        raise ValueError("off_diagonal expects square matrix")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
