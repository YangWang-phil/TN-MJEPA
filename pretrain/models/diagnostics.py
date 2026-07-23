"""Label-free diagnostics for retrieval, JEPA prediction, and collapse."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def contrastive_topk_accuracy(
    embeddings: torch.Tensor,
    positive_sets: Sequence[Sequence[int]],
    temperature: float,
    ks: Tuple[int, ...] = (1, 5),
) -> Dict[str, float]:
    """MoCo-style pretext-task Top-K accuracy.

    For every anchor, rank all other in-batch samples by cosine similarity
    (the same similarity matrix used by the InfoNCE loss) and check whether
    at least one true positive lands in the top-K neighbours. Uses only
    unlabeled batch structure (the mileage-overlap / augmented-pair positive
    sets already built for the contrastive loss), so it costs nothing beyond
    a topk() on a matrix we compute anyway.
    """
    emb = F.normalize(embeddings, dim=-1)
    sim = torch.matmul(emb, emb.transpose(0, 1)) / temperature
    n = sim.shape[0]
    sim.fill_diagonal_(float("-inf"))
    max_k = min(max(ks), n - 1) if n > 1 else 0
    hits = {k: 0 for k in ks}
    valid = 0
    if max_k <= 0:
        return {f"top{k}": float("nan") for k in ks}
    for i, positives in enumerate(positive_sets):
        pos = {p for p in positives if p != i}
        if not pos:
            continue
        valid += 1
        topk_idx = torch.topk(sim[i], k=max_k).indices.tolist()
        for k in ks:
            if any(idx in pos for idx in topk_idx[:k]):
                hits[k] += 1
    denom = max(valid, 1)
    return {f"top{k}": hits[k] / denom for k in ks}


@torch.no_grad()
def representation_health(x: torch.Tensor, eps: float = 1e-6) -> Dict[str, float]:
    """Collapse diagnostics for an embedding batch ``[N, D]``.

    - ``std_mean`` / ``std_min``: per-dimension std across the batch. A
      healthy run keeps ``std_min`` well above 0; a value near 0 means at
      least one dimension has collapsed to a constant.
    - ``effective_rank``: ``exp(entropy)`` of the normalized eigenvalue
      spectrum of the centred covariance matrix. Ranges from 1 (all variance
      in a single direction -- full collapse) to ``D`` (isotropic, every
      dimension carries independent information).
    """
    x = x - x.mean(dim=0, keepdim=True)
    # unbiased=False: population std stays well-defined (0, not NaN) even for
    # the occasional batch-size-1 tail batch from bucket batching.
    std = x.std(dim=0, unbiased=False)
    cov = (x.T @ x) / max(x.shape[0] - 1, 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(0)
    p = eigvals / eigvals.sum().clamp_min(eps)
    entropy = -(p * (p + eps).log()).sum()
    return {
        "std_mean": std.mean().item(),
        "std_min": std.min().item(),
        "effective_rank": torch.exp(entropy).item(),
        "dim": float(x.shape[1]),
    }


@torch.no_grad()
def alignment_uniformity(x: torch.Tensor, y: torch.Tensor, t: float = 2.0) -> Dict[str, float]:
    """Alignment & uniformity (Wang & Isola, 2020), both label-free.

    ``x[i]`` and ``y[i]`` must be two augmented views of the *same* sample.
    Lower is better for both: ``alignment`` -> 0 means the two views of a
    sample map to (nearly) the same point; ``uniformity`` -> more negative
    means embeddings spread out more evenly on the hypersphere instead of
    clumping together (another collapse symptom).
    """
    xn = F.normalize(x, dim=-1)
    yn = F.normalize(y, dim=-1)
    align = (xn - yn).pow(2).sum(dim=-1).mean().item()
    z = torch.cat([xn, yn], dim=0)
    sq_dist = torch.cdist(z, z).pow(2)
    n = z.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=z.device)
    uniform = torch.log(torch.exp(-t * sq_dist[mask]).mean().clamp_min(1e-12)).item()
    return {"alignment": align, "uniformity": uniform}


@torch.no_grad()
def jepa_prediction_quality(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean cosine similarity between predicted and true target tokens at
    masked positions -- scale-free complement to the raw Smooth-L1 loss.
    """
    if not mask.any():
        return float("nan")
    return F.cosine_similarity(pred[mask], target[mask], dim=-1).mean().item()
