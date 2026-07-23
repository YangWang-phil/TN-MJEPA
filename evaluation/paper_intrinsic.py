"""Source-disjoint, sample-level intrinsic evaluation for Section 5.

The evaluator never trains or selects a checkpoint. It reuses the eight frozen
Section 5 checkpoints with one deterministic test protocol, writes per-run
artifacts as soon as they are available, and then aggregates publication source
data. Full-model seeds additionally receive clean-feature and mechanism passes.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from pretrain.checkpoints import load_model_state_compat
from pretrain.configs import ExperimentConfig, get_preset
from pretrain.data.dataset import SampleRecord, TrainNoiseDataset, _build_record, collate_samples
from pretrain.data.normalization import TrainStats
from pretrain.models.tn_mjepa import TNMJEPA
from pretrain.utils import set_seed


RUN_SPECS = (
    ("full_seed42", "Full", "full", 42),
    ("full_seed43", "Full", "full", 43),
    ("full_seed44", "Full", "full", 44),
    ("no_aux_seed42", "w/o auxiliary", "no_aux", 42),
    ("no_sga_seed42", "w/o SGA", "no_sga", 42),
    ("no_position_seed42", "w/o position", "no_position", 42),
    ("jepa_only_seed42", "JEPA-only", "jepa_only", 42),
    ("contrastive_only_seed42", "Contrastive-only", "contrastive_only", 42),
)
FULL_RUNS = {"full_seed42", "full_seed43", "full_seed44"}
MODALITIES = ("tf", "wave", "vel", "fusion")
FUSION_MODALITIES = ("TF", "Wave", "Velocity", "Metadata")
EVAL_SEED = 20260719


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"Cannot infer CSV columns for empty output: {path}")
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def update_dataclass(target: object, values: Mapping[str, object]) -> None:
    for name, value in values.items():
        if hasattr(target, name):
            setattr(target, name, value)


def load_run_config(
    run_dir: Path,
    *,
    dataset_dir: str,
    stats_path: str,
    num_workers: int,
    prefetch_factor: int,
    eval_seed: int,
) -> ExperimentConfig:
    payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    cfg = get_preset("tn-mjepa-base")
    cfg.name = str(payload.get("name", cfg.name))
    for section in ("data", "augmentation", "model", "loss", "train", "eval"):
        update_dataclass(getattr(cfg, section), payload.get(section, {}))
    cfg.data.dataset_dir = dataset_dir
    cfg.data.stats_path = stats_path
    cfg.data.random_seed = eval_seed
    cfg.data.num_workers = num_workers
    cfg.data.prefetch_factor = prefetch_factor
    cfg.model.use_grad_checkpoint = False
    return cfg


def dataset_from_records(
    cfg,
    stats: TrainStats,
    records: Sequence[SampleRecord],
    *,
    split: str,
    aug_cfg,
) -> TrainNoiseDataset:
    dataset = object.__new__(TrainNoiseDataset)
    dataset.cfg = cfg
    dataset.dataset_dir = Path(cfg.dataset_dir)
    dataset.records = list(records)
    dataset.stats = stats
    dataset.aug_cfg = aug_cfg
    dataset.split = split
    suggested = stats.length_distribution.get("suggested_T_max")
    dataset.t_max = cfg.max_time_tokens if cfg.max_time_tokens is not None else (int(suggested) if suggested else None)
    return dataset


def load_frozen_split_records(
    dataset_dir: Path,
    stats_path: Path,
    *,
    split: str,
    num_workers: int,
) -> list[SampleRecord]:
    """Index only the filenames in the frozen source-disjoint split manifest."""
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    filenames = payload.get("split_files", {}).get(split, [])
    if not filenames:
        raise ValueError(f"No filenames found for split={split!r} in {stats_path}")
    paths = [dataset_dir / filename for filename in filenames]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Frozen split references {len(missing)} missing files; first: {missing[0]}")
    if num_workers <= 1:
        records = [_build_record(path) for path in paths]
    else:
        with ThreadPoolExecutor(max_workers=min(num_workers, 32)) as executor:
            records = list(executor.map(_build_record, paths))
    return records


def make_loader(
    dataset: TrainNoiseDataset,
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    pad_to_multiple_of: int,
) -> DataLoader:
    kwargs: dict[str, object] = {}
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=partial(collate_samples, pad_to_multiple_of=pad_to_multiple_of),
        **kwargs,
    )


def shutdown_loader(loader: DataLoader | None) -> None:
    """Release persistent DataLoader workers before entering another eval phase."""
    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        iterator._shutdown_workers()
        loader._iterator = None


def move_batch(batch: Mapping[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def build_cross_view_positive_mask(records: Sequence[SampleRecord]) -> np.ndarray:
    """Return overlap-aware cross-view positives for a fixed record order."""
    size = len(records)
    mask = np.eye(size, dtype=bool)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if record.slice_type == "mileage":
            groups[record.source_file].append(index)
    for indices in groups.values():
        starts = np.asarray([records[index].mileage_start_km for index in indices], dtype=np.float64)
        ends = np.asarray([records[index].mileage_end_km for index in indices], dtype=np.float64)
        valid = np.isfinite(starts) & np.isfinite(ends)
        overlap = np.minimum(ends[:, None], ends[None, :]) - np.maximum(starts[:, None], starts[None, :])
        overlap = (overlap > 0) & valid[:, None] & valid[None, :]
        mask[np.ix_(indices, indices)] |= overlap
    return mask


def global_cross_view_retrieval(
    view1: np.ndarray,
    view2: np.ndarray,
    positive_mask: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Compute overlap-aware cross-view retrieval and per-anchor margins."""
    if view1.shape != view2.shape:
        raise ValueError(f"Expected paired feature matrices, got {view1.shape} and {view2.shape}")
    if positive_mask.shape != (view1.shape[0], view1.shape[0]):
        raise ValueError("positive_mask shape does not match the paired feature count")
    x = F.normalize(torch.from_numpy(np.asarray(view1, dtype=np.float32)), dim=1)
    y = F.normalize(torch.from_numpy(np.asarray(view2, dtype=np.float32)), dim=1)
    similarity = x @ y.T
    positives = torch.from_numpy(positive_mask)
    negatives = ~positives
    best_positive = similarity.masked_fill(~positives, float("-inf")).max(dim=1).values
    best_negative = similarity.masked_fill(~negatives, float("-inf")).max(dim=1).values
    predicted = similarity.argmax(dim=1)
    hit = positives[torch.arange(similarity.shape[0]), predicted]
    return {
        "retrieval_at1": float(hit.float().mean().item()),
        "hit": hit.numpy().astype(np.int8),
        "best_positive": best_positive.numpy(),
        "best_negative": best_negative.numpy(),
        "margin": (best_positive - best_negative).numpy(),
    }


def covariance_spectrum(features: np.ndarray) -> tuple[np.ndarray, float]:
    """Return descending normalized covariance eigenvalues and effective rank."""
    x = np.asarray(features, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    covariance = (x.T @ x) / max(x.shape[0] - 1, 1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)[::-1]
    fractions = eigenvalues / max(float(eigenvalues.sum()), np.finfo(np.float64).eps)
    positive = fractions > 0
    entropy = -float(np.sum(fractions[positive] * np.log(fractions[positive])))
    return fractions, float(np.exp(entropy))


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear centered-kernel alignment computed without materializing Gram matrices."""
    x_centered = np.asarray(x, dtype=np.float64) - np.asarray(x, dtype=np.float64).mean(axis=0, keepdims=True)
    y_centered = np.asarray(y, dtype=np.float64) - np.asarray(y, dtype=np.float64).mean(axis=0, keepdims=True)
    cross = x_centered.T @ y_centered
    xx = x_centered.T @ x_centered
    yy = y_centered.T @ y_centered
    numerator = float(np.sum(cross * cross))
    denominator = math.sqrt(float(np.sum(xx * xx)) * float(np.sum(yy * yy)))
    return numerator / max(denominator, np.finfo(np.float64).eps)


def cka_matrix(features: Mapping[str, np.ndarray], names: Sequence[str] = MODALITIES) -> np.ndarray:
    matrix = np.empty((len(names), len(names)), dtype=np.float64)
    for row, name_x in enumerate(names):
        for column, name_y in enumerate(names):
            matrix[row, column] = linear_cka(features[name_x], features[name_y])
    return matrix


def samplewise_masked_cosine(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    cosine = F.cosine_similarity(pred, target, dim=-1)
    output = torch.full((cosine.shape[0],), float("nan"), device=cosine.device)
    for index in range(cosine.shape[0]):
        selected = mask[index]
        if selected.any():
            output[index] = cosine[index][selected].mean()
    return output.detach().cpu().numpy()


def applied_gate_indices(num_frequency_tokens: int, num_frequency_groups: int) -> np.ndarray:
    return np.linspace(0, num_frequency_groups - 1, num=num_frequency_tokens).astype(np.int64)


def attention_log2_enrichment(observed_mass: float, valid_key_count: int, total_valid_count: int) -> float:
    expected = valid_key_count / max(total_valid_count, 1)
    eps = np.finfo(np.float64).eps
    return float(np.log2((observed_mass + eps) / (expected + eps)))


def _load_model(cfg: ExperimentConfig, checkpoint_path: Path, device: torch.device) -> TNMJEPA:
    model = TNMJEPA(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    load_model_state_compat(model, checkpoint["model"], checkpoint_path=checkpoint_path)
    model.to(device).eval()
    return model


@torch.inference_mode()
def evaluate_augmented_run(
    model: TNMJEPA,
    loader: DataLoader,
    device: torch.device,
    *,
    positive_mask: np.ndarray,
    eval_seed: int,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    set_seed(eval_seed)
    ids: list[str] = []
    pooled_view1: list[np.ndarray] = []
    pooled_view2: list[np.ndarray] = []
    jepa_values: dict[str, list[np.ndarray]] = {name: [] for name in ("tf", "wave", "vel")}

    for batch in tqdm(loader, desc="Augmented intrinsic", leave=False):
        moved = move_batch(batch, device)
        output = model(moved, return_extras=True)
        ids.extend(record.stem for record in batch["records"])
        pooled_view1.append(output["pooled_online"].detach().cpu().numpy())
        pooled_view2.append(output["pooled_online_v2"].detach().cpu().numpy())
        for name in ("tf", "wave", "vel"):
            extra = output["jepa_extras"][name]
            jepa_values[name].append(samplewise_masked_cosine(extra["pred"], extra["target"], extra["mask"]))

    view1 = np.concatenate(pooled_view1, axis=0)
    view2 = np.concatenate(pooled_view2, axis=0)
    per_modality = {name: np.concatenate(values) for name, values in jepa_values.items()}
    retrieval = global_cross_view_retrieval(view1, view2, positive_mask)
    _, effective_rank = covariance_spectrum(view1)
    rows: list[dict[str, object]] = []
    for index, sample_id in enumerate(ids):
        rows.append(
            {
                "sample_index": index,
                "sample_id": sample_id,
                "retrieval_hit": int(retrieval["hit"][index]),
                "retrieval_margin": float(retrieval["margin"][index]),
                "best_positive_cosine": float(retrieval["best_positive"][index]),
                "best_negative_cosine": float(retrieval["best_negative"][index]),
                "jepa_tf": float(per_modality["tf"][index]),
                "jepa_wave": float(per_modality["wave"][index]),
                "jepa_velocity": float(per_modality["vel"][index]),
            }
        )
    summary = {
        "num_samples": int(len(rows)),
        "retrieval_at1": float(retrieval["retrieval_at1"]),
        "retrieval_margin_mean": float(np.mean(retrieval["margin"])),
        "retrieval_margin_median": float(np.median(retrieval["margin"])),
        "jepa_tf": float(np.nanmean(per_modality["tf"])),
        "jepa_wave": float(np.nanmean(per_modality["wave"])),
        "jepa_velocity": float(np.nanmean(per_modality["vel"])),
        "effective_rank": effective_rank,
    }
    return rows, summary


@torch.inference_mode()
def extract_clean_features_and_sga(
    model: TNMJEPA,
    loader: DataLoader,
    device: torch.device,
    stats: TrainStats,
    cfg: ExperimentConfig,
) -> tuple[list[str], np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ids: list[str] = []
    mean_speeds: list[np.ndarray] = []
    features: dict[str, list[np.ndarray]] = {name: [] for name in MODALITIES}
    gate_values: list[np.ndarray] = []
    token_speeds: list[np.ndarray] = []
    gate_indices: np.ndarray | None = None
    band_centers: np.ndarray | None = None

    for batch in tqdm(loader, desc="Clean features and SGA", leave=False):
        moved = move_batch(batch, device)
        vel_out = model.vel_encoder(moved["vel_view1"], moved["time_view1"], moved["time_mask_view1"])
        tf_out = model.tf_encoder(
            moved["tf_log_view1"],
            moved["freq_axis"],
            moved["time_view1"],
            moved["time_mask_view1"],
            speed_hidden=vel_out["hidden"],
            speed_valid_mask=vel_out["mask"],
        )
        wave_out = model.wave_encoder(
            moved["pa_view1"],
            moved["time_view1"],
            moved["time_mask_view1"],
            speed_hidden=vel_out["hidden"],
            speed_valid_mask=vel_out["mask"],
        )
        meta_token = model.meta_mlp(moved["meta"])
        fusion_out = model.fusion(
            tf_out["tokens"],
            tf_out["mask"],
            wave_out["tokens"],
            wave_out["mask"],
            vel_out["tokens"],
            vel_out["mask"],
            meta_token,
        )
        ids.extend(record.stem for record in batch["records"])
        mean_speed = moved["meta"][:, 1].detach().cpu().numpy()
        mean_speed = mean_speed * stats.mean_vel.std.item() + stats.mean_vel.mean.item()
        mean_speeds.append(mean_speed)
        for name, output in (("tf", tf_out), ("wave", wave_out), ("vel", vel_out)):
            features[name].append(output["pooled"].detach().cpu().numpy())
        features["fusion"].append(fusion_out["pooled"].detach().cpu().numpy())

        sga = model.tf_encoder.sga
        if sga is None or not sga.use_freq_gate:
            continue
        num_frequency_tokens = int(moved["freq_axis"].shape[1] // cfg.model.tf_patch_freq)
        time_tokens = int(tf_out["tokens"].shape[1] // num_frequency_tokens)
        speed_hidden = F.interpolate(
            vel_out["hidden"].transpose(1, 2),
            size=time_tokens,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        speed_mask = F.interpolate(
            vel_out["mask"].float().unsqueeze(1),
            size=time_tokens,
            mode="nearest",
        ).squeeze(1) > 0.5
        all_gates = sga.frequency_gate(speed_hidden, sga.num_freq_groups, speed_mask)
        gate_indices = applied_gate_indices(num_frequency_tokens, sga.num_freq_groups)
        selected_gates = all_gates[..., torch.as_tensor(gate_indices, device=device)]
        physical_speed = moved["vel_view1"] * stats.vel_resampled.std.item() + stats.vel_resampled.mean.item()
        physical_speed = F.interpolate(
            physical_speed.unsqueeze(1),
            size=time_tokens,
            mode="linear",
            align_corners=False,
        ).squeeze(1)
        gate_values.append(selected_gates[speed_mask].detach().cpu().numpy())
        token_speeds.append(physical_speed[speed_mask].detach().cpu().numpy())
        if band_centers is None:
            frequency_axis = stats.reference_freq_axis
            usable = num_frequency_tokens * cfg.model.tf_patch_freq
            band_centers = frequency_axis[:usable].reshape(num_frequency_tokens, cfg.model.tf_patch_freq).mean(axis=1)

    if gate_indices is None or band_centers is None:
        raise RuntimeError("Full checkpoint did not expose an active SGA frequency gate")
    return (
        ids,
        np.concatenate(mean_speeds),
        {name: np.concatenate(values, axis=0) for name, values in features.items()},
        np.concatenate(token_speeds),
        np.concatenate(gate_values, axis=0),
        gate_indices,
        band_centers,
    )


def summarize_sga(
    *,
    seed: int,
    speeds: np.ndarray,
    gates: np.ndarray,
    gate_indices: np.ndarray,
    band_centers: np.ndarray,
    num_bins: int = 12,
) -> list[dict[str, object]]:
    order = np.argsort(speeds, kind="stable")
    chunks = np.array_split(order, num_bins)
    baseline = gates.mean(axis=0)
    correlations = [
        float(spearmanr(speeds, gates[:, band]).statistic)
        for band in range(gates.shape[1])
    ]
    rows: list[dict[str, object]] = []
    for bin_index, indices in enumerate(chunks):
        mean_gate = gates[indices].mean(axis=0)
        speed_center = float(np.median(speeds[indices]))
        for band in range(gates.shape[1]):
            rows.append(
                {
                    "seed": seed,
                    "speed_bin": bin_index + 1,
                    "speed_center": speed_center,
                    "band_index": band + 1,
                    "gate_index": int(gate_indices[band]),
                    "frequency_center_hz": float(band_centers[band]),
                    "mean_gate": float(mean_gate[band]),
                    "log2_enrichment": float(np.log2(mean_gate[band] / max(baseline[band], 1e-12))),
                    "spearman_rho": correlations[band],
                }
            )
    return rows


def select_mechanism_records(
    ids: Sequence[str],
    mean_speeds: np.ndarray,
    records: Sequence[SampleRecord],
    *,
    seed: int,
    num_bins: int = 12,
    per_bin: int = 32,
) -> tuple[list[SampleRecord], list[dict[str, object]]]:
    if [record.stem for record in records] != list(ids):
        raise ValueError("Clean-feature sample order does not match the source-disjoint test records")
    rng = np.random.default_rng(seed)
    ordered = np.argsort(mean_speeds, kind="stable")
    bins = np.array_split(ordered, num_bins)
    selected: list[tuple[int, int]] = []
    for bin_index, candidates in enumerate(bins):
        chosen = rng.choice(candidates, size=min(per_bin, candidates.size), replace=False)
        selected.extend((bin_index + 1, int(index)) for index in np.sort(chosen))
    rows = [
        {
            "speed_bin": bin_index,
            "sample_index": index,
            "sample_id": records[index].stem,
            "mean_speed": float(mean_speeds[index]),
        }
        for bin_index, index in selected
    ]
    return [records[index] for _, index in selected], rows


@torch.inference_mode()
def extract_fusion_mechanism(
    model: TNMJEPA,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    attention_rows: list[np.ndarray] = []
    pool_rows: list[np.ndarray] = []
    for batch in tqdm(loader, desc="Fusion mechanism", leave=False):
        moved = move_batch(batch, device)
        vel_out = model.vel_encoder(moved["vel_view1"], moved["time_view1"], moved["time_mask_view1"])
        tf_out = model.tf_encoder(
            moved["tf_log_view1"],
            moved["freq_axis"],
            moved["time_view1"],
            moved["time_mask_view1"],
            speed_hidden=vel_out["hidden"],
            speed_valid_mask=vel_out["mask"],
        )
        wave_out = model.wave_encoder(
            moved["pa_view1"],
            moved["time_view1"],
            moved["time_mask_view1"],
            speed_hidden=vel_out["hidden"],
            speed_valid_mask=vel_out["mask"],
        )
        fusion_out = model.fusion(
            tf_out["tokens"],
            tf_out["mask"],
            wave_out["tokens"],
            wave_out["mask"],
            vel_out["tokens"],
            vel_out["mask"],
            model.meta_mlp(moved["meta"]),
            return_attention=True,
        )
        mask = fusion_out["mask"]
        lengths = fusion_out["lengths"]
        boundaries = np.cumsum([0, lengths["tf"], lengths["wave"], lengths["vel"], lengths["meta"]])
        for sample in range(mask.shape[0]):
            valid_total = int(mask[sample].sum().item())
            layer_values = np.empty((len(fusion_out["attn_maps"]), 4, 4), dtype=np.float64)
            for layer_index, layer_attention in enumerate(fusion_out["attn_maps"]):
                attention = layer_attention[sample].mean(dim=0)
                for query in range(4):
                    query_slice = slice(boundaries[query], boundaries[query + 1])
                    query_valid = mask[sample, query_slice]
                    rows = attention[query_slice][query_valid]
                    if rows.shape[0] == 0:
                        raise ValueError(f"No valid query tokens for modality index {query}")
                    observed_masses: list[float] = []
                    valid_key_counts: list[int] = []
                    for key in range(4):
                        key_slice = slice(boundaries[key], boundaries[key + 1])
                        key_valid = mask[sample, key_slice]
                        observed = float(rows[:, key_slice][:, key_valid].sum(dim=1).mean().item())
                        observed_masses.append(observed)
                        valid_key_counts.append(int(key_valid.sum().item()))
                    if not np.isclose(sum(observed_masses), 1.0, rtol=1e-4, atol=1e-5):
                        raise ValueError("Cross-modal attention mass does not sum to one over valid keys")
                    for key, (observed, valid_key_count) in enumerate(zip(observed_masses, valid_key_counts)):
                        layer_values[layer_index, query, key] = attention_log2_enrichment(
                            observed,
                            valid_key_count,
                            valid_total,
                        )
            attention_rows.append(layer_values)
            pool_values = np.empty(4, dtype=np.float64)
            weights = fusion_out["pool_weights"][sample]
            observed_pool_masses: list[float] = []
            valid_pool_counts: list[int] = []
            for key in range(4):
                key_slice = slice(boundaries[key], boundaries[key + 1])
                key_valid = mask[sample, key_slice]
                observed = float(weights[key_slice][key_valid].sum().item())
                observed_pool_masses.append(observed)
                valid_pool_counts.append(int(key_valid.sum().item()))
            if not np.isclose(sum(observed_pool_masses), 1.0, rtol=1e-4, atol=1e-5):
                raise ValueError("Pooling mass does not sum to one over valid tokens")
            for key, (observed, valid_key_count) in enumerate(zip(observed_pool_masses, valid_pool_counts)):
                pool_values[key] = attention_log2_enrichment(observed, valid_key_count, valid_total)
            pool_rows.append(pool_values)
    return np.stack(attention_rows), np.stack(pool_rows)


def write_run_outputs(
    run_output: Path,
    *,
    run_name: str,
    label: str,
    variant: str,
    seed: int,
    sample_rows: Sequence[Mapping[str, object]],
    summary: Mapping[str, float],
) -> None:
    enriched_rows = [
        {"run": run_name, "label": label, "variant": variant, "seed": seed, **row}
        for row in sample_rows
    ]
    write_csv(run_output / "sample_metrics.csv", enriched_rows)
    payload = {"run": run_name, "label": label, "variant": variant, "seed": seed, **summary}
    (run_output / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_clean_outputs(
    run_output: Path,
    *,
    seed: int,
    sample_ids: Sequence[str],
    mean_speeds: np.ndarray,
    features: Mapping[str, np.ndarray],
    sga_rows: Sequence[Mapping[str, object]],
) -> None:
    spectrum, effective_rank = covariance_spectrum(features["fusion"])
    spectrum_rows = [
        {
            "seed": seed,
            "component": index + 1,
            "eigenvalue_fraction": float(value),
            "cumulative_fraction": float(np.sum(spectrum[: index + 1])),
            "effective_rank": effective_rank,
        }
        for index, value in enumerate(spectrum)
    ]
    matrix = cka_matrix(features)
    cka_rows = [
        {
            "seed": seed,
            "representation_x": MODALITIES[row],
            "representation_y": MODALITIES[column],
            "cka": float(matrix[row, column]),
        }
        for row in range(len(MODALITIES))
        for column in range(len(MODALITIES))
    ]
    index_rows = [
        {"sample_index": index, "sample_id": sample_id, "mean_speed": float(mean_speeds[index])}
        for index, sample_id in enumerate(sample_ids)
    ]
    write_csv(run_output / "spectrum.csv", spectrum_rows)
    write_csv(run_output / "cka.csv", cka_rows)
    write_csv(run_output / "sga_gate.csv", list(sga_rows))
    write_csv(run_output / "clean_sample_index.csv", index_rows)


def write_mechanism_outputs(
    run_output: Path,
    *,
    seed: int,
    attention: np.ndarray,
    pooling: np.ndarray,
) -> None:
    attention_rows = []
    for layer in range(attention.shape[1]):
        for query in range(4):
            for key in range(4):
                values = attention[:, layer, query, key]
                attention_rows.append(
                    {
                        "seed": seed,
                        "layer": layer + 1,
                        "query_modality": FUSION_MODALITIES[query],
                        "key_modality": FUSION_MODALITIES[key],
                        "mean_log2_enrichment": float(values.mean()),
                        "std_log2_enrichment": float(values.std(ddof=1)),
                        "num_samples": int(values.size),
                    }
                )
    pooling_rows = []
    for modality in range(4):
        values = pooling[:, modality]
        pooling_rows.append(
            {
                "seed": seed,
                "modality": FUSION_MODALITIES[modality],
                "mean_log2_enrichment": float(values.mean()),
                "std_log2_enrichment": float(values.std(ddof=1)),
                "num_samples": int(values.size),
            }
        )
    write_csv(run_output / "fusion_attention.csv", attention_rows)
    write_csv(run_output / "pool_enrichment.csv", pooling_rows)


def aggregate_outputs(run_root: Path, experiment_root: Path, output_root: Path) -> None:
    summary_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    spectrum_rows: list[dict[str, object]] = []
    cka_rows: list[dict[str, object]] = []
    sga_rows: list[dict[str, object]] = []
    attention_rows: list[dict[str, object]] = []
    pooling_rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []

    for run_name, label, variant, seed in RUN_SPECS:
        run_output = run_root / run_name
        summary_rows.append(json.loads((run_output / "summary.json").read_text(encoding="utf-8")))
        sample_rows.extend(read_csv(run_output / "sample_metrics.csv"))
        if run_name in FULL_RUNS:
            spectrum_rows.extend(read_csv(run_output / "spectrum.csv"))
            cka_rows.extend(read_csv(run_output / "cka.csv"))
            sga_rows.extend(read_csv(run_output / "sga_gate.csv"))
            attention_rows.extend(read_csv(run_output / "fusion_attention.csv"))
            pooling_rows.extend(read_csv(run_output / "pool_enrichment.csv"))
        with (experiment_root / run_name / "metrics.csv").open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                trajectory_rows.append(
                    {
                        "run": run_name,
                        "label": label,
                        "variant": variant,
                        "seed": seed,
                        "epoch": int(row["epoch"]),
                        "val_top1": float(row["val_top1"]),
                        "val_jepa_tf": float(row["val_jepa_cos_tf"]),
                        "val_jepa_wave": float(row["val_jepa_cos_wave"]),
                        "val_jepa_velocity": float(row["val_jepa_cos_vel"]),
                        "val_jepa_mean": float(
                            np.mean(
                                [
                                    float(row["val_jepa_cos_tf"]),
                                    float(row["val_jepa_cos_wave"]),
                                    float(row["val_jepa_cos_vel"]),
                                ]
                            )
                        ),
                        "val_effective_rank": float(row["val_repr_effective_rank"]),
                    }
                )

    write_csv(output_root / "summary.csv", summary_rows)
    write_csv(output_root / "sample_metrics.csv", sample_rows)
    write_csv(output_root / "spectrum.csv", spectrum_rows)
    write_csv(output_root / "cka.csv", cka_rows)
    write_csv(output_root / "trajectory.csv", trajectory_rows)
    write_csv(output_root / "sga_gate.csv", sga_rows)
    write_csv(output_root / "fusion_attention.csv", attention_rows)
    write_csv(output_root / "pool_enrichment.csv", pooling_rows)
    mechanism_samples = read_csv(run_root / "full_seed42" / "mechanism_samples.csv")
    write_csv(output_root / "mechanism_samples.csv", mechanism_samples)
    manifest = {
        "evaluation_seed": EVAL_SEED,
        "test_samples_per_run": 2514,
        "runs": [spec[0] for spec in RUN_SPECS],
        "retrieval": "global cross-view retrieval on fusion embeddings; positives include paired views and overlapping mileage slices",
        "jepa": "per-sample mean cosine over masked target tokens",
        "spectrum": "clean fusion embeddings; normalized covariance eigenvalues",
        "cka": "linear centered-kernel alignment on clean pooled representations",
        "mechanism_samples": "12 mean-speed strata x 32 held-out samples; identical across Full seeds",
        "inference_boundary": "sample distributions are descriptive; model-selection and hyperparameters were not changed",
    }
    (output_root / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def all_outputs_complete(run_output: Path, full: bool) -> bool:
    required = {"summary.json", "sample_metrics.csv"}
    if full:
        required |= {
            "spectrum.csv",
            "cka.csv",
            "sga_gate.csv",
            "clean_sample_index.csv",
            "fusion_attention.csv",
            "pool_enrichment.csv",
            "mechanism_samples.csv",
        }
    return all((run_output / name).exists() for name in required)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Section 5 v3 intrinsic and mechanism source data.")
    parser.add_argument("--experiment-root", default="runs/paper")
    parser.add_argument("--output-root", default="results/paper_intrinsic")
    parser.add_argument("--stats-path", required=True)
    parser.add_argument("--dataset-dir", default="sswpt_dataset")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--attention-batch-size", type=int, default=2)
    parser.add_argument("--eval-seed", type=int, default=EVAL_SEED)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    experiment_root = Path(args.experiment_root)
    output_root = Path(args.output_root)
    run_root = output_root / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    stats_path = Path(args.stats_path)
    stats = TrainStats.from_json(stats_path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    configs = {
        run_name: load_run_config(
            experiment_root / run_name,
            dataset_dir=args.dataset_dir,
            stats_path=str(stats_path.resolve()),
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            eval_seed=args.eval_seed,
        )
        for run_name, *_ in RUN_SPECS
    }
    first_cfg = configs[RUN_SPECS[0][0]]
    records = load_frozen_split_records(
        Path(args.dataset_dir),
        stats_path,
        split="test",
        num_workers=args.num_workers,
    )
    if len(records) != 2514:
        raise ValueError(f"Expected 2,514 source-disjoint test samples, got {len(records)}")
    augmented_dataset = dataset_from_records(
        first_cfg.data,
        stats,
        records,
        split="test",
        aug_cfg=first_cfg.augmentation,
    )
    augmented_loader = make_loader(
        augmented_dataset,
        batch_size=first_cfg.eval.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pad_to_multiple_of=first_cfg.data.pad_to_multiple_of,
    )
    positive_mask = build_cross_view_positive_mask(records)

    # Phase 1: keep one augmented worker pool alive while evaluating all eight
    # checkpoints, then release it before the clean/mechanism passes.
    for run_name, label, variant, seed in RUN_SPECS:
        run_output = run_root / run_name
        run_output.mkdir(parents=True, exist_ok=True)
        if (
            not args.force
            and (run_output / "sample_metrics.csv").exists()
            and (run_output / "summary.json").exists()
        ):
            print(f"[skip augmented] {run_name}")
            continue

        print(f"[load augmented] {run_name}")
        cfg = configs[run_name]
        model = _load_model(cfg, experiment_root / run_name / "best.pt", device)
        sample_rows, summary = evaluate_augmented_run(
            model,
            augmented_loader,
            device,
            positive_mask=positive_mask,
            eval_seed=args.eval_seed,
        )
        expected_ids = [record.stem for record in records]
        if [str(row["sample_id"]) for row in sample_rows] != expected_ids:
            raise ValueError(f"Sample order mismatch in {run_name}")
        write_run_outputs(
            run_output,
            run_name=run_name,
            label=label,
            variant=variant,
            seed=seed,
            sample_rows=sample_rows,
            summary=summary,
        )
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[complete augmented] {run_name}")

    shutdown_loader(augmented_loader)
    del augmented_loader, augmented_dataset
    gc.collect()

    # Phase 2: reuse one clean worker pool for the three Full checkpoints.
    clean_dataset = dataset_from_records(first_cfg.data, stats, records, split="test", aug_cfg=None)
    clean_loader = make_loader(
        clean_dataset,
        batch_size=first_cfg.eval.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pad_to_multiple_of=first_cfg.data.pad_to_multiple_of,
    )
    mechanism_records: list[SampleRecord] | None = None
    mechanism_rows: list[dict[str, object]] | None = None
    mechanism_loader: DataLoader | None = None

    for run_name, label, variant, seed in (spec for spec in RUN_SPECS if spec[0] in FULL_RUNS):
        run_output = run_root / run_name
        if not args.force and all_outputs_complete(run_output, full=True):
            print(f"[skip clean/mechanism] {run_name}")
            continue
        print(f"[load clean/mechanism] {run_name}")
        cfg = configs[run_name]
        model = _load_model(cfg, experiment_root / run_name / "best.pt", device)

        clean_required = ("spectrum.csv", "cka.csv", "sga_gate.csv", "clean_sample_index.csv")
        if args.force or any(not (run_output / name).exists() for name in clean_required):
            set_seed(args.eval_seed)
            clean = extract_clean_features_and_sga(model, clean_loader, device, stats, cfg)
            sample_ids, mean_speeds, features, token_speeds, gates, gate_indices, band_centers = clean
            sga_rows = summarize_sga(
                seed=seed,
                speeds=token_speeds,
                gates=gates,
                gate_indices=gate_indices,
                band_centers=band_centers,
            )
            write_clean_outputs(
                run_output,
                seed=seed,
                sample_ids=sample_ids,
                mean_speeds=mean_speeds,
                features=features,
                sga_rows=sga_rows,
            )
            if run_name == "full_seed42":
                mechanism_records, mechanism_rows = select_mechanism_records(
                    sample_ids,
                    mean_speeds,
                    records,
                    seed=args.eval_seed,
                )
                write_csv(run_output / "mechanism_samples.csv", mechanism_rows)
            del features, gates, token_speeds

        if mechanism_records is None:
            reference_path = run_root / "full_seed42" / "mechanism_samples.csv"
            if not reference_path.exists():
                index_rows = read_csv(run_root / "full_seed42" / "clean_sample_index.csv")
                sample_ids = [row["sample_id"] for row in index_rows]
                mean_speeds = np.asarray([float(row["mean_speed"]) for row in index_rows])
                mechanism_records, mechanism_rows = select_mechanism_records(
                    sample_ids,
                    mean_speeds,
                    records,
                    seed=args.eval_seed,
                )
                write_csv(reference_path, mechanism_rows)
            else:
                reference_rows = read_csv(reference_path)
                index_by_id = {record.stem: record for record in records}
                mechanism_records = [index_by_id[row["sample_id"]] for row in reference_rows]
                mechanism_rows = [dict(row) for row in reference_rows]
        if not (run_output / "mechanism_samples.csv").exists():
            write_csv(run_output / "mechanism_samples.csv", mechanism_rows or [])
        if mechanism_loader is None:
            mechanism_dataset = dataset_from_records(
                first_cfg.data,
                stats,
                mechanism_records,
                split="test",
                aug_cfg=None,
            )
            mechanism_loader = make_loader(
                mechanism_dataset,
                batch_size=args.attention_batch_size,
                num_workers=0,
                prefetch_factor=args.prefetch_factor,
                pad_to_multiple_of=first_cfg.data.pad_to_multiple_of,
            )
        mechanism_required = ("fusion_attention.csv", "pool_enrichment.csv")
        if args.force or any(not (run_output / name).exists() for name in mechanism_required):
            attention, pooling = extract_fusion_mechanism(model, mechanism_loader, device)
            write_mechanism_outputs(run_output, seed=seed, attention=attention, pooling=pooling)
            del attention, pooling

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[complete clean/mechanism] {run_name}")

    shutdown_loader(clean_loader)
    aggregate_outputs(run_root, experiment_root, output_root)
    print(f"[complete] source data: {output_root.resolve()}")


if __name__ == "__main__":
    main()
