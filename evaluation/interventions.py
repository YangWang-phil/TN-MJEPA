"""Quantify frozen-representation dependence under controlled interventions.

Section 5.5 uses the same source-disjoint test records and Full checkpoints as
the Section 5 intrinsic evaluation.  It keeps the encoded sensory tokens fixed
and applies post-hoc fusion-token masks for modality removal.  Metadata is
either replaced by train-partition reference values or permuted across test
samples.  This is an inference-time stress test, not a retrained ablation.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from evaluation.paper_intrinsic import (
    _load_model,
    build_cross_view_positive_mask,
    covariance_spectrum,
    dataset_from_records,
    global_cross_view_retrieval,
    load_frozen_split_records,
    load_run_config,
    make_loader,
    move_batch,
    shutdown_loader,
)
from pretrain.data.normalization import TrainStats


CONDITIONS = ("clean", "drop_tf", "drop_wave", "drop_vel", "metadata_reference", "metadata_shuffle")
SEEDS = (42, 43, 44)
DEFAULT_EVAL_SEED = 20260723


def load_train_metadata_reference(metadata_cache: Path, stats_path: Path) -> Dict[str, float]:
    payload = json.loads(metadata_cache.read_text(encoding="utf-8"))
    train_names = set(
        json.loads(stats_path.read_text(encoding="utf-8"))["split_files"]["train"]
    )
    records = [record for record in payload["records"] if record["name"] in train_names]
    if not records:
        raise ValueError("Metadata cache has no train records")
    durations = np.asarray([float(record["duration_s"]) for record in records], dtype=np.float64)
    mileage_fraction = np.mean([record["slice_type"] == "mileage" for record in records])
    return {
        "duration_median": float(np.median(durations)),
        "mileage_fraction": float(mileage_fraction),
    }


def build_metadata_table(records: Sequence[object]) -> torch.Tensor:
    return torch.tensor(
        [[float(record.duration_s), 1.0 if record.slice_type == "mileage" else 0.0] for record in records],
        dtype=torch.float32,
    )


def condition_meta(
    meta: torch.Tensor,
    condition: str,
    *,
    metadata_table: torch.Tensor,
    permutation: np.ndarray,
    offset: int,
    reference: Mapping[str, float],
) -> torch.Tensor:
    if condition not in {"metadata_reference", "metadata_shuffle"}:
        return meta
    out = meta.clone()
    if condition == "metadata_reference":
        out[:, 2] = float(reference["duration_median"])
        out[:, 3] = float(reference["mileage_fraction"])
    else:
        indices = torch.from_numpy(permutation[offset : offset + meta.shape[0]])
        replacement = metadata_table[indices].to(device=meta.device)
        out[:, 2:] = replacement
    return out


def condition_pooled(
    model: torch.nn.Module,
    encoded: Mapping[str, Mapping[str, torch.Tensor]],
    meta: torch.Tensor,
    condition: str,
    *,
    metadata_table: torch.Tensor,
    permutation: np.ndarray,
    offset: int,
    reference: Mapping[str, float],
) -> torch.Tensor:
    if condition == "clean":
        return encoded["fusion"]["pooled"]

    masks = {name: encoded[name]["mask"] for name in ("tf", "wave", "vel")}
    if condition in {"drop_tf", "drop_wave", "drop_vel"}:
        masks[condition.removeprefix("drop_")] = torch.zeros_like(masks[condition.removeprefix("drop_")])
    intervention_meta = condition_meta(
        meta,
        condition,
        metadata_table=metadata_table,
        permutation=permutation,
        offset=offset,
        reference=reference,
    )
    meta_token = model.meta_mlp(intervention_meta)
    return model.fusion(
        encoded["tf"]["tokens"], masks["tf"],
        encoded["wave"]["tokens"], masks["wave"],
        encoded["vel"]["tokens"], masks["vel"],
        meta_token,
    )["pooled"]


def intervention_metrics(
    clean: np.ndarray,
    intervention: np.ndarray,
    positive_mask: np.ndarray,
    clean_effective_rank: float,
) -> tuple[Dict[str, float], Dict[str, np.ndarray]]:
    if clean.shape != intervention.shape:
        raise ValueError(f"Expected matching clean/intervention features, got {clean.shape} and {intervention.shape}")
    clean_tensor = torch.from_numpy(clean.astype(np.float32, copy=False))
    intervention_tensor = torch.from_numpy(intervention.astype(np.float32, copy=False))
    cosine = F.cosine_similarity(clean_tensor, intervention_tensor, dim=1).numpy()
    retrieval = global_cross_view_retrieval(intervention, clean, positive_mask)
    _, effective_rank = covariance_spectrum(intervention)
    summary = {
        "num_samples": int(intervention.shape[0]),
        "embedding_cosine_mean": float(np.mean(cosine)),
        "embedding_cosine_median": float(np.median(cosine)),
        "embedding_cosine_p10": float(np.percentile(cosine, 10)),
        "retrieval_at1": float(retrieval["retrieval_at1"]),
        "retrieval_margin_mean": float(np.mean(retrieval["margin"])),
        "retrieval_margin_median": float(np.median(retrieval["margin"])),
        "effective_rank": float(effective_rank),
        "effective_rank_ratio": float(effective_rank / max(clean_effective_rank, np.finfo(np.float64).eps)),
    }
    rows = {
        "embedding_cosine": cosine,
        "retrieval_hit": retrieval["hit"],
        "retrieval_margin": retrieval["margin"],
    }
    return summary, rows


@torch.inference_mode()
def evaluate_seed(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    records: Sequence[object],
    positive_mask: np.ndarray,
    metadata_table: torch.Tensor,
    metadata_permutation: np.ndarray,
    metadata_reference: Mapping[str, float],
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    clean_features: List[np.ndarray] = []
    condition_features: Dict[str, List[np.ndarray]] = {condition: [] for condition in CONDITIONS if condition != "clean"}
    offset = 0
    for batch in tqdm(loader, desc="Section 5.5 interventions", leave=False):
        moved = move_batch(batch, device)
        encoded = model._encode_online(
            tf_log=moved["tf_log_view1"],
            pa=moved["pa_view1"],
            vel=moved["vel_view1"],
            time_axis=moved["time_view1"],
            time_mask=moved["time_mask_view1"],
            freq_axis=moved["freq_axis"],
            meta=moved["meta"],
        )
        clean_features.append(encoded["fusion"]["pooled"].cpu().numpy())
        for condition in condition_features:
            pooled = condition_pooled(
                model,
                encoded,
                moved["meta"],
                condition,
                metadata_table=metadata_table,
                permutation=metadata_permutation,
                offset=offset,
                reference=metadata_reference,
            )
            condition_features[condition].append(pooled.cpu().numpy())
        offset += len(batch["records"])

    clean = np.concatenate(clean_features, axis=0)
    _, clean_effective_rank = covariance_spectrum(clean)
    all_features = {"clean": clean}
    all_features.update({condition: np.concatenate(values, axis=0) for condition, values in condition_features.items()})
    summaries: List[Dict[str, object]] = []
    sample_rows: List[Dict[str, object]] = []
    for condition in CONDITIONS:
        summary, rows = intervention_metrics(
            clean,
            all_features[condition],
            positive_mask,
            clean_effective_rank,
        )
        summary.update({"condition": condition, "clean_effective_rank": float(clean_effective_rank)})
        summaries.append(summary)
        for index, record in enumerate(records):
            sample_rows.append(
                {
                    "condition": condition,
                    "sample_index": index,
                    "sample_id": record.stem,
                    "embedding_cosine": float(rows["embedding_cosine"][index]),
                    "retrieval_hit": int(rows["retrieval_hit"][index]),
                    "retrieval_margin": float(rows["retrieval_margin"][index]),
                }
            )
    return summaries, sample_rows


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_summaries(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    metric_names = (
        "embedding_cosine_mean",
        "embedding_cosine_median",
        "embedding_cosine_p10",
        "retrieval_at1",
        "retrieval_margin_mean",
        "retrieval_margin_median",
        "effective_rank",
        "effective_rank_ratio",
    )
    aggregate: List[Dict[str, object]] = []
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        if not selected:
            continue
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            aggregate.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "n_runs": int(values.size),
                    "mean": float(values.mean()),
                    "sample_std": float(values.std(ddof=1)) if values.size > 1 else None,
                }
            )
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Section 5.5 frozen representation intervention evaluation.")
    parser.add_argument("--experiment-root", default="runs/paper")
    parser.add_argument("--stats-path", required=True)
    parser.add_argument("--dataset-dir", default="sswpt_dataset")
    parser.add_argument("--metadata-cache", required=True)
    parser.add_argument("--output-dir", default="results/interventions")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-seed", type=int, default=DEFAULT_EVAL_SEED)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--max-samples", type=int, default=None, help="Smoke-test limit; omit for the formal 2,514-sample protocol.")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    experiment_root = Path(args.experiment_root)
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = Path(args.stats_path)
    stats = TrainStats.from_json(stats_path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    reference = load_train_metadata_reference(Path(args.metadata_cache), stats_path)
    records = load_frozen_split_records(dataset_dir, stats_path, split="test", num_workers=args.num_workers)
    if args.max_samples is not None:
        records = records[: args.max_samples]
    if args.max_samples is None and len(records) != 2514:
        raise ValueError(f"Expected 2,514 source-disjoint test samples, got {len(records)}")
    positive_mask = build_cross_view_positive_mask(records)
    metadata_table = build_metadata_table(records)
    permutation = np.random.default_rng(args.eval_seed).permutation(len(records))

    base_cfg = load_run_config(
        experiment_root / f"full_seed{args.seeds[0]}",
        dataset_dir=str(dataset_dir.resolve()),
        stats_path=str(stats_path.resolve()),
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        eval_seed=args.eval_seed,
    )
    dataset = dataset_from_records(base_cfg.data, stats, records, split="test", aug_cfg=None)
    loader = make_loader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pad_to_multiple_of=base_cfg.data.pad_to_multiple_of,
    )

    manifest = {
        "protocol": "post-hoc fusion-token intervention on frozen Full encoders",
        "experiment_root": str(experiment_root.resolve()),
        "dataset_dir": str(dataset_dir.resolve()),
        "stats_path": str(stats_path.resolve()),
        "num_test_samples": len(records),
        "max_samples": args.max_samples,
        "seeds": list(args.seeds),
        "conditions": list(CONDITIONS),
        "eval_seed": args.eval_seed,
        "metadata_reference": reference,
        "positive_definition": "same sample or overlapping mileage slices, matching Section 5 intrinsic evaluation",
        "causal_scope": "inference-time intervention; no retraining and no claim of causal training ablation",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    summaries: List[Dict[str, object]] = []
    sample_rows: List[Dict[str, object]] = []
    for seed in args.seeds:
        seed_summary_path = output_dir / f"full_seed{seed}.json"
        seed_sample_path = output_dir / f"full_seed{seed}_sample_metrics.csv"
        if not args.force and seed_summary_path.exists() and seed_sample_path.exists():
            seed_payload = json.loads(seed_summary_path.read_text(encoding="utf-8"))
            summaries.extend(seed_payload["summaries"])
            sample_rows.extend(csv.DictReader(seed_sample_path.open("r", encoding="utf-8-sig", newline="")))
            continue
        cfg = load_run_config(
            experiment_root / f"full_seed{seed}",
            dataset_dir=str(dataset_dir.resolve()),
            stats_path=str(stats_path.resolve()),
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            eval_seed=args.eval_seed,
        )
        model = _load_model(cfg, experiment_root / f"full_seed{seed}" / "best.pt", device)
        seed_summaries, seed_sample_rows = evaluate_seed(
            model,
            loader,
            device,
            records,
            positive_mask,
            metadata_table,
            permutation,
            reference,
        )
        for row in seed_summaries:
            row["encoder_seed"] = seed
        for row in seed_sample_rows:
            row["encoder_seed"] = seed
        summaries.extend(seed_summaries)
        sample_rows.extend(seed_sample_rows)
        seed_summary_path.write_text(json.dumps({"encoder_seed": seed, "summaries": seed_summaries}, ensure_ascii=False, indent=2), encoding="utf-8")
        write_csv(seed_sample_path, seed_sample_rows)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    shutdown_loader(loader)
    aggregate = aggregate_summaries(summaries)
    write_csv(output_dir / "summary_long.csv", summaries)
    write_csv(output_dir / "summary_aggregate.csv", aggregate)
    (output_dir / "results.json").write_text(
        json.dumps({"manifest": manifest, "summaries": summaries, "aggregate": aggregate}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(output_dir / "sample_metrics.csv", sample_rows)
    print(json.dumps({"output_dir": str(output_dir.resolve()), "num_summary_rows": len(summaries), "num_sample_rows": len(sample_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
