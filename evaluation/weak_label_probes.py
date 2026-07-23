"""Run frozen probes for weak scalar and slice-type targets."""

from __future__ import annotations

import argparse
import csv
import json
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from pretrain.checkpoints import load_config_from_checkpoint, load_model_state_compat
from pretrain.configs import ExperimentConfig, resolve_path
from pretrain.data.dataset import SampleRecord, TrainNoiseDataset, collate_samples
from pretrain.data.normalization import TrainStats
from pretrain.models.tn_mjepa import TNMJEPA
from pretrain.utils import load_json, save_json, set_seed


SPLITS = ("train", "val", "test")
TASKS = ("mean_vel", "rms_pa_db", "sliceType")
REPRESENTATIONS = ("tn_mjepa", "random_init", "traditional")
CACHE_VERSION = "weak_labels_v1"


def collate_probe_batch(batch: Sequence[Dict[str, object]], pad_to_multiple_of: int) -> Dict[str, object]:
    return collate_samples(batch, pad_to_multiple_of)


def parse_float_list(values: Sequence[str]) -> List[float]:
    return [float(value) for value in values]


def load_checkpoint(path: Path, preset: str, dataset_dir: str, stats_path: str, device: torch.device) -> Tuple[TNMJEPA, ExperimentConfig, Dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = load_config_from_checkpoint(checkpoint, preset, dataset_dir, stats_path)
    model = TNMJEPA(cfg)
    load_model_state_compat(model, checkpoint["model"], checkpoint_path=path)
    model.to(device).eval()
    return model, cfg, checkpoint


def make_probe_datasets(
    cfg: ExperimentConfig,
    stats: TrainStats,
    stats_path: Path,
    metadata_cache: Path,
    max_samples: Optional[int],
) -> Dict[str, TrainNoiseDataset]:
    cached_records = None
    if metadata_cache.exists():
        metadata = {str(record["name"]): record for record in load_json(metadata_cache)["records"]}
        split_files = load_json(stats_path)["split_files"]
        cached_records = {}
        for split in SPLITS:
            records = []
            for name in split_files[split]:
                record = metadata[name]
                records.append(
                    SampleRecord(
                        path=Path(cfg.data.dataset_dir) / name,
                        stem=Path(name).stem,
                        source_tag=str(record["source"]),
                        source_file=str(record["source"]),
                        slice_type=str(record["slice_type"]),
                        mileage_start_km=float("nan"),
                        mileage_end_km=float("nan"),
                        duration_s=float(record["duration_s"]),
                        t_length=int(record.get("n_samples", 0)),
                        f_length=0,
                    )
                )
            cached_records[split] = records

    datasets: Dict[str, TrainNoiseDataset] = {}
    for split in SPLITS:
        dataset = TrainNoiseDataset(
            cfg.data,
            split=split,
            stats=stats,
            aug_cfg=None,
            records=None if cached_records is None else cached_records[split],
        )
        # The training dataset normally uses a fresh random crop.  A probe must
        # see a reproducible frozen input, so use the same index-seeded crop rule
        # for all three partitions without changing the partition membership.
        dataset.split = "probe"
        if max_samples is not None:
            dataset.records = dataset.records[:max_samples]
        datasets[split] = dataset
    return datasets


def inverse_target(meta: torch.Tensor, stats: TrainStats, column: int, kind: str) -> np.ndarray:
    values = meta[:, column].numpy()
    target_stats = getattr(stats, kind)
    return values * float(target_stats.std[0]) + float(target_stats.mean[0])


def _summary_stats(values: np.ndarray) -> List[float]:
    values = values.astype(np.float32, copy=False)
    return [
        float(values.mean()),
        float(values.std()),
        float(np.mean(np.abs(values))),
        float(values.min()),
        float(values.max()),
    ]


def traditional_features(batch: Dict[str, object]) -> np.ndarray:
    """Build a compact signal-only baseline from the frozen input views.

    The TF part uses one-third-octave band summaries.  Pressure and velocity
    use distribution summaries.  Target scalars and the slice-type metadata
    field are intentionally not read.
    """
    tf = batch["tf_log_view1"].numpy()
    pa = batch["pa_view1"].numpy()
    vel = batch["vel_view1"].numpy()
    time_mask = batch["time_mask_view1"].numpy()
    freq_axis = batch["freq_axis"][0].numpy()
    centers = []
    center = 50.0
    while center <= float(freq_axis.max()) * 1.001:
        if center >= float(freq_axis.min()) * 0.99:
            centers.append(center)
        center *= 2.0 ** (1.0 / 3.0)
    band_masks = [
        (freq_axis >= center / 2.0 ** (1.0 / 6.0))
        & (freq_axis < center * 2.0 ** (1.0 / 6.0))
        for center in centers
    ]

    rows: List[List[float]] = []
    for i in range(tf.shape[0]):
        valid = time_mask[i]
        tf_valid = tf[i][:, valid]
        row: List[float] = []
        for band in band_masks:
            band_values = tf_valid[band].reshape(-1)
            row.extend([float(band_values.mean()), float(band_values.std())])
        row.extend(_summary_stats(pa[i, valid]))
        row.extend(_summary_stats(vel[i, valid]))
        row.append(float(batch["meta"][i, 2]))
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def extract_features(
    model: Optional[TNMJEPA],
    dataset: TrainNoiseDataset,
    stats: TrainStats,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    include_traditional: bool,
) -> Dict[str, np.ndarray]:
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": partial(collate_probe_batch, pad_to_multiple_of=dataset.cfg.pad_to_multiple_of),
        "pin_memory": model is not None and device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 1
    loader = DataLoader(dataset, **loader_kwargs)
    normal_features: List[np.ndarray] = []
    masked_features: List[np.ndarray] = []
    traditional: List[np.ndarray] = []
    targets = {"mean_vel": [], "rms_pa_db": [], "sliceType": []}

    if model is not None:
        model.eval()
    for batch in tqdm(loader, desc=f"Extracting {dataset.split}", leave=False):
        targets["mean_vel"].append(inverse_target(batch["meta"], stats, 1, "mean_vel"))
        targets["rms_pa_db"].append(inverse_target(batch["meta"], stats, 0, "rms_pa_db"))
        targets["sliceType"].append(
            np.asarray([1 if record.slice_type == "mileage" else 0 for record in batch["records"]], dtype=np.int64)
        )
        if include_traditional:
            traditional.append(traditional_features(batch))
        if model is None:
            continue

        moved = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}
        with torch.inference_mode():
            encoded = model._encode_online(
                tf_log=moved["tf_log_view1"],
                pa=moved["pa_view1"],
                vel=moved["vel_view1"],
                time_axis=moved["time_view1"],
                time_mask=moved["time_mask_view1"],
                freq_axis=moved["freq_axis"],
                meta=moved["meta"],
            )
            normal_features.append(encoded["fusion"]["pooled"].float().cpu().numpy())

            masked_meta = moved["meta"].clone()
            masked_meta[:, 3] = 0.0
            masked_meta_token = model.meta_mlp(masked_meta)
            masked_fusion = model.fusion(
                encoded["tf"]["tokens"],
                encoded["tf"]["mask"],
                encoded["wave"]["tokens"],
                encoded["wave"]["mask"],
                encoded["vel"]["tokens"],
                encoded["vel"]["mask"],
                masked_meta_token,
            )
            masked_features.append(masked_fusion["pooled"].float().cpu().numpy())

    result = {
        "x": np.concatenate(normal_features, axis=0) if normal_features else np.empty((0, 0), dtype=np.float32),
        "x_masked_slice": np.concatenate(masked_features, axis=0) if masked_features else np.empty((0, 0), dtype=np.float32),
        "mean_vel": np.concatenate(targets["mean_vel"], axis=0).astype(np.float32),
        "rms_pa_db": np.concatenate(targets["rms_pa_db"], axis=0).astype(np.float32),
        "sliceType": np.concatenate(targets["sliceType"], axis=0).astype(np.int64),
    }
    if include_traditional:
        result["x_traditional"] = np.concatenate(traditional, axis=0)
    return result


def cache_path(output_dir: Path, representation: str, seed: Optional[int], split: str, masked: bool = False) -> Path:
    suffix = "_masked_slice" if masked else ""
    seed_name = "shared" if seed is None else f"seed{seed}"
    return output_dir / "cache" / f"{CACHE_VERSION}_{representation}_{seed_name}_{split}{suffix}.npz"


def save_feature_cache(path: Path, features: Dict[str, np.ndarray], x_key: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=features[x_key],
        mean_vel=features["mean_vel"],
        rms_pa_db=features["rms_pa_db"],
        sliceType=features["sliceType"],
    )


def load_feature_cache(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def standardize_train_val_test(arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    train = arrays["train"].astype(np.float32)
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return {split: ((arrays[split] - mean) / scale).astype(np.float32) for split in SPLITS}


def binary_auc(target: np.ndarray, probability: np.ndarray) -> float:
    positive = probability[target == 1]
    negative = probability[target == 0]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    comparisons = positive[:, None] - negative[None, :]
    return float((comparisons > 0).mean() + 0.5 * (comparisons == 0).mean())


def classification_metrics(target: np.ndarray, probability: np.ndarray) -> Dict[str, float]:
    prediction = (probability >= 0.5).astype(np.int64)
    accuracy = float((prediction == target).mean())
    f1_values = []
    for label in (0, 1):
        tp = float(np.sum((prediction == label) & (target == label)))
        fp = float(np.sum((prediction == label) & (target != label)))
        fn = float(np.sum((prediction != label) & (target == label)))
        denominator = 2.0 * tp + fp + fn
        f1_values.append(0.0 if denominator == 0 else 2.0 * tp / denominator)
    return {"accuracy": accuracy, "macro_f1": float(np.mean(f1_values)), "auc": binary_auc(target, probability)}


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    error = prediction - target
    mse = float(np.mean(error**2))
    target_centered = target - target.mean()
    denominator = float(np.sum(target_centered**2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(mse)),
        "r2": float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else float("nan"),
    }


def reference_baselines(features: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, object]:
    """Compute train-only constant and majority references on the test split."""
    result: Dict[str, object] = {
        "regression_reference": "Predict the training-partition target mean for every test sample.",
        "classification_reference": "Predict the training-partition majority class for every test sample.",
    }
    for task in ("mean_vel", "rms_pa_db"):
        train_target = features["train"][task].astype(np.float32)
        test_target = features["test"][task].astype(np.float32)
        train_mean = float(train_target.mean())
        prediction = np.full_like(test_target, train_mean)
        result[task] = {
            "train_mean": train_mean,
            "test_metrics": regression_metrics(test_target, prediction),
        }

    train_labels = features["train"]["sliceType"].astype(np.int64)
    test_labels = features["test"]["sliceType"].astype(np.int64)
    majority_class = int(np.bincount(train_labels, minlength=2).argmax())
    probability = np.full(test_labels.shape, float(majority_class), dtype=np.float32)
    result["sliceType"] = {
        "train_majority_class": "mileage" if majority_class == 1 else "stationary",
        "test_counts": {
            "stationary": int(np.sum(test_labels == 0)),
            "mileage": int(np.sum(test_labels == 1)),
        },
        "test_metrics": classification_metrics(test_labels, probability),
    }
    return result


def fit_probe(
    x: Dict[str, np.ndarray],
    target: Dict[str, np.ndarray],
    task: str,
    seed: int,
    device: torch.device,
    epochs: int,
    patience: int,
    learning_rates: Sequence[float],
    weight_decays: Sequence[float],
) -> Dict[str, object]:
    x_scaled = standardize_train_val_test(x)
    x_train = torch.from_numpy(x_scaled["train"]).to(device)
    x_val = torch.from_numpy(x_scaled["val"]).to(device)
    x_test = torch.from_numpy(x_scaled["test"]).to(device)
    y_train_raw = target["train"].astype(np.float32)
    y_val_raw = target["val"].astype(np.float32)
    y_test_raw = target["test"].astype(np.float32)

    is_classification = task == "sliceType"
    if is_classification:
        y_train = torch.from_numpy(y_train_raw).to(device)
    else:
        target_mean = float(y_train_raw.mean())
        target_scale = float(y_train_raw.std())
        if target_scale < 1e-6:
            target_scale = 1.0
        y_train = torch.from_numpy((y_train_raw - target_mean) / target_scale).to(device)

    best_score = -float("inf") if is_classification else float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_params = None
    best_val_metrics: Dict[str, float] = {}
    for learning_rate in learning_rates:
        for weight_decay in weight_decays:
            set_seed(seed)
            head = nn.Linear(x_train.shape[1], 1).to(device)
            optimizer = AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)
            candidate_best = -float("inf") if is_classification else float("inf")
            candidate_state = None
            candidate_metrics: Dict[str, float] = {}
            stale = 0
            for _ in range(epochs):
                head.train()
                prediction = head(x_train).squeeze(-1)
                loss = (
                    nn.functional.binary_cross_entropy_with_logits(prediction, y_train)
                    if is_classification
                    else nn.functional.mse_loss(prediction, y_train)
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                head.eval()
                with torch.inference_mode():
                    val_prediction = head(x_val).squeeze(-1)
                if is_classification:
                    val_probability = torch.sigmoid(val_prediction).cpu().numpy()
                    metrics = classification_metrics(y_val_raw.astype(np.int64), val_probability)
                    score = metrics["macro_f1"]
                    improved = score > candidate_best + 1e-8
                else:
                    val_raw_prediction = val_prediction.cpu().numpy() * target_scale + target_mean
                    metrics = regression_metrics(y_val_raw, val_raw_prediction)
                    score = metrics["rmse"]
                    improved = score < candidate_best - 1e-8
                if improved:
                    candidate_best = score
                    candidate_metrics = metrics
                    candidate_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                    if stale >= patience:
                        break
            if candidate_state is None:
                raise RuntimeError(f"Probe did not produce a validation checkpoint for task {task}")
            better_global = candidate_best > best_score + 1e-8 if is_classification else candidate_best < best_score - 1e-8
            if better_global:
                best_score = candidate_best
                best_state = candidate_state
                best_params = {"lr": learning_rate, "weight_decay": weight_decay}
                best_val_metrics = candidate_metrics

    assert best_state is not None and best_params is not None
    final_head = nn.Linear(x_train.shape[1], 1).to(device)
    final_head.load_state_dict(best_state)
    final_head.eval()
    with torch.inference_mode():
        test_prediction = final_head(x_test).squeeze(-1)
    if is_classification:
        test_metrics = classification_metrics(y_test_raw.astype(np.int64), torch.sigmoid(test_prediction).cpu().numpy())
    else:
        prediction = test_prediction.cpu().numpy() * target_scale + target_mean
        test_metrics = regression_metrics(y_test_raw, prediction)
    return {
        "task": task,
        "n_train": int(len(y_train_raw)),
        "n_val": int(len(y_val_raw)),
        "n_test": int(len(y_test_raw)),
        "best_params": best_params,
        "val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "target_train_mean": target_mean if not is_classification else None,
        "target_train_std": target_scale if not is_classification else None,
    }


def save_summary_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["representation", "encoder_seed", "probe_seed", "task", "metric", "value"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_summary_rows(results: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for result in results:
        metrics = result["test_metrics"]
        assert isinstance(metrics, dict)
        for metric, value in metrics.items():
            rows.append(
                {
                    "representation": result["representation"],
                    "encoder_seed": result["encoder_seed"],
                    "probe_seed": result["probe_seed"],
                    "task": result["task"],
                    "metric": metric,
                    "value": value,
                }
            )
    return rows


def aggregate_results(results: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str], List[float]] = {}
    for result in results:
        metrics = result["test_metrics"]
        assert isinstance(metrics, dict)
        for metric, value in metrics.items():
            key = (str(result["representation"]), str(result["task"]), metric)
            grouped.setdefault(key, []).append(float(value))

    rows = []
    for (representation, task, metric), values in sorted(grouped.items()):
        array = np.asarray(values, dtype=np.float64)
        rows.append(
            {
                "representation": representation,
                "task": task,
                "metric": metric,
                "n_runs": int(array.size),
                "mean": float(array.mean()),
                "sample_std": float(array.std(ddof=1)) if array.size > 1 else None,
            }
        )
    return rows


def save_aggregate_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["representation", "task", "metric", "n_runs", "mean", "sample_std"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def get_checkpoints(args: argparse.Namespace) -> Dict[int, Path]:
    if args.checkpoint:
        checkpoint = resolve_path(args.checkpoint)
        return {args.seeds[0]: checkpoint}
    return {seed: resolve_path(Path(args.checkpoint_root) / f"full_seed{seed}" / "best.pt") for seed in args.seeds}


def load_or_extract_model_features(
    representation: str,
    encoder_seed: int,
    model: TNMJEPA,
    datasets: Dict[str, TrainNoiseDataset],
    stats: TrainStats,
    output_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Dict[str, np.ndarray]]:
    all_features: Dict[str, Dict[str, np.ndarray]] = {}
    for split in SPLITS:
        normal_path = cache_path(output_dir, representation, encoder_seed, split)
        masked_path = cache_path(output_dir, representation, encoder_seed, split, masked=True)
        if normal_path.exists() and masked_path.exists() and not args.overwrite:
            normal = load_feature_cache(normal_path)
            masked = load_feature_cache(masked_path)
            all_features[split] = {"normal": normal, "masked": masked}
            continue
        extracted = extract_features(
            model=model,
            dataset=datasets[split],
            stats=stats,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            include_traditional=False,
        )
        save_feature_cache(normal_path, extracted, "x")
        save_feature_cache(masked_path, extracted, "x_masked_slice")
        all_features[split] = {
            "normal": {key: extracted[key] for key in ("x", "mean_vel", "rms_pa_db", "sliceType")},
            "masked": {
                "x": extracted["x_masked_slice"],
                "mean_vel": extracted["mean_vel"],
                "rms_pa_db": extracted["rms_pa_db"],
                "sliceType": extracted["sliceType"],
            },
        }
    return all_features


def load_or_extract_traditional(
    datasets: Dict[str, TrainNoiseDataset],
    stats: TrainStats,
    output_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Dict[str, np.ndarray]]:
    result = {}
    for split in SPLITS:
        path = cache_path(output_dir, "traditional", None, split)
        if path.exists() and not args.overwrite:
            result[split] = load_feature_cache(path)
            continue
        extracted = extract_features(
            model=None,
            dataset=datasets[split],
            stats=stats,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            include_traditional=True,
        )
        save_feature_cache(path, {**extracted, "x": extracted["x_traditional"]}, "x")
        result[split] = load_feature_cache(path)
    return result


def evaluate_representation(
    representation: str,
    encoder_seed: int,
    features: Dict[str, Dict[str, np.ndarray]],
    probe_seed: int,
    device: torch.device,
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    results = []
    for task in TASKS:
        masked = task == "sliceType" and representation != "traditional"
        view = "masked" if masked else "normal"
        x = {split: features[split][view]["x"] for split in SPLITS}
        target = {split: features[split][view][task] for split in SPLITS}
        result = fit_probe(
            x=x,
            target=target,
            task=task,
            seed=probe_seed,
            device=device,
            epochs=args.probe_epochs,
            patience=args.probe_patience,
            learning_rates=args.probe_lrs,
            weight_decays=args.probe_weight_decays,
        )
        result.update(
            {
                "representation": representation,
                "encoder_seed": encoder_seed,
                "probe_seed": probe_seed,
                "slice_input_masked": masked,
            }
        )
        results.append(result)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Source-disjoint frozen linear probes for TN-MJEPA weak labels.")
    parser.add_argument("--checkpoint", default=None, help="Single checkpoint compatibility mode.")
    parser.add_argument("--checkpoint-root", default="runs")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--preset", default="tn-mjepa-base")
    parser.add_argument("--dataset-dir", default="sswpt_dataset")
    parser.add_argument("--stats-path", default="pretrain/train_stats.json")
    parser.add_argument("--metadata-cache", default="dataset_metadata.json")
    parser.add_argument("--output-dir", default="evaluation/weak_labels")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--probe-epochs", type=int, default=300)
    parser.add_argument("--probe-patience", type=int, default=35)
    parser.add_argument("--probe-lrs", nargs="+", type=float, default=[1e-3, 3e-4, 1e-4])
    parser.add_argument("--probe-weight-decays", nargs="+", type=float, default=[0.0, 1e-4])
    parser.add_argument("--representations", nargs="+", choices=REPRESENTATIONS, default=list(REPRESENTATIONS))
    parser.add_argument("--max-samples", type=int, default=None, help="Smoke-test limit per split; do not use for paper results.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summarize-only", action="store_true", help="Regenerate aggregate files from an existing results.json.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.checkpoint and len(args.seeds) != 1:
        raise SystemExit("--checkpoint compatibility mode requires exactly one --seeds value")
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        results = load_json(output_dir / "results.json")
        aggregate = aggregate_results(results)
        save_json(output_dir / "summary_aggregate.json", aggregate)
        save_aggregate_csv(output_dir / "summary_aggregate.csv", aggregate)
        traditional_paths = {split: cache_path(output_dir, "traditional", None, split) for split in SPLITS}
        if all(path.exists() for path in traditional_paths.values()):
            traditional_features = {split: load_feature_cache(path) for split, path in traditional_paths.items()}
            save_json(output_dir / "reference_baselines.json", reference_baselines(traditional_features))
        print(json.dumps({"output_dir": str(output_dir), "num_results": len(results)}, ensure_ascii=False))
        return
    device = torch.device(args.device)
    checkpoints = get_checkpoints(args)
    missing = [str(path) for path in checkpoints.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoints: " + ", ".join(missing))

    first_checkpoint = torch.load(next(iter(checkpoints.values())), map_location="cpu", weights_only=False)
    base_cfg = load_config_from_checkpoint(first_checkpoint, args.preset, args.dataset_dir, args.stats_path)
    stats = TrainStats.from_json(resolve_path(args.stats_path))
    datasets = make_probe_datasets(
        base_cfg,
        stats,
        stats_path=resolve_path(args.stats_path),
        metadata_cache=resolve_path(args.metadata_cache),
        max_samples=args.max_samples,
    )
    manifest = {
        "cache_version": CACHE_VERSION,
        "splits": {split: len(dataset) for split, dataset in datasets.items()},
        "tasks": list(TASKS),
        "representations": list(args.representations),
        "seeds": list(checkpoints),
        "device": str(device),
        "dataset_dir": str(resolve_path(args.dataset_dir)),
        "stats_path": str(resolve_path(args.stats_path)),
        "sliceType_input_masked_for_probe": True,
        "slice_type_counts": {
            split: {
                "stationary": sum(record.slice_type != "mileage" for record in dataset.records),
                "mileage": sum(record.slice_type == "mileage" for record in dataset.records),
            }
            for split, dataset in datasets.items()
        },
        "max_samples": args.max_samples,
    }
    save_json(output_dir / "manifest.json", manifest)

    all_results: List[Dict[str, object]] = []
    reference_features: Optional[Dict[str, Dict[str, np.ndarray]]] = None
    traditional_features_by_split = None
    if "traditional" in args.representations:
        traditional_features_by_split = load_or_extract_traditional(datasets, stats, output_dir, device, args)
        reference_features = traditional_features_by_split
        traditional_views = {
            split: {"normal": traditional_features_by_split[split], "masked": traditional_features_by_split[split]}
            for split in SPLITS
        }
        all_results.extend(evaluate_representation("traditional", 0, traditional_views, 42, device, args))

    for seed, checkpoint_path in checkpoints.items():
        if "tn_mjepa" in args.representations:
            model, cfg, _ = load_checkpoint(checkpoint_path, args.preset, args.dataset_dir, args.stats_path, device)
            model_features = load_or_extract_model_features("tn_mjepa", seed, model, datasets, stats, output_dir, device, args)
            if reference_features is None:
                reference_features = {split: model_features[split]["normal"] for split in SPLITS}
            all_results.extend(evaluate_representation("tn_mjepa", seed, model_features, seed, device, args))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if "random_init" in args.representations:
            set_seed(seed)
            random_cfg = load_config_from_checkpoint(first_checkpoint, args.preset, args.dataset_dir, args.stats_path)
            random_model = TNMJEPA(random_cfg).to(device).eval()
            random_features = load_or_extract_model_features("random_init", seed, random_model, datasets, stats, output_dir, device, args)
            if reference_features is None:
                reference_features = {split: random_features[split]["normal"] for split in SPLITS}
            all_results.extend(evaluate_representation("random_init", seed, random_features, seed, device, args))
            del random_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for result in all_results:
        output_name = f"{result['representation']}_encoder{result['encoder_seed']}_{result['task']}.json"
        save_json(output_dir / output_name, result)
    save_json(output_dir / "results.json", all_results)
    save_summary_csv(output_dir / "summary_long.csv", make_summary_rows(all_results))
    aggregate = aggregate_results(all_results)
    save_json(output_dir / "summary_aggregate.json", aggregate)
    save_aggregate_csv(output_dir / "summary_aggregate.csv", aggregate)
    if reference_features is not None:
        save_json(output_dir / "reference_baselines.json", reference_baselines(reference_features))
    print(json.dumps({"output_dir": str(output_dir), "num_results": len(all_results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
