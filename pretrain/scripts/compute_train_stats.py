"""Compute training-only normalization statistics from MAT samples."""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.interpolate import interp1d
from scipy.io import loadmat
from tqdm import tqdm

from pretrain.configs import DataConfig, resolve_path
from pretrain.utils import compute_quantiles, save_json

STAT_VARIABLES = [
    "tf_log",
    "pa_resampled",
    "vel_resampled",
    "rms_pa_db",
    "mean_vel",
    "time_axis",
    "freq_axis",
    "sampleMeta",
]
REF_FREQ_LOW_HZ = 50.0
REF_FREQ_HIGH_HZ = 1500.0
REF_FREQ_BINS = 64


@dataclass
class RunningStats1D:
    count: int = 0
    sum_: float = 0.0
    sum_sq: float = 0.0

    def update_from_sums(self, sum_: float, sum_sq: float, count: int) -> None:
        self.count += count
        self.sum_ += sum_
        self.sum_sq += sum_sq

    def to_dict(self) -> Dict[str, float]:
        mean = self.sum_ / max(self.count, 1)
        var = self.sum_sq / max(self.count, 1) - mean * mean
        return {
            "count": int(self.count),
            "mean": float(mean),
            "std": float(np.sqrt(max(var, 1e-12))),
        }


class RunningStatsPerFreq:
    def __init__(self, num_freq_bins: int) -> None:
        self.num_freq_bins = num_freq_bins
        self.count = 0
        self.sum_ = np.zeros(num_freq_bins, dtype=np.float64)
        self.sum_sq = np.zeros(num_freq_bins, dtype=np.float64)

    def update_from_sums(self, sum_: np.ndarray, sum_sq: np.ndarray, count: int) -> None:
        self.sum_ += sum_
        self.sum_sq += sum_sq
        self.count += count

    def to_dict(self, reference_freq_axis: List[float]) -> Dict[str, object]:
        mean = self.sum_ / max(self.count, 1)
        var = self.sum_sq / max(self.count, 1) - np.square(mean)
        return {
            "count_frames": int(self.count),
            "reference_freq_axis_hz": reference_freq_axis,
            "num_freq_bins": self.num_freq_bins,
            "mean": mean.astype(np.float32).tolist(),
            "std": np.sqrt(np.maximum(var, 1e-12)).astype(np.float32).tolist(),
        }


def build_reference_freq_axis(num_bins: int = REF_FREQ_BINS) -> np.ndarray:
    return np.geomspace(REF_FREQ_LOW_HZ, REF_FREQ_HIGH_HZ, num_bins).astype(np.float32)


def interpolate_tf_to_reference_freq(
    tf_log: np.ndarray,
    freq_axis: np.ndarray,
    reference_freq_axis: np.ndarray,
) -> np.ndarray:
    """Resample one TF map onto the shared reference frequency grid."""
    source_freq = np.asarray(freq_axis, dtype=np.float32).reshape(-1)
    data = np.asarray(tf_log, dtype=np.float32)
    if source_freq.size != data.shape[0]:
        raise ValueError(f"freq_axis length {source_freq.size} != tf_log F {data.shape[0]}")
    if source_freq.size == reference_freq_axis.size and np.allclose(source_freq, reference_freq_axis):
        return data

    if np.all(np.diff(source_freq) >= 0):
        source_freq_sorted = source_freq
        data_sorted = data
    else:
        order = np.argsort(source_freq)
        source_freq_sorted = source_freq[order]
        data_sorted = data[order]

    interp_fn = interp1d(
        source_freq_sorted,
        data_sorted,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value=(data_sorted[0], data_sorted[-1]),
        assume_sorted=True,
    )
    return interp_fn(reference_freq_axis).astype(np.float32)


def parse_sample_meta(meta: np.ndarray) -> Dict[str, object]:
    if isinstance(meta, np.ndarray) and meta.dtype.names:
        out: Dict[str, object] = {}
        item = meta[0, 0] if meta.shape == (1, 1) else meta.reshape(-1)[0]
        for name in meta.dtype.names:
            value = item[name]
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    out[name] = value.reshape(-1)[0].item() if hasattr(value.reshape(-1)[0], "item") else value.reshape(-1)[0]
                else:
                    out[name] = value.squeeze().tolist()
            else:
                out[name] = value
        return out
    return {}


def build_split(files: List[Path], cfg: DataConfig) -> Dict[str, List[Path]]:
    total = len(files)
    n_train = int(total * cfg.train_split)
    n_val = int(total * cfg.val_split)
    if n_train + n_val >= total:
        n_train = max(total - 2, 1)
        n_val = 1 if total > 2 else 0
    train = files[:n_train]
    val = files[n_train : n_train + n_val]
    test = files[n_train + n_val :]
    return {"train": train, "val": val, "test": test}


def build_source_split(
    files: List[Path],
    metadata_cache: Path,
    val_source: str,
    test_source: str,
    train_sources: Optional[List[str]] = None,
) -> Dict[str, List[Path]]:
    payload = json.loads(metadata_cache.read_text(encoding="utf-8"))
    source_by_name = {record["name"]: record["source"] for record in payload["records"]}
    splits: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}
    for path in files:
        source = source_by_name.get(path.name)
        if source is None:
            raise KeyError(f"{path.name} is missing from {metadata_cache}")
        if source == test_source:
            split = "test"
        elif source == val_source:
            split = "val"
        elif train_sources is None or source in train_sources:
            split = "train"
        else:
            raise ValueError(f"Source {source!r} is not assigned by the source-split protocol")
        splits[split].append(path)
    if not all(splits.values()):
        raise ValueError(f"Source split produced an empty partition: { {k: len(v) for k, v in splits.items()} }")
    return splits


@dataclass
class FileStatResult:
    file_name: str
    f_value: int
    t_value: int
    duration: float
    slice_type: str
    tf_sum: np.ndarray
    tf_sumsq: np.ndarray
    tf_count: int
    pa_sum: float
    pa_sumsq: float
    pa_count: int
    vel_sum: float
    vel_sumsq: float
    vel_count: int
    rms_val: float
    mean_vel_val: float
    error: Optional[str] = None


def process_one_file(path: Path, reference_freq_axis: np.ndarray) -> FileStatResult:
    """Load one sample and return compact sufficient statistics."""
    try:
        mat = loadmat(path, variable_names=STAT_VARIABLES, squeeze_me=True, struct_as_record=False)
        tf_log_raw = np.asarray(mat["tf_log"], dtype=np.float32)
        freq_axis = np.asarray(mat["freq_axis"], dtype=np.float32).reshape(-1)
        original_f = int(tf_log_raw.shape[0])
        tf_log = interpolate_tf_to_reference_freq(tf_log_raw, freq_axis, reference_freq_axis)

        pa = np.asarray(mat["pa_resampled"], dtype=np.float64).reshape(-1)
        vel = np.asarray(mat["vel_resampled"], dtype=np.float64).reshape(-1)
        rms = float(np.asarray(mat["rms_pa_db"]).reshape(-1)[0])
        mean_vel = float(np.asarray(mat["mean_vel"]).reshape(-1)[0])
        time_axis = np.asarray(mat["time_axis"], dtype=np.float32).reshape(-1)
        meta = parse_sample_meta(mat.get("sampleMeta", np.array([])))

        data64 = tf_log.astype(np.float64)
        t_value = int(tf_log.shape[1])
        duration = float(time_axis[-1]) if time_axis.size else float(meta.get("duration_s", 0.0) or 0.0)

        return FileStatResult(
            file_name=path.name,
            f_value=original_f,
            t_value=t_value,
            duration=duration,
            slice_type=str(meta.get("sliceType", "unknown")),
            tf_sum=data64.sum(axis=1),
            tf_sumsq=np.square(data64).sum(axis=1),
            tf_count=t_value,
            pa_sum=float(pa.sum()),
            pa_sumsq=float(np.square(pa).sum()),
            pa_count=int(pa.size),
            vel_sum=float(vel.sum()),
            vel_sumsq=float(np.square(vel).sum()),
            vel_count=int(vel.size),
            rms_val=rms,
            mean_vel_val=mean_vel,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to caller via error field
        return FileStatResult(
            file_name=path.name,
            f_value=-1,
            t_value=0,
            duration=0.0,
            slice_type="error",
            tf_sum=np.zeros_like(reference_freq_axis, dtype=np.float64),
            tf_sumsq=np.zeros_like(reference_freq_axis, dtype=np.float64),
            tf_count=0,
            pa_sum=0.0,
            pa_sumsq=0.0,
            pa_count=0,
            vel_sum=0.0,
            vel_sumsq=0.0,
            vel_count=0,
            rms_val=0.0,
            mean_vel_val=0.0,
            error=f"{type(exc).__name__}: {exc}",
        )


def compute_stats(
    dataset_dir: Path,
    output_path: Path,
    cfg: DataConfig,
    num_workers: Optional[int] = None,
    splits_override: Optional[Dict[str, List[Path]]] = None,
    split_sources: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, object]:
    files = sorted([p for p in dataset_dir.glob("*.mat") if p.name.lower() != "train_stats.mat"])
    if not files:
        raise FileNotFoundError(f"No .mat samples found in {dataset_dir}")

    splits = splits_override if splits_override is not None else build_split(files, cfg)
    train_files = splits["train"]
    reference_freq_axis = build_reference_freq_axis()

    tf_stats = RunningStatsPerFreq(num_freq_bins=reference_freq_axis.size)
    pa_stats = RunningStats1D()
    vel_stats = RunningStats1D()
    rms_stats = RunningStats1D()
    mean_vel_stats = RunningStats1D()

    lengths_t: List[int] = []
    durations: List[float] = []
    ratios_t_per_sec: List[float] = []
    f_values: List[int] = []
    f_histogram: Dict[str, int] = {}
    sample_summaries: List[Dict[str, object]] = []
    failed_files: List[Dict[str, str]] = []

    workers = num_workers or min(32, max(1, os.cpu_count() or 4))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_one_file, path, reference_freq_axis): path for path in train_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing train stats", unit="sample"):
            result = future.result()
            if result.error is not None:
                failed_files.append({"file": result.file_name, "error": result.error})
                continue

            tf_stats.update_from_sums(result.tf_sum, result.tf_sumsq, result.tf_count)
            pa_stats.update_from_sums(result.pa_sum, result.pa_sumsq, result.pa_count)
            vel_stats.update_from_sums(result.vel_sum, result.vel_sumsq, result.vel_count)
            rms_stats.update_from_sums(result.rms_val, result.rms_val**2, 1)
            mean_vel_stats.update_from_sums(result.mean_vel_val, result.mean_vel_val**2, 1)

            f_values.append(result.f_value)
            f_histogram[str(result.f_value)] = f_histogram.get(str(result.f_value), 0) + 1
            lengths_t.append(result.t_value)
            durations.append(result.duration)
            if result.duration > 0:
                ratios_t_per_sec.append(result.t_value / result.duration)

            if len(sample_summaries) < 10:
                sample_summaries.append(
                    {
                        "file": result.file_name,
                        "F": result.f_value,
                        "T": result.t_value,
                        "duration_s": result.duration,
                        "slice_type": result.slice_type,
                    }
                )

    if failed_files:
        print(f"WARNING: {len(failed_files)} files failed to process (see 'failed_files' in output JSON).")

    unique_f = sorted(set(f_values))
    t_quantiles = compute_quantiles(lengths_t, [0.5, 0.9, 0.95, 0.98, 0.99, 1.0])
    duration_quantiles = compute_quantiles(durations, [0.5, 0.9, 0.95, 0.98, 0.99, 1.0])
    t_per_sec_quantiles = compute_quantiles(ratios_t_per_sec, [0.5, 0.9, 0.95, 0.99, 1.0])
    suggested_t_max = int(np.quantile(lengths_t, cfg.crop_quantile)) if lengths_t else None

    output = {
        "dataset_dir": str(dataset_dir),
        "num_files_total": len(files),
        "num_files_train": len(splits["train"]),
        "num_files_val": len(splits["val"]),
        "num_files_test": len(splits["test"]),
        "split_files": {
            key: [p.name for p in value] for key, value in splits.items()
        },
        "split_sources": split_sources,
        "tf_log": tf_stats.to_dict(reference_freq_axis.tolist()),
        "pa_resampled": pa_stats.to_dict(),
        "vel_resampled": vel_stats.to_dict(),
        "rms_pa_db": rms_stats.to_dict(),
        "mean_vel": mean_vel_stats.to_dict(),
        "shape_validation": {
            "unique_F_values": unique_f,
            "F_is_constant": len(unique_f) == 1,
            "num_unique_F": len(unique_f),
            "F_histogram_top20": dict(sorted(f_histogram.items(), key=lambda item: item[1], reverse=True)[:20]),
            "reference_freq_bins": int(reference_freq_axis.size),
            "sample_examples": sample_summaries,
        },
        "length_distribution": {
            "T_quantiles": t_quantiles,
            "duration_s_quantiles": duration_quantiles,
            "T_per_second_quantiles": t_per_sec_quantiles,
            "suggested_T_max": suggested_t_max,
        },
        "failed_files": failed_files,
        "config": asdict(cfg),
    }
    save_json(output_path, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute train statistics for TN-MJEPA.")
    parser.add_argument("--dataset-dir", default="sswpt_dataset", help="Directory containing exported .mat samples.")
    parser.add_argument("--output", default="pretrain/train_stats.json", help="Path to output JSON stats file.")
    parser.add_argument("--crop-quantile", type=float, default=0.98, help="Quantile used to suggest T_max.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes. Defaults to min(32, cpu_count()).",
    )
    parser.add_argument("--metadata-cache", default=None, help="Metadata JSON used to construct a source-disjoint split.")
    parser.add_argument("--val-source", default=None)
    parser.add_argument("--test-source", default=None)
    parser.add_argument(
        "--protocol",
        default=None,
        help="Optional JSON protocol containing source_split.train/val/test source IDs.",
    )
    args = parser.parse_args()

    cfg = DataConfig(dataset_dir=args.dataset_dir, stats_path=args.output, crop_quantile=args.crop_quantile)
    dataset_dir = resolve_path(cfg.dataset_dir)
    output_path = resolve_path(cfg.stats_path)
    splits_override = None
    split_sources = None
    train_sources = None
    if args.protocol:
        protocol = json.loads(resolve_path(args.protocol).read_text(encoding="utf-8"))
        source_split = protocol["source_split"]
        train_sources = list(source_split["train"])
        args.val_source = str(source_split["val"])
        args.test_source = str(source_split["test"])
        split_sources = {
            "train": train_sources,
            "val": [args.val_source],
            "test": [args.test_source],
        }
    source_args = [args.metadata_cache, args.val_source, args.test_source]
    if any(source_args):
        if not all(source_args):
            parser.error("--metadata-cache, --val-source, and --test-source must be provided together")
        files = sorted([p for p in dataset_dir.glob("*.mat") if p.name.lower() != "train_stats.mat"])
        splits_override = build_source_split(
            files,
            resolve_path(args.metadata_cache),
            args.val_source,
            args.test_source,
            train_sources,
        )
        if args.protocol:
            expected = protocol.get("expected_samples", {})
            for split, paths in splits_override.items():
                if split in expected and len(paths) != int(expected[split]):
                    raise ValueError(
                        f"{split} count mismatch: expected {expected[split]}, got {len(paths)}"
                    )
    stats = compute_stats(
        dataset_dir=dataset_dir,
        output_path=output_path,
        cfg=cfg,
        num_workers=args.num_workers,
        splits_override=splits_override,
        split_sources=split_sources,
    )
    print(f"Saved stats to {output_path}")
    print(f"F constant: {stats['shape_validation']['F_is_constant']}, num unique F={stats['shape_validation']['num_unique_F']}")
    print(f"Suggested T_max (q={args.crop_quantile}): {stats['length_distribution']['suggested_T_max']}")
    if stats["failed_files"]:
        print(f"Failed files: {len(stats['failed_files'])}")


if __name__ == "__main__":
    main()
