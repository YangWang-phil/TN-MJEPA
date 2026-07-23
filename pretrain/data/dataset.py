"""MAT-file dataset, source-aware records, bucketing, and collation."""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.io import loadmat, whosmat
from torch.utils.data import BatchSampler, Dataset

from pretrain.configs import AugmentationConfig, DataConfig
from pretrain.data.augmentations import make_two_views
from pretrain.data.normalization import TrainStats
from pretrain.metadata import METADATA_TENSOR_FIELDS
from pretrain.utils import load_json, pad_to_multiple, pairwise_overlap

RECORD_META_VARIABLES = ["sampleMeta", "time_axis"]
SAMPLE_VARIABLES = [
    "tf_log",
    "pa_resampled",
    "vel_resampled",
    "rms_pa_db",
    "mean_vel",
    "freq_axis",
    "time_axis",
    "sampleMeta",
]


def mat_struct_to_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "_fieldnames"):
        out: Dict[str, Any] = {}
        for name in obj._fieldnames:
            value = getattr(obj, name)
            out[name] = mat_value_to_python(value)
        return out
    return {}


def mat_value_to_python(value: Any) -> Any:
    if hasattr(value, "_fieldnames"):
        return mat_struct_to_dict(value)
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return mat_value_to_python(value.item())
        if value.size == 1:
            return mat_value_to_python(value.reshape(-1)[0])
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass
class SampleRecord:
    path: Path
    stem: str
    source_tag: str
    source_file: str
    slice_type: str
    mileage_start_km: float
    mileage_end_km: float
    duration_s: float
    t_length: int
    f_length: int


def read_mat_sample(path: Path) -> Dict[str, Any]:
    mat = loadmat(path, variable_names=SAMPLE_VARIABLES, squeeze_me=True, struct_as_record=False)
    meta = mat_struct_to_dict(mat["sampleMeta"]) if "sampleMeta" in mat else {}
    return {
        "tf_log": np.asarray(mat["tf_log"], dtype=np.float32),
        "pa_resampled": np.asarray(mat["pa_resampled"], dtype=np.float32).reshape(-1),
        "vel_resampled": np.asarray(mat["vel_resampled"], dtype=np.float32).reshape(-1),
        "rms_pa_db": float(np.asarray(mat["rms_pa_db"]).reshape(-1)[0]),
        "mean_vel": float(np.asarray(mat["mean_vel"]).reshape(-1)[0]),
        "freq_axis": np.asarray(mat["freq_axis"], dtype=np.float32).reshape(-1),
        "time_axis": np.asarray(mat["time_axis"], dtype=np.float32).reshape(-1),
        "sample_meta": meta,
    }


def infer_source_file(source_tag: str) -> str:
    parts = source_tag.split("_")
    if len(parts) >= 2 and parts[-2].isdigit():
        return "_".join(parts[:-2])
    return source_tag


def read_mat_record_meta(path: Path) -> Dict[str, Any]:
    """Read metadata and array shapes without loading signal tensors."""
    mat = loadmat(path, variable_names=RECORD_META_VARIABLES, squeeze_me=True, struct_as_record=False)
    meta = mat_struct_to_dict(mat["sampleMeta"]) if "sampleMeta" in mat else {}
    time_axis = np.asarray(mat.get("time_axis", np.array([])), dtype=np.float32).reshape(-1)
    f_length = 0
    t_length = 0
    for name, shape, _ in whosmat(path):
        if name == "tf_log" and len(shape) == 2:
            f_length, t_length = int(shape[0]), int(shape[1])
            break
    return {"sample_meta": meta, "time_axis": time_axis, "f_length": f_length, "t_length": t_length}


def _build_record(path: Path) -> SampleRecord:
    info = read_mat_record_meta(path)
    meta = info["sample_meta"]
    source_tag = str(meta.get("sourceTag", path.stem))
    time_axis = info["time_axis"]
    return SampleRecord(
        path=path,
        stem=path.stem,
        source_tag=source_tag,
        source_file=infer_source_file(source_tag),
        slice_type=str(meta.get("sliceType", "unknown")),
        mileage_start_km=float(meta.get("mileageStart_km", float("nan"))),
        mileage_end_km=float(meta.get("mileageEnd_km", float("nan"))),
        duration_s=float(meta.get("duration_s", time_axis[-1] if time_axis.size else 0.0)),
        t_length=info["t_length"],
        f_length=info["f_length"],
    )


def discover_records(dataset_dir: Path, num_workers: Optional[int] = None) -> List[SampleRecord]:
    paths = sorted(p for p in dataset_dir.glob("*.mat") if p.name.lower() != "train_stats.mat")
    if not paths:
        return []

    workers = num_workers if num_workers is not None else min(32, max(1, os.cpu_count() or 4))
    if workers <= 1 or len(paths) < 64:
        return [_build_record(path) for path in paths]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        records = list(executor.map(_build_record, paths, chunksize=32))
    records.sort(key=lambda record: record.path)
    return records


def split_records(records: List[SampleRecord], stats_path: Optional[Path], split: str) -> List[SampleRecord]:
    if stats_path is not None and stats_path.exists():
        split_files = load_json(stats_path).get("split_files", {})
        names = set(split_files.get(split, []))
        if names:
            return [record for record in records if record.path.name in names]
    n = len(records)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    if split == "train":
        return records[:n_train]
    if split == "val":
        return records[n_train : n_train + n_val]
    return records[n_train + n_val :]


def fixed_length_resize(sample: Dict[str, Any], target_t: int) -> Dict[str, Any]:
    tf_log = sample["tf_log"]
    cur_t = tf_log.shape[1]
    if cur_t == target_t:
        return sample
    q = np.linspace(0.0, 1.0, target_t)
    p = np.linspace(0.0, 1.0, cur_t)
    tf_interp_fn = interp1d(p, tf_log, axis=1, kind="linear", assume_sorted=True)
    tf_resized = tf_interp_fn(q).astype(np.float32)
    pa_resized = np.interp(q, p, sample["pa_resampled"]).astype(np.float32)
    vel_resized = np.interp(q, p, sample["vel_resampled"]).astype(np.float32)
    time_axis = np.interp(q, p, sample["time_axis"]).astype(np.float32)
    out = dict(sample)
    out["tf_log"] = tf_resized
    out["pa_resampled"] = pa_resized
    out["vel_resampled"] = vel_resized
    out["time_axis"] = time_axis
    return out


def crop_long_sample(sample: Dict[str, Any], t_max: int, rng: np.random.Generator) -> Dict[str, Any]:
    cur_t = sample["tf_log"].shape[1]
    if cur_t <= t_max:
        return sample
    start = int(rng.integers(0, cur_t - t_max + 1))
    end = start + t_max
    out = dict(sample)
    out["tf_log"] = sample["tf_log"][:, start:end]
    out["pa_resampled"] = sample["pa_resampled"][start:end]
    out["vel_resampled"] = sample["vel_resampled"][start:end]
    out["time_axis"] = sample["time_axis"][start:end]
    return out


class TrainNoiseDataset(Dataset):
    def __init__(
        self,
        cfg: DataConfig,
        split: str = "train",
        stats: Optional[TrainStats] = None,
        aug_cfg: Optional[AugmentationConfig] = None,
        records: Optional[Sequence[SampleRecord]] = None,
    ) -> None:
        self.cfg = cfg
        self.dataset_dir = Path(cfg.dataset_dir)
        self.records = list(records) if records is not None else discover_records(self.dataset_dir)
        stats_path = Path(cfg.stats_path) if cfg.stats_path else None
        if records is None:
            self.records = split_records(self.records, stats_path=stats_path, split=split)
        self.stats = stats
        self.aug_cfg = aug_cfg
        self.split = split

        if cfg.max_time_tokens is not None:
            self.t_max = cfg.max_time_tokens
        elif stats is not None and stats.length_distribution.get("suggested_T_max") is not None:
            self.t_max = int(stats.length_distribution["suggested_T_max"])
        else:
            self.t_max = None

    def __len__(self) -> int:
        return len(self.records)

    def _normalize(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if self.stats is None:
            return sample
        out = dict(sample)
        out["tf_log"] = self.stats.normalize_tf_log(out["tf_log"], out["freq_axis"]).astype(np.float32)
        if self.stats.reference_freq_axis.size:
            out["freq_axis"] = self.stats.reference_freq_axis.copy()
        out["pa_resampled"] = self.stats.normalize_signal(out["pa_resampled"], "pa_resampled").astype(np.float32)
        out["vel_resampled"] = self.stats.normalize_signal(out["vel_resampled"], "vel_resampled").astype(np.float32)
        out["rms_pa_db_norm"] = self.stats.normalize_signal(np.asarray([out["rms_pa_db"]], dtype=np.float32), "rms_pa_db")[0]
        out["mean_vel_norm"] = self.stats.normalize_signal(np.asarray([out["mean_vel"]], dtype=np.float32), "mean_vel")[0]
        return out

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        rng = np.random.default_rng(seed=(self.cfg.random_seed + index) if self.split != "train" else None)
        sample = read_mat_sample(record.path)
        sample["record"] = record

        if self.cfg.fixed_length_mode:
            sample = fixed_length_resize(sample, self.cfg.fixed_t0)
        elif self.t_max is not None:
            sample = crop_long_sample(sample, self.t_max, rng)

        sample = self._normalize(sample)

        if self.aug_cfg is not None and getattr(self.aug_cfg, "enable", True):
            view1, view2 = make_two_views(sample, self.aug_cfg, seed=int(rng.integers(0, 1_000_000)))
            sample["view1"] = view1
            sample["view2"] = view2
        else:
            sample["view1"] = sample
            sample["view2"] = sample
        return sample


class BucketBatchSampler(BatchSampler):
    def __init__(self, records: Sequence[SampleRecord], boundaries: Sequence[int], batch_size: int, drop_last: bool = False) -> None:
        self.records = list(records)
        self.boundaries = list(boundaries)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.buckets = self._build_buckets()

    def _bucket_id(self, t_length: int) -> int:
        for i, boundary in enumerate(self.boundaries):
            if t_length <= boundary:
                return i
        return len(self.boundaries)

    def _build_buckets(self) -> Dict[int, List[int]]:
        buckets: Dict[int, List[int]] = {}
        for idx, record in enumerate(self.records):
            buckets.setdefault(self._bucket_id(record.t_length), []).append(idx)
        return buckets

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng()
        for bucket_indices in self.buckets.values():
            perm = rng.permutation(bucket_indices).tolist()
            for i in range(0, len(perm), self.batch_size):
                batch = perm[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self) -> int:
        count = 0
        for bucket_indices in self.buckets.values():
            count += len(bucket_indices) // self.batch_size
            if len(bucket_indices) % self.batch_size and not self.drop_last:
                count += 1
        return count


def build_positive_map(records: Sequence[SampleRecord]) -> Dict[int, List[int]]:
    positives: Dict[int, List[int]] = {idx: [] for idx in range(len(records))}
    for i, a in enumerate(records):
        if a.slice_type != "mileage":
            continue
        for j in range(i + 1, len(records)):
            b = records[j]
            if b.slice_type != "mileage" or a.source_file != b.source_file:
                continue
            if any(math.isnan(v) for v in [a.mileage_start_km, a.mileage_end_km, b.mileage_start_km, b.mileage_end_km]):
                continue
            overlap = pairwise_overlap(a.mileage_start_km, a.mileage_end_km, b.mileage_start_km, b.mileage_end_km)
            if overlap > 0:
                positives[i].append(j)
                positives[j].append(i)
    return positives


def collate_samples(batch: Sequence[Dict[str, Any]], pad_to_multiple_of: int = 8) -> Dict[str, Any]:
    f_dim = batch[0]["tf_log"].shape[0]
    max_t = max(
        max(item["view1"]["tf_log"].shape[1] for item in batch),
        max(item["view2"]["tf_log"].shape[1] for item in batch),
    )
    max_t = pad_to_multiple(max_t, pad_to_multiple_of)
    batch_size = len(batch)

    def alloc_tf() -> torch.Tensor:
        return torch.zeros(batch_size, f_dim, max_t, dtype=torch.float32)

    def alloc_seq() -> torch.Tensor:
        return torch.zeros(batch_size, max_t, dtype=torch.float32)

    tf_v1, tf_v2 = alloc_tf(), alloc_tf()
    pa_v1, pa_v2 = alloc_seq(), alloc_seq()
    vel_v1, vel_v2 = alloc_seq(), alloc_seq()
    time_v1, time_v2 = alloc_seq(), alloc_seq()
    mask_v1 = torch.zeros(batch_size, max_t, dtype=torch.bool)
    mask_v2 = torch.zeros(batch_size, max_t, dtype=torch.bool)
    freq_axis = torch.stack([torch.from_numpy(item["freq_axis"]).float() for item in batch], dim=0)
    # Column order follows METADATA_TENSOR_FIELDS. The first two values are
    # auxiliary targets; only duration_s and slice_type enter MetaMLP.
    meta = torch.tensor(
        [
            [
                float(item.get("rms_pa_db_norm", item["rms_pa_db"])),
                float(item.get("mean_vel_norm", item["mean_vel"])),
                float(item["record"].duration_s),
                1.0 if item["record"].slice_type == "mileage" else 0.0,
            ]
            for item in batch
        ],
        dtype=torch.float32,
    )
    if meta.shape[1] != len(METADATA_TENSOR_FIELDS):
        raise RuntimeError("Metadata tensor does not match the declared column contract")

    for i, item in enumerate(batch):
        for view_name, tf_buf, pa_buf, vel_buf, time_buf, mask_buf in [
            ("view1", tf_v1, pa_v1, vel_v1, time_v1, mask_v1),
            ("view2", tf_v2, pa_v2, vel_v2, time_v2, mask_v2),
        ]:
            view = item[view_name]
            cur_t = view["tf_log"].shape[1]
            tf_buf[i, :, :cur_t] = torch.from_numpy(view["tf_log"]).float()
            pa_buf[i, :cur_t] = torch.from_numpy(view["pa_resampled"]).float()
            vel_buf[i, :cur_t] = torch.from_numpy(view["vel_resampled"]).float()
            time_buf[i, :cur_t] = torch.from_numpy(view["time_axis"]).float()
            mask_buf[i, :cur_t] = True

    return {
        "tf_log_view1": tf_v1,
        "tf_log_view2": tf_v2,
        "pa_view1": pa_v1,
        "pa_view2": pa_v2,
        "vel_view1": vel_v1,
        "vel_view2": vel_v2,
        "time_view1": time_v1,
        "time_view2": time_v2,
        "time_mask_view1": mask_v1,
        "time_mask_view2": mask_v2,
        "freq_axis": freq_axis,
        "meta": meta,
        "records": [item["record"] for item in batch],
    }
