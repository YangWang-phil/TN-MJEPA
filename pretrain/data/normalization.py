"""Training-only statistics and frequency-axis normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.interpolate import interp1d

from pretrain.utils import load_json


@dataclass
class TensorStats:
    mean: np.ndarray
    std: np.ndarray


class TrainStats:
    def __init__(self, stats: Dict[str, object]) -> None:
        tf_stats = stats["tf_log"]
        self.reference_freq_axis = np.asarray(
            tf_stats.get("reference_freq_axis_hz", []),
            dtype=np.float32,
        )
        self.tf_log = TensorStats(
            mean=np.asarray(tf_stats["mean"], dtype=np.float32),
            std=np.asarray(tf_stats["std"], dtype=np.float32),
        )
        self.pa_resampled = TensorStats(
            mean=np.asarray([stats["pa_resampled"]["mean"]], dtype=np.float32),
            std=np.asarray([stats["pa_resampled"]["std"]], dtype=np.float32),
        )
        self.vel_resampled = TensorStats(
            mean=np.asarray([stats["vel_resampled"]["mean"]], dtype=np.float32),
            std=np.asarray([stats["vel_resampled"]["std"]], dtype=np.float32),
        )
        self.rms_pa_db = TensorStats(
            mean=np.asarray([stats["rms_pa_db"]["mean"]], dtype=np.float32),
            std=np.asarray([stats["rms_pa_db"]["std"]], dtype=np.float32),
        )
        self.mean_vel = TensorStats(
            mean=np.asarray([stats["mean_vel"]["mean"]], dtype=np.float32),
            std=np.asarray([stats["mean_vel"]["std"]], dtype=np.float32),
        )
        self.length_distribution = stats.get("length_distribution", {})

    @classmethod
    def from_json(cls, path: Path) -> "TrainStats":
        return cls(load_json(path))

    def align_tf_log(self, tf_log: np.ndarray, freq_axis: np.ndarray) -> np.ndarray:
        """Resample tf_log onto the reference frequency grid.

        This runs on every __getitem__ call during training (unlike the
        one-off stats computation), so it must stay vectorized: a per-time-
        column ``np.interp`` loop takes minutes on the largest samples
        (F~1451, T~2e5) vs. ~2s with ``interp1d(..., axis=0)``.
        """
        if self.reference_freq_axis.size == 0:
            return tf_log
        source_freq = np.asarray(freq_axis, dtype=np.float32).reshape(-1)
        data = np.asarray(tf_log, dtype=np.float32)
        if source_freq.size == self.reference_freq_axis.size and np.allclose(source_freq, self.reference_freq_axis):
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
        return interp_fn(self.reference_freq_axis).astype(np.float32)

    def normalize_tf_log(self, tf_log: np.ndarray, freq_axis: np.ndarray | None = None) -> np.ndarray:
        if freq_axis is not None:
            tf_log = self.align_tf_log(tf_log, freq_axis)
        mean = self.tf_log.mean[:, None]
        std = self.tf_log.std[:, None]
        return (tf_log - mean) / np.maximum(std, 1e-6)

    def normalize_signal(self, x: np.ndarray, kind: str) -> np.ndarray:
        stats = getattr(self, kind)
        return (x - stats.mean) / np.maximum(stats.std, 1e-6)
