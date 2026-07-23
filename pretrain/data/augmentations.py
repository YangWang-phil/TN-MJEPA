"""Joint multimodal crops and stochastic training views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from pretrain.configs import AugmentationConfig


@dataclass
class JointCrop:
    start: int
    end: int


def sample_joint_crop(length: int, cfg: AugmentationConfig, rng: np.random.Generator) -> JointCrop:
    if length <= 1:
        return JointCrop(start=0, end=length)
    ratio = rng.uniform(cfg.joint_crop_min_ratio, cfg.joint_crop_max_ratio)
    crop_len = max(1, int(round(length * ratio)))
    crop_len = min(crop_len, length)
    start = int(rng.integers(0, max(length - crop_len + 1, 1)))
    return JointCrop(start=start, end=start + crop_len)


def apply_joint_crop(sample: Dict[str, np.ndarray], crop: JointCrop) -> Dict[str, np.ndarray]:
    out = dict(sample)
    start, end = crop.start, crop.end
    out["tf_log"] = sample["tf_log"][:, start:end]
    out["pa_resampled"] = sample["pa_resampled"][start:end]
    out["vel_resampled"] = sample["vel_resampled"][start:end]
    out["time_axis"] = sample["time_axis"][start:end]
    return out


def mask_axis(arr: np.ndarray, axis: int, width: int, rng: np.random.Generator) -> np.ndarray:
    out = arr.copy()
    dim = out.shape[axis]
    if dim <= 1 or width <= 0:
        return out
    width = min(width, dim)
    start = int(rng.integers(0, dim - width + 1))
    index = [slice(None)] * out.ndim
    index[axis] = slice(start, start + width)
    out[tuple(index)] = 0.0
    return out


def shift_1d(arr: np.ndarray, max_shift: int, rng: np.random.Generator) -> np.ndarray:
    if arr.size <= 1 or max_shift <= 0:
        return arr
    shift = int(rng.integers(-max_shift, max_shift + 1))
    if shift == 0:
        return arr
    out = np.roll(arr, shift)
    if shift > 0:
        out[:shift] = 0.0
    else:
        out[shift:] = 0.0
    return out


def augment_sample(sample: Dict[str, np.ndarray], cfg: AugmentationConfig, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    if not cfg.enable:
        return dict(sample)

    out = dict(sample)
    crop = sample_joint_crop(out["tf_log"].shape[1], cfg, rng)
    out = apply_joint_crop(out, crop)

    if rng.random() < cfg.tf_time_mask_prob:
        out["tf_log"] = mask_axis(out["tf_log"], axis=1, width=cfg.tf_time_mask_width, rng=rng)
    if rng.random() < cfg.tf_freq_mask_prob:
        out["tf_log"] = mask_axis(out["tf_log"], axis=0, width=cfg.tf_freq_mask_width, rng=rng)
    out["tf_log"] = out["tf_log"] + rng.normal(0.0, cfg.tf_log_noise_std, size=out["tf_log"].shape).astype(np.float32)

    pa = out["pa_resampled"].copy()
    gain = float(rng.uniform(-cfg.pa_gain_db, cfg.pa_gain_db))
    pa = pa * (10.0 ** (gain / 20.0))
    pa = pa + rng.normal(0.0, max(np.std(pa), 1e-6) * cfg.pa_noise_std_ratio, size=pa.shape).astype(np.float32)
    if rng.random() < cfg.pa_dropout_prob:
        width = min(cfg.pa_dropout_width, pa.shape[0])
        start = int(rng.integers(0, max(pa.shape[0] - width + 1, 1)))
        pa[start : start + width] = 0.0
    out["pa_resampled"] = shift_1d(pa, cfg.pa_shift_max, rng)

    vel = out["vel_resampled"].copy()
    vel = vel + rng.normal(0.0, cfg.vel_noise_std, size=vel.shape).astype(np.float32)
    if rng.random() < cfg.vel_mask_prob and vel.size > 1:
        width = min(cfg.vel_mask_width, vel.shape[0])
        start = int(rng.integers(0, max(vel.shape[0] - width + 1, 1)))
        vel[start : start + width] = 0.0
    out["vel_resampled"] = vel
    return out


def make_two_views(sample: Dict[str, np.ndarray], cfg: AugmentationConfig, seed: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed + 1)
    return augment_sample(sample, cfg, rng1), augment_sample(sample, cfg, rng2)
