"""Strict checkpoint loading with documented legacy compatibility."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch

from pretrain.configs import ExperimentConfig, get_preset, resolve_path


LEGACY_UNUSED_PREFIXES = ("target_fusion.", "target_meta_mlp.")


def _update_known_fields(target: object, values: Mapping[str, object]) -> None:
    """Restore only fields that still exist in the public configuration."""
    for name, value in values.items():
        if hasattr(target, name):
            setattr(target, name, value)


def load_config_from_checkpoint(
    checkpoint: Mapping[str, object],
    preset: str,
    dataset_dir: str,
    stats_path: str,
) -> ExperimentConfig:
    cfg = get_preset(preset)
    saved = checkpoint.get("config", {})
    if isinstance(saved, Mapping):
        for section in ("data", "augmentation", "model", "loss", "train", "eval"):
            values = saved.get(section)
            if isinstance(values, Mapping):
                _update_known_fields(getattr(cfg, section), values)
    cfg.data.dataset_dir = str(resolve_path(dataset_dir))
    cfg.data.stats_path = str(resolve_path(stats_path))
    cfg.model.use_grad_checkpoint = False
    return cfg


def load_model_state_compat(
    model: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    checkpoint_path: str | Path = "<in-memory>",
) -> tuple[str, ...]:
    """Load current and legacy paper checkpoints without hiding mismatches.

    Early checkpoints stored unused EMA metadata/fusion branches. Those keys
    are ignored because the public model correctly keeps only modality target
    encoders. Any missing key or any other unexpected key remains an error.
    """
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = tuple(incompatible.missing_keys)
    unexpected = tuple(incompatible.unexpected_keys)
    disallowed = tuple(
        key for key in unexpected if not key.startswith(LEGACY_UNUSED_PREFIXES)
    )
    if missing or disallowed:
        raise RuntimeError(
            f"Checkpoint mismatch for {checkpoint_path}: "
            f"missing={list(missing)}, unexpected={list(disallowed)}"
        )
    return unexpected
