"""Dataclasses for data, model, loss, training, and evaluation settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DataConfig:
    dataset_dir: str = "sswpt_dataset"
    stats_path: str = "pretrain/train_stats.json"
    pipeline_version: str = "single_normalization_v2"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42
    fixed_length_mode: bool = False
    fixed_t0: int = 128
    bucket_boundaries: List[int] = field(
        default_factory=lambda: [9000, 10000, 11000, 12500, 14500, 17500, 22000, 28500, 33500, 40345]
    )
    bucket_batch_size: int = 32
    pad_to_multiple_of: int = 8
    max_time_tokens: Optional[int] = None
    crop_quantile: float = 0.98
    num_workers: int = 8
    prefetch_factor: int = 1


@dataclass
class AugmentationConfig:
    enable: bool = True
    joint_crop_min_ratio: float = 0.6
    joint_crop_max_ratio: float = 1.0
    tf_time_mask_prob: float = 0.3
    tf_time_mask_width: int = 8
    tf_freq_mask_prob: float = 0.3
    tf_freq_mask_width: int = 8
    tf_log_noise_std: float = 0.05
    pa_gain_db: float = 2.0
    pa_noise_std_ratio: float = 0.01
    pa_dropout_prob: float = 0.2
    pa_dropout_width: int = 128
    pa_shift_max: int = 64
    vel_noise_std: float = 0.05
    vel_mask_prob: float = 0.15
    vel_mask_width: int = 8


@dataclass
class ModelConfig:
    embed_dim: int = 384
    tf_patch_freq: int = 8
    tf_patch_time: int = 800
    tf_depth: int = 10
    wave_depth: int = 6
    vel_depth: int = 3
    fusion_depth: int = 4
    predictor_depth: int = 4
    predictor_use_position: bool = True
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attn_dropout: float = 0.1
    vel_hidden_dim: int = 128
    vel_num_heads: int = 4
    meta_hidden_dim: int = 128
    wave_stem_strides: List[int] = field(default_factory=lambda: [4, 4, 4, 4])
    vel_stem_strides: List[int] = field(default_factory=lambda: [4, 4, 4, 4])
    max_fourier_bands: int = 16
    sga_num_freq_groups: int = 16
    sga_film: bool = True
    sga_attn_bias: bool = True
    sga_freq_gate: bool = True
    modality_dropout_prob: float = 0.15
    use_grad_checkpoint: bool = True
    jepa_num_tube_masks: int = 4
    jepa_tube_ratio_min: float = 0.15
    jepa_tube_ratio_max: float = 0.20
    use_vicreg: bool = False


@dataclass
class LossConfig:
    lambda_jepa: float = 1.0
    lambda_contrastive: float = 0.3
    lambda_aux: float = 0.05
    lambda_vicreg: float = 0.1
    contrastive_temperature: float = 0.07
    vicreg_var_weight: float = 25.0
    vicreg_cov_weight: float = 10.0
    vicreg_sim_weight: float = 0.0
    use_uncertainty_weighting: bool = True
    uncertainty_exclude_contrastive: bool = True


@dataclass
class TrainConfig:
    output_dir: str = "pretrain/runs"
    device: str = "cuda"
    epochs: int = 100
    lr: float = 2e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    amp: bool = True
    grad_clip_norm: float = 1.0
    ema_start: float = 0.996
    ema_end: float = 0.9999
    log_every: int = 20
    save_every: int = 1


@dataclass
class EvalConfig:
    batch_size: int = 16
    probe_epochs: int = 50
    probe_lr: float = 1e-3
    hidden_dim: int = 256


@dataclass
class ExperimentConfig:
    name: str = "tn-mjepa-base"
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_preset(name: str) -> ExperimentConfig:
    cfg = ExperimentConfig(name=name)
    model = cfg.model
    if name.endswith("-small"):
        model.embed_dim = 256
        model.num_heads = 4
        model.tf_depth = 8
        model.wave_depth = 4
        model.fusion_depth = 3
        model.predictor_depth = 3
    elif name.endswith("-large"):
        model.embed_dim = 512
        model.num_heads = 8
        model.tf_depth = 12
        model.wave_depth = 8
        model.vel_depth = 4
        model.fusion_depth = 6
        model.predictor_depth = 6
    return cfg


def resolve_path(path_str: str, workspace_root: Optional[Path] = None) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    if workspace_root is None:
        workspace_root = Path.cwd()
    return workspace_root / path
