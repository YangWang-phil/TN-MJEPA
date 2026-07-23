"""Train TN-MJEPA with variable-length multimodal batches."""

from __future__ import annotations

import argparse
import csv
import math
from functools import partial
from pathlib import Path
from typing import Dict

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pretrain.configs import ExperimentConfig, get_preset, resolve_path
from pretrain.data.dataset import BucketBatchSampler, TrainNoiseDataset, collate_samples
from pretrain.data.normalization import TrainStats
from pretrain.models.diagnostics import (
    alignment_uniformity,
    contrastive_topk_accuracy,
    jepa_prediction_quality,
    representation_health,
)
from pretrain.models.ema import update_ema_model
from pretrain.models.tn_mjepa import TNMJEPA
from pretrain.utils import cosine_ema, save_json, set_seed


def build_loader(cfg: ExperimentConfig, stats: TrainStats, split: str) -> DataLoader:
    dataset = TrainNoiseDataset(cfg.data, split=split, stats=stats, aug_cfg=cfg.augmentation)
    num_workers = cfg.data.num_workers
    collate_fn = partial(collate_samples, pad_to_multiple_of=cfg.data.pad_to_multiple_of)
    extra_kwargs: Dict[str, object] = {}
    if num_workers > 0:
        extra_kwargs["persistent_workers"] = True
        extra_kwargs["prefetch_factor"] = cfg.data.prefetch_factor
    if split == "train" and not cfg.data.fixed_length_mode:
        batch_sampler = BucketBatchSampler(dataset.records, cfg.data.bucket_boundaries, cfg.data.bucket_batch_size)
        return DataLoader(
            dataset, batch_sampler=batch_sampler, collate_fn=collate_fn,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(), **extra_kwargs,
        )
    batch_size = cfg.data.bucket_batch_size if split == "train" else max(1, cfg.eval.batch_size)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=(split == "train"), collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), **extra_kwargs,
    )


def build_lr_scheduler(optimizer: AdamW, cfg: ExperimentConfig, steps_per_epoch: int) -> LambdaLR:
    """Linear warmup followed by cosine decay."""
    total_steps = max(cfg.train.epochs * steps_per_epoch, 1)
    warmup_steps = max(cfg.train.warmup_epochs * steps_per_epoch, 1)
    floor = cfg.train.min_lr / cfg.train.lr

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, lr_lambda)


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


LOSS_KEYS = ["loss", "loss_jepa", "loss_jepa_tf", "loss_jepa_wave", "loss_jepa_vel", "loss_contrastive", "loss_aux", "loss_vicreg"]


def stack_loss_values(output: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([output[key] for key in LOSS_KEYS])


def finalize_loss_meter(meter: torch.Tensor, steps: int) -> Dict[str, float]:
    values = (meter / max(steps, 1)).detach().cpu().tolist()
    return {key: float(value) for key, value in zip(LOSS_KEYS, values)}


def optimizer_step_succeeded(scale_before: float, scale_after: float) -> bool:
    """GradScaler reduces its scale only when it skipped an overflowing step."""
    return scale_after >= scale_before


def train_one_epoch(
    model: TNMJEPA,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: GradScaler,
    device: torch.device,
    cfg: ExperimentConfig,
    epoch: int,
    total_epochs: int,
    writer: SummaryWriter,
) -> Dict[str, float]:
    model.train()
    meter = torch.zeros(len(LOSS_KEYS), device=device)
    steps = 0
    progress = tqdm(loader, desc=f"Train {epoch+1}/{total_epochs}", leave=False)
    for step, batch in enumerate(progress):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device.type, enabled=cfg.train.amp and device.type == "cuda"):
            output = model(batch)
            loss = output["loss"]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        stepped = optimizer_step_succeeded(scale_before, scaler.get_scale())
        if stepped:
            scheduler.step()

        global_step = epoch * max(len(loader), 1) + step
        momentum = cosine_ema(cfg.train.ema_start, cfg.train.ema_end, global_step, total_epochs * max(len(loader), 1))
        if stepped:
            update_ema_model(model.tf_encoder, model.target_tf_encoder, momentum)
            update_ema_model(model.wave_encoder, model.target_wave_encoder, momentum)
            update_ema_model(model.vel_encoder, model.target_vel_encoder, momentum)

        loss_values = stack_loss_values(output)
        meter += loss_values.detach()
        steps += 1

        if global_step % cfg.train.log_every == 0:
            current_metrics = finalize_loss_meter(loss_values, steps=1)
            progress.set_postfix(loss=f"{current_metrics['loss']:.4f}")
            for key, value in current_metrics.items():
                writer.add_scalar(f"train/{key}", value, global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
            writer.add_scalar("train/ema_momentum", momentum, global_step)
            for name, weight in output["task_weights"].items():
                writer.add_scalar(f"train/task_weight_{name}", float(weight.item()), global_step)
    return finalize_loss_meter(meter, steps)


@torch.no_grad()
def evaluate(model: TNMJEPA, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Compute validation losses and label-free representation diagnostics."""
    model.eval()
    meter = torch.zeros(len(LOSS_KEYS), device=device)
    diag = {
        "top1": 0.0, "top5": 0.0, "jepa_cos_tf": 0.0, "jepa_cos_wave": 0.0, "jepa_cos_vel": 0.0,
        "alignment": 0.0, "uniformity": 0.0,
    }
    diag_counts = {key: 0 for key in diag}
    pooled_all = []
    steps = 0
    for batch in tqdm(loader, desc="Validate", leave=False):
        batch = move_batch_to_device(batch, device)
        output = model(batch, return_extras=True)
        meter += stack_loss_values(output).detach()
        steps += 1

        topk = contrastive_topk_accuracy(
            output["contrastive_extras"]["embeddings"],
            output["contrastive_extras"]["positive_sets"],
            model.cfg.loss.contrastive_temperature,
        )
        for short, full in [("top1", "top1"), ("top5", "top5")]:
            if topk[full] == topk[full]:
                diag[short] += topk[full]
                diag_counts[short] += 1
        for name in ["tf", "wave", "vel"]:
            extra = output["jepa_extras"][name]
            q = jepa_prediction_quality(extra["pred"], extra["target"], extra["mask"])
            if q == q:
                diag[f"jepa_cos_{name}"] += q
                diag_counts[f"jepa_cos_{name}"] += 1
        au = alignment_uniformity(output["pooled_online"], output["pooled_online_v2"])
        for key in ["alignment", "uniformity"]:
            diag[key] += au[key]
            diag_counts[key] += 1
        pooled_all.append(output["pooled_online"].detach().cpu())

    metrics = finalize_loss_meter(meter, steps)
    for key in diag:
        metrics[key] = diag[key] / max(diag_counts[key], 1)
    health = representation_health(torch.cat(pooled_all, dim=0))
    metrics["repr_std_mean"] = health["std_mean"]
    metrics["repr_std_min"] = health["std_min"]
    metrics["repr_effective_rank"] = health["effective_rank"]
    return metrics


def append_metrics_csv(path: Path, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float], lr: float) -> None:
    row = {"epoch": epoch + 1, "lr": lr}
    row.update({f"train_{k}": v for k, v in train_metrics.items()})
    row.update({f"val_{k}": v for k, v in val_metrics.items()})
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def apply_section5_variant(cfg: ExperimentConfig, variant: str) -> None:
    """Apply a paper ablation variant."""
    cfg.name = f"{cfg.name}-{variant}"
    if variant == "full":
        return
    if variant == "no_aux":
        cfg.loss.lambda_aux = 0.0
        return
    if variant == "no_sga":
        cfg.model.sga_film = False
        cfg.model.sga_attn_bias = False
        cfg.model.sga_freq_gate = False
        return
    if variant == "no_position":
        cfg.model.predictor_use_position = False
        return
    if variant == "jepa_only":
        cfg.loss.lambda_contrastive = 0.0
        cfg.loss.lambda_vicreg = 0.0
        return
    if variant == "contrastive_only":
        cfg.loss.lambda_jepa = 0.0
        cfg.loss.use_uncertainty_weighting = False
        return
    raise ValueError(f"Unknown training variant: {variant}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TN-MJEPA.")
    parser.add_argument("--preset", default="tn-mjepa-base", help="Model preset: tn-mjepa-small/base/large.")
    parser.add_argument("--dataset-dir", default="sswpt_dataset")
    parser.add_argument("--stats-path", default="pretrain/train_stats.json")
    parser.add_argument("--output-dir", default="pretrain/runs")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument(
        "--variant",
        choices=["full", "no_aux", "no_sga", "no_position", "jepa_only", "contrastive_only"],
        default=None,
    )
    args = parser.parse_args()

    cfg = get_preset(args.preset)
    cfg.data.dataset_dir = args.dataset_dir
    cfg.data.stats_path = args.stats_path
    cfg.train.output_dir = args.output_dir
    if args.variant is not None:
        apply_section5_variant(cfg, args.variant)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.device is not None:
        cfg.train.device = args.device
    if args.seed is not None:
        cfg.data.random_seed = args.seed
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    if args.prefetch_factor is not None:
        cfg.data.prefetch_factor = args.prefetch_factor

    set_seed(cfg.data.random_seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(cfg.train.device if torch.cuda.is_available() or cfg.train.device == "cpu" else "cpu")
    stats = TrainStats.from_json(resolve_path(cfg.data.stats_path))

    model = TNMJEPA(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = GradScaler(device.type, enabled=cfg.train.amp and device.type == "cuda")

    train_loader = build_loader(cfg, stats, split="train")
    val_loader = build_loader(cfg, stats, split="val")
    scheduler = build_lr_scheduler(optimizer, cfg, steps_per_epoch=max(len(train_loader), 1))
    output_dir = resolve_path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=output_dir / "tb")
    save_json(output_dir / "config.json", cfg.to_dict())
    metrics_csv = output_dir / "metrics.csv"

    best_val = float("inf")
    for epoch in range(cfg.train.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, cfg, epoch, cfg.train.epochs, writer)
        val_metrics = evaluate(model, val_loader, device)
        val_metrics["loss_core"] = (
            cfg.loss.lambda_jepa * val_metrics["loss_jepa"]
            + cfg.loss.lambda_contrastive * val_metrics["loss_contrastive"]
            + cfg.loss.lambda_aux * val_metrics["loss_aux"]
        )
        for key, value in val_metrics.items():
            writer.add_scalar(f"val/{key}", value, epoch)
        append_metrics_csv(metrics_csv, epoch, train_metrics, val_metrics, lr=scheduler.get_last_lr()[0])

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": cfg.to_dict(),
        }
        torch.save(ckpt, output_dir / "last.pt")
        if val_metrics["loss_core"] < best_val:
            best_val = val_metrics["loss_core"]
            torch.save(ckpt, output_dir / "best.pt")

        print(
            f"Epoch {epoch+1}: train={train_metrics['loss']:.4f}, val={val_metrics['loss']:.4f}, "
            f"val_top1={val_metrics['top1']:.4f}, val_jepa_cos(tf/wave/vel)="
            f"{val_metrics['jepa_cos_tf']:.3f}/{val_metrics['jepa_cos_wave']:.3f}/{val_metrics['jepa_cos_vel']:.3f}, "
            f"eff_rank={val_metrics['repr_effective_rank']:.1f}"
        )

    writer.close()

    del train_loader, val_loader
    best_checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["model"], strict=True)
    model.to(device).eval()
    test_loader = build_loader(cfg, stats, split="test")
    test_metrics = evaluate(model, test_loader, device)
    save_json(
        output_dir / "test_intrinsic.json",
        {
            "checkpoint_epoch": int(best_checkpoint["epoch"] + 1),
            "num_samples": len(test_loader.dataset),
            "metrics": test_metrics,
        },
    )


if __name__ == "__main__":
    main()
