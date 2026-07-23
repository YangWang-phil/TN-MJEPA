"""Fine-tune TN-MJEPA on externally supplied line-attribute labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from pretrain.configs import resolve_path
from pretrain.data.dataset import SampleRecord, TrainNoiseDataset, collate_samples
from pretrain.data.normalization import TrainStats
from pretrain.checkpoints import load_config_from_checkpoint, load_model_state_compat
from pretrain.models.tn_mjepa import TNMJEPA
from pretrain.train_pretrain import move_batch_to_device
from pretrain.utils import set_seed


TASKS = {
    "track_structure": {"target": "track_class", "classes": 3},
    "curve_geometry": {"target": "curve_geometry", "classes": 3},
    "corrugation": {"target": "corrugation", "classes": 2},
}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def curve_class(row: Mapping[str, str]) -> int:
    if row["curve_radius_m"] == "":
        return 0  # straight
    return 1 if float(row["curve_radius_m"]) < 800.0 else 2


def target_value(row: Mapping[str, str], task: str) -> int:
    if task == "curve_geometry":
        return curve_class(row)
    raw = row[TASKS[task]["target"]]
    return int(float(raw)) if raw not in ("", "None") else -1


def pooled_spatial_split(rows: Sequence[Mapping[str, str]]) -> tuple[list[int], list[int], list[int], dict[str, Any]]:
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    dropped_boundary = 0
    for index, row in enumerate(rows):
        start = float(row["mileage_start_km"])
        end = float(row["mileage_end_km"])
        block = math.floor(start / 0.2)
        # Drop windows crossing a 200-m block boundary.  This ensures that
        # samples assigned to different splits cannot share raw observations.
        if end > (block + 1) * 0.2 + 1e-7:
            dropped_boundary += 1
            continue
        residue = block % 10
        (val if residue == 0 else test if residue == 1 else train).append(index)
    audit: dict[str, Any] = {
        "split_rule": "200-m spatial blocks; block residue 0=validation, 1=test, all others=train",
        "dropped_boundary_windows": dropped_boundary,
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "task_counts": {},
    }
    for split_name, indices in (("train", train), ("val", val), ("test", test)):
        for task in TASKS:
            labels = [target_value(rows[i], task) for i in indices]
            labels = [label for label in labels if label >= 0]
            audit["task_counts"].setdefault(task, {})[split_name] = dict(Counter(labels))
    return train, val, test, audit


def random_sample_split(
    rows: Sequence[Mapping[str, str]], seed: int
) -> tuple[list[int], list[int], list[int], dict[str, Any]]:
    indices = np.random.default_rng(seed).permutation(len(rows)).tolist()
    train_end = int(len(indices) * 0.8)
    val_end = train_end + int(len(indices) * 0.1)
    train, val, test = indices[:train_end], indices[train_end:val_end], indices[val_end:]
    audit: dict[str, Any] = {
        "split_rule": f"sample-level random 80/10/10 split; seed={seed}",
        "dropped_boundary_windows": 0,
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "task_counts": {},
    }
    for split_name, split_indices in (("train", train), ("val", val), ("test", test)):
        for task in TASKS:
            labels = [target_value(rows[i], task) for i in split_indices]
            labels = [label for label in labels if label >= 0]
            audit["task_counts"].setdefault(task, {})[split_name] = dict(Counter(labels))
    return train, val, test, audit


def collate_labeled(batch: Sequence[dict[str, Any]], task_names: Sequence[str] | None = None) -> dict[str, Any]:
    output = collate_samples(batch)
    task_names = tuple(task_names) if task_names is not None else tuple(TASKS)
    labels = []
    for item in batch:
        row = item["label_row"]
        labels.append([target_value(row, task) for task in task_names])
    output["downstream_targets"] = torch.tensor(labels, dtype=torch.long)
    return output


class LabelDataset(TrainNoiseDataset):
    def __init__(self, *args: Any, label_rows: Mapping[str, Mapping[str, str]], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.label_rows = label_rows

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        sample["label_row"] = self.label_rows[sample["record"].stem]
        return sample


def make_records(rows: Sequence[Mapping[str, str]], dataset_dir: Path) -> list[SampleRecord]:
    return [
        SampleRecord(
            path=dataset_dir / row["file_name"],
            stem=row["sample_id"],
            source_tag=row["sample_id"],
            source_file=row["source_file"],
            slice_type=row["slice_type"],
            mileage_start_km=float(row["mileage_start_km"]),
            mileage_end_km=float(row["mileage_end_km"]),
            duration_s=0.0,
            t_length=0,
            f_length=0,
        )
        for row in rows
    ]


def make_loader(dataset: LabelDataset, indices: Sequence[int], batch_size: int, workers: int, shuffle: bool = False) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": partial(collate_labeled, task_names=tuple(TASKS)),
    }
    if workers > 0:
        kwargs.update(num_workers=workers, persistent_workers=True, prefetch_factor=1)
    return DataLoader(Subset(dataset, list(indices)), **kwargs)


class FineTuneModel(nn.Module):
    def __init__(self, encoder: TNMJEPA, representation: str = "fusion") -> None:
        super().__init__()
        self.encoder = encoder
        self.representation = representation
        dim = encoder.cfg.model.embed_dim
        self.heads = nn.ModuleDict({task: nn.Linear(dim, spec["classes"]) for task, spec in TASKS.items()})

    def forward(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        if self.representation == "fusion":
            encoded = self.encoder._encode_online(
                tf_log=batch["tf_log_view1"], pa=batch["pa_view1"], vel=batch["vel_view1"],
                time_axis=batch["time_view1"], time_mask=batch["time_mask_view1"],
                freq_axis=batch["freq_axis"], meta=batch["meta"],
            )
            pooled = encoded["fusion"]["pooled"]
        else:
            vel_out = self.encoder.vel_encoder(batch["vel_view1"], batch["time_view1"], batch["time_mask_view1"])
            if self.representation == "vel":
                pooled = vel_out["pooled"]
            elif self.representation == "tf":
                pooled = self.encoder.tf_encoder(
                    batch["tf_log_view1"], batch["freq_axis"], batch["time_view1"], batch["time_mask_view1"],
                    speed_hidden=vel_out["hidden"], speed_valid_mask=vel_out["mask"],
                )["pooled"]
            else:
                pooled = self.encoder.wave_encoder(
                    batch["pa_view1"], batch["time_view1"], batch["time_mask_view1"],
                    speed_hidden=vel_out["hidden"], speed_valid_mask=vel_out["mask"],
                )["pooled"]
        return {task: head(pooled) for task, head in self.heads.items()}


def metrics_from_logits(logits: np.ndarray, labels: np.ndarray, num_classes: int) -> dict[str, float]:
    mask = labels >= 0
    y = labels[mask]
    pred = logits[mask].argmax(axis=1)
    recalls: list[float] = []
    f1s: list[float] = []
    for cls in range(num_classes):
        tp = int(np.sum((y == cls) & (pred == cls)))
        fn = int(np.sum((y == cls) & (pred != cls)))
        fp = int(np.sum((y != cls) & (pred == cls)))
        recalls.append(tp / max(tp + fn, 1))
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return {"balanced_accuracy": float(np.mean(recalls)), "macro_f1": float(np.mean(f1s)), "n": int(y.size)}


@torch.no_grad()
def evaluate(model: FineTuneModel, loader: DataLoader, device: torch.device) -> dict[str, dict[str, float]]:
    model.eval()
    logits = {task: [] for task in TASKS}
    targets: list[np.ndarray] = []
    for batch in loader:
        moved = move_batch_to_device(batch, device)
        output = model(moved)
        for task in TASKS:
            logits[task].append(output[task].float().cpu().numpy())
        targets.append(batch["downstream_targets"].numpy())
    y = np.concatenate(targets, axis=0)
    return {task: metrics_from_logits(np.concatenate(logits[task]), y[:, index], TASKS[task]["classes"]) for index, task in enumerate(TASKS)}


def train_one_epoch(
    model: FineTuneModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    class_weights: Mapping[str, torch.Tensor],
) -> float:
    model.train()
    total = 0.0
    steps = 0
    for batch in loader:
        moved = move_batch_to_device(batch, device)
        targets = moved.pop("downstream_targets")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = model(moved)
            losses = []
            for index, (task, spec) in enumerate(TASKS.items()):
                mask = targets[:, index] >= 0
                if bool(mask.any()):
                    losses.append(nn.functional.cross_entropy(output[task][mask], targets[mask, index], weight=class_weights[task]))
            loss = sum(losses) / max(len(losses), 1)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach().cpu())
        steps += 1
    return total / max(steps, 1)


def main() -> None:
    global TASKS
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="pretrain/downstream_line/line_label_manifest.csv")
    parser.add_argument("--dataset-dir", default="sswpt_dataset")
    parser.add_argument("--checkpoint", default="runs/tn-mjepa-base/best.pt")
    parser.add_argument("--stats-path", default="pretrain/train_stats.json")
    parser.add_argument("--out-dir", default="evaluation/line_finetune")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--init", choices=("full", "random"), default="full")
    parser.add_argument("--task", choices=("all", "track_structure", "curve_geometry", "corrugation"), default="all")
    parser.add_argument("--representation", choices=("tf", "wave", "vel", "fusion"), default="fusion")
    parser.add_argument("--split-strategy", choices=("spatial", "random"), default="spatial")
    parser.add_argument("--split-seed", type=int, default=20260722)
    parser.add_argument("--split-before-task-filter", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.task != "all":
        TASKS = {args.task: TASKS[args.task]}
    set_seed(20260719)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(Path(args.manifest))
    preserve_full_split = args.task != "all" and args.split_before_task_filter
    if args.task != "all" and not preserve_full_split:
        rows = [row for row in rows if target_value(row, args.task) >= 0]
    if args.split_strategy == "random":
        train_idx, val_idx, test_idx, audit = random_sample_split(rows, args.split_seed)
    else:
        train_idx, val_idx, test_idx, audit = pooled_spatial_split(rows)
    if preserve_full_split:
        audit["full_pool_counts"] = dict(audit["counts"])
        labeled_before_boundary = sum(target_value(row, args.task) >= 0 for row in rows)
        train_idx = [index for index in train_idx if target_value(rows[index], args.task) >= 0]
        val_idx = [index for index in val_idx if target_value(rows[index], args.task) >= 0]
        test_idx = [index for index in test_idx if target_value(rows[index], args.task) >= 0]
        audit["counts"] = {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)}
        audit["task_labeled_records_before_boundary_drop"] = labeled_before_boundary
        audit["task_boundary_dropped_windows"] = labeled_before_boundary - sum(audit["counts"].values())
    (out_dir / "split_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    records = make_records(rows, Path(args.dataset_dir))
    row_by_id = {row["sample_id"]: row for row in rows}
    checkpoint_path = resolve_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = load_config_from_checkpoint(checkpoint, "tn-mjepa-base", args.dataset_dir, args.stats_path)
    cfg.model.use_grad_checkpoint = True
    stats = TrainStats.from_json(resolve_path(args.stats_path))
    dataset = LabelDataset(
        cfg.data,
        split="probe",
        stats=stats,
        aug_cfg=None,
        records=records,
        label_rows=row_by_id,
    )
    train_loader = make_loader(dataset, train_idx, args.batch_size, args.workers, shuffle=True)
    val_loader = make_loader(dataset, val_idx, args.batch_size, args.workers)
    test_loader = make_loader(dataset, test_idx, args.batch_size, args.workers)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = TNMJEPA(cfg)
    if args.init == "full":
        load_model_state_compat(model, checkpoint["model"], checkpoint_path=checkpoint_path)
    model = FineTuneModel(model, representation=args.representation).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    class_weights: dict[str, torch.Tensor] = {}
    for task, spec in TASKS.items():
        labels = [target_value(rows[index], task) for index in train_idx]
        labels = [label for label in labels if label >= 0]
        counts = np.bincount(labels, minlength=spec["classes"]).astype(np.float32)
        weight = counts.sum() / np.maximum(counts, 1.0)
        class_weights[task] = torch.from_numpy((weight / weight.mean()).astype(np.float32)).to(device)
    task_suffix = args.task if args.task != "all" else "all"
    rep_suffix = args.representation
    state_path = out_dir / f"finetune_{args.init}_{task_suffix}_{rep_suffix}_training_state.pt"
    best_score = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    history = []
    start_epoch = 1
    if args.resume:
        if not state_path.exists():
            raise FileNotFoundError(f"resume state not found: {state_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        expected = {"task": args.task, "representation": args.representation, "split_strategy": args.split_strategy}
        if state["run_signature"] != expected:
            raise ValueError(f"resume state mismatch: {state['run_signature']} != {expected}")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        best_score = float(state["best_score"])
        best_state = state["best_state"]
        best_epoch = int(state["best_epoch"])
        history = state["history"]
        start_epoch = int(state["completed_epoch"]) + 1
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        if device.type == "cuda" and state["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    for epoch in range(start_epoch, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device, scaler, class_weights)
        val = evaluate(model, val_loader, device)
        score = float(np.mean([value["macro_f1"] for value in val.values() if value["n"] > 0]))
        history.append({"epoch": epoch, "train_loss": loss, "val": val})
        print(json.dumps({"epoch": epoch, "train_loss": loss, "val_macro_f1": score}, ensure_ascii=False))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    completed_epoch = int(history[-1]["epoch"]) if history else 0
    torch.save(
        {
            "run_signature": {"task": args.task, "representation": args.representation, "split_strategy": args.split_strategy},
            "completed_epoch": completed_epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_score": best_score,
            "best_state": best_state,
            "best_epoch": best_epoch,
            "history": history,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        },
        state_path,
    )
    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(model, test_loader, device)
    report = {
        "init": args.init,
        "checkpoint": args.checkpoint,
        "split_strategy": args.split_strategy,
        "split_seed": args.split_seed,
        "split_before_task_filter": args.split_before_task_filter,
        "completed_epochs": completed_epoch,
        "best_epoch": best_epoch,
        "split_audit": audit,
        "history": history,
        "test": test,
    }
    report["representation"] = args.representation
    (out_dir / f"finetune_{args.init}_{task_suffix}_{rep_suffix}_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out_dir / f"finetune_{args.init}_{task_suffix}_{rep_suffix}_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "balanced_accuracy", "macro_f1", "n"])
        for task, metrics in test.items():
            writer.writerow([task, metrics["balanced_accuracy"], metrics["macro_f1"], metrics["n"]])
    print(json.dumps({"init": args.init, "best_epoch": best_epoch, "test": test}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
