"""Run the frozen 60-epoch, three-seed full-model training protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_stats(stats: dict, protocol: dict) -> None:
    expected = protocol["expected_samples"]
    for split in ("train", "val", "test"):
        actual = len(stats.get("split_files", {}).get(split, []))
        if actual != int(expected[split]):
            raise ValueError(f"{split} count mismatch: expected {expected[split]}, got {actual}")

    source_split = protocol["source_split"]
    expected_sources = {
        "train": list(source_split["train"]),
        "val": [source_split["val"]],
        "test": [source_split["test"]],
    }
    declared = stats.get("split_sources")
    if not declared:
        # Compatibility for the original paper manifest, which predates the
        # explicit split_sources audit but retains every frozen filename.
        all_sources = sum(expected_sources.values(), [])
        declared = {}
        for split, names in stats.get("split_files", {}).items():
            matched = {
                source for name in names for source in all_sources
                if str(name).startswith(f"{source}_")
            }
            if len(matched) != len(expected_sources[split]):
                raise ValueError(
                    "Cannot verify source assignment from legacy split filenames; "
                    "recompute train_stats.json with --protocol."
                )
            declared[split] = sorted(matched)
    if {key: sorted(value) for key, value in declared.items()} != {
        key: sorted(value) for key, value in expected_sources.items()
    }:
        raise ValueError("Source assignment in train_stats.json does not match the paper protocol")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--stats-path", required=True)
    parser.add_argument("--output-root", default="runs/paper")
    parser.add_argument("--protocol", default="configs/paper_protocol.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument(
        "--include-ablations",
        action="store_true",
        help="Also train the five 60-epoch seed-42 ablations used in the paper.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol_path = Path(args.protocol)
    stats_path = Path(args.stats_path)
    protocol = load_json(protocol_path)
    validate_stats(load_json(stats_path), protocol)

    run_specs = [
        (f"full_seed{seed}", "full", int(seed)) for seed in protocol["seeds"]
    ]
    if args.include_ablations:
        run_specs.extend(
            (run_name, variant, 42)
            for run_name, variant in protocol["single_seed_ablations"].items()
        )

    for run_name, variant, seed in run_specs:
        command = [
            sys.executable,
            "-m",
            "pretrain.train_pretrain",
            "--preset",
            "tn-mjepa-base",
            "--dataset-dir",
            args.dataset_dir,
            "--stats-path",
            args.stats_path,
            "--output-dir",
            str(Path(args.output_root) / run_name),
            "--variant",
            variant,
            "--epochs",
            str(protocol["epochs"]),
            "--seed",
            str(seed),
            "--device",
            args.device,
            "--num-workers",
            str(args.num_workers),
            "--prefetch-factor",
            str(args.prefetch_factor),
        ]
        print(subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
