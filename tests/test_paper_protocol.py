import json
from pathlib import Path

import pytest

from scripts.run_paper_pretraining import validate_stats


ROOT = Path(__file__).resolve().parents[1]


def load_protocol() -> dict:
    return json.loads((ROOT / "configs" / "paper_protocol.json").read_text(encoding="utf-8"))


def compatible_stats(protocol: dict) -> dict:
    source_split = protocol["source_split"]
    expected = protocol["expected_samples"]
    return {
        "split_files": {split: [None] * count for split, count in expected.items()},
        "split_sources": {
            "train": list(source_split["train"]),
            "val": [source_split["val"]],
            "test": [source_split["test"]],
        },
    }


def test_paper_protocol_freezes_epochs_seeds_and_counts() -> None:
    protocol = load_protocol()
    assert protocol["epochs"] == 60
    assert protocol["seeds"] == [42, 43, 44]
    assert protocol["expected_samples"] == {"train": 16038, "val": 2586, "test": 2514}
    validate_stats(compatible_stats(protocol), protocol)


def test_protocol_validation_rejects_count_drift() -> None:
    protocol = load_protocol()
    stats = compatible_stats(protocol)
    stats["split_files"]["test"].pop()
    with pytest.raises(ValueError, match="test count mismatch"):
        validate_stats(stats, protocol)
