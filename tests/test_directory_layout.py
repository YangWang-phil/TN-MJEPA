from pathlib import Path


def test_pretrain_package_contains_no_evaluation_or_finetuning_entry_points() -> None:
    root = Path(__file__).resolve().parents[1]
    names = {path.name for path in (root / "pretrain").rglob("*.py")}
    assert "eval_intrinsic.py" not in names
    assert "eval_probe.py" not in names
    assert "downstream_finetune.py" not in names
    assert (root / "evaluation" / "paper_intrinsic.py").exists()
    assert (root / "evaluation" / "weak_label_probes.py").exists()
    assert (root / "downstream" / "finetune.py").exists()
