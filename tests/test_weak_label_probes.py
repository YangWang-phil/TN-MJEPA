import numpy as np
import pytest

from evaluation.weak_label_probes import (
    aggregate_results,
    classification_metrics,
    reference_baselines,
    regression_metrics,
)


def test_probe_metrics_for_perfect_predictions() -> None:
    target = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    assert regression_metrics(target, target.copy()) == {"mae": 0.0, "rmse": 0.0, "r2": 1.0}
    labels = np.array([0, 1, 0, 1], dtype=np.int64)
    probability = np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float32)
    assert classification_metrics(labels, probability) == {"accuracy": 1.0, "macro_f1": 1.0, "auc": 1.0}


def test_probe_aggregation_uses_sample_standard_deviation() -> None:
    results = [
        {"representation": "tn_mjepa", "task": "mean_vel", "test_metrics": {"rmse": 1.0}},
        {"representation": "tn_mjepa", "task": "mean_vel", "test_metrics": {"rmse": 3.0}},
        {"representation": "traditional", "task": "mean_vel", "test_metrics": {"rmse": 2.0}},
    ]
    by_representation = {row["representation"]: row for row in aggregate_results(results)}
    assert by_representation["tn_mjepa"]["sample_std"] == pytest.approx(np.sqrt(2.0))
    assert by_representation["traditional"]["sample_std"] is None


def test_reference_baselines_fit_training_partition_only() -> None:
    features = {
        "train": {"mean_vel": np.array([1.0, 3.0]), "rms_pa_db": np.array([10.0, 14.0]), "sliceType": np.array([1, 1])},
        "test": {"mean_vel": np.array([2.0, 4.0]), "rms_pa_db": np.array([12.0, 16.0]), "sliceType": np.array([0, 1])},
    }
    baselines = reference_baselines(features)
    assert baselines["mean_vel"]["train_mean"] == 2.0
    assert baselines["rms_pa_db"]["train_mean"] == 12.0
    assert baselines["sliceType"]["train_majority_class"] == "mileage"
