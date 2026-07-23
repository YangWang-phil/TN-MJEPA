import numpy as np
import pytest

from evaluation.interventions import aggregate_summaries, intervention_metrics


def test_intervention_identity_and_rank_ratio() -> None:
    clean = np.eye(3, dtype=np.float32)
    summary, rows = intervention_metrics(clean, clean.copy(), np.eye(3, dtype=bool), clean_effective_rank=2.0)
    assert summary["embedding_cosine_mean"] == 1.0
    assert summary["retrieval_at1"] == 1.0
    assert summary["effective_rank_ratio"] == pytest.approx(1.0)
    assert np.all(rows["retrieval_hit"] == 1)


def test_intervention_aggregation_uses_run_variation() -> None:
    rows = [
        {"condition": "clean", "embedding_cosine_mean": 1.0, "embedding_cosine_median": 1.0,
         "embedding_cosine_p10": 1.0, "retrieval_at1": 1.0, "retrieval_margin_mean": 0.5,
         "retrieval_margin_median": 0.5, "effective_rank": 3.0, "effective_rank_ratio": 1.0},
        {"condition": "clean", "embedding_cosine_mean": 1.0, "embedding_cosine_median": 1.0,
         "embedding_cosine_p10": 1.0, "retrieval_at1": 1.0, "retrieval_margin_mean": 0.7,
         "retrieval_margin_median": 0.7, "effective_rank": 4.0, "effective_rank_ratio": 1.0},
    ]
    aggregate = aggregate_summaries(rows)
    margin = [row for row in aggregate if row["condition"] == "clean" and row["metric"] == "retrieval_margin_mean"]
    assert len(margin) == 1
    assert margin[0]["n_runs"] == 2
    assert margin[0]["mean"] == pytest.approx(0.6)
