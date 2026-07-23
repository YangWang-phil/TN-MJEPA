import numpy as np
import torch
from dataclasses import dataclass

from evaluation.paper_intrinsic import (
    applied_gate_indices,
    attention_log2_enrichment,
    build_cross_view_positive_mask,
    covariance_spectrum,
    global_cross_view_retrieval,
    linear_cka,
    samplewise_masked_cosine,
)


@dataclass
class Record:
    source_file: str
    slice_type: str
    mileage_start_km: float
    mileage_end_km: float


def test_global_retrieval_uses_the_complete_similarity_matrix() -> None:
    features = np.eye(4, dtype=np.float32)
    positives = np.eye(4, dtype=bool)
    result = global_cross_view_retrieval(features, features, positives)
    assert result["retrieval_at1"] == 1.0
    assert np.all(result["margin"] > 0)


def test_overlap_positive_can_differ_from_the_paired_index() -> None:
    view1 = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    view2 = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    positives = np.ones((2, 2), dtype=bool)
    result = global_cross_view_retrieval(view1, view2, positives)
    assert result["retrieval_at1"] == 1.0


def test_covariance_and_cka_are_well_formed() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(30, 5))
    spectrum, effective_rank = covariance_spectrum(features)
    assert np.isclose(spectrum.sum(), 1.0)
    assert 1.0 <= effective_rank <= 5.0
    assert np.isclose(linear_cka(features, features), 1.0)


def test_positive_mask_includes_mileage_overlap() -> None:
    records = [
        Record("source", "mileage", 0.0, 1.0),
        Record("source", "mileage", 0.5, 1.5),
        Record("source", "mileage", 2.0, 3.0),
        Record("source", "stationary", float("nan"), float("nan")),
    ]
    mask = build_cross_view_positive_mask(records)
    assert np.all(np.diag(mask))
    assert mask[0, 1] and mask[1, 0]
    assert not mask[0, 2]


def test_samplewise_masked_cosine_handles_empty_mask() -> None:
    pred = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])
    mask = torch.tensor([[True, False], [False, False]])
    result = samplewise_masked_cosine(pred, pred.clone(), mask)
    assert result[0] == 1.0
    assert np.isnan(result[1])


def test_gate_mapping_and_token_corrected_enrichment() -> None:
    np.testing.assert_array_equal(
        applied_gate_indices(8, 16), np.array([0, 2, 4, 6, 8, 10, 12, 15])
    )
    assert abs(attention_log2_enrichment(0.25, 2, 8)) < 1e-12
