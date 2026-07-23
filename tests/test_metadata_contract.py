import torch

from pretrain.configs import get_preset
from pretrain.metadata import AUXILIARY_TARGET_FIELDS, METADATA_TENSOR_FIELDS, MODEL_METADATA_FIELDS
from pretrain.models.encoders import MetaMLP


def test_only_duration_and_slice_type_enter_metadata_encoder() -> None:
    cfg = get_preset("tn-mjepa-base")
    cfg.model.embed_dim = 16
    cfg.model.meta_hidden_dim = 8
    encoder = MetaMLP(cfg.model).eval()
    first = torch.tensor([[0.0, 0.0, 1.5, 1.0]])
    changed_targets = torch.tensor([[100.0, -50.0, 1.5, 1.0]])
    assert torch.equal(encoder(first), encoder(changed_targets))


def test_metadata_field_contract_is_explicit() -> None:
    assert AUXILIARY_TARGET_FIELDS == ("rms_pa_db", "mean_vel")
    assert MODEL_METADATA_FIELDS == ("duration_s", "slice_type")
    assert METADATA_TENSOR_FIELDS == AUXILIARY_TARGET_FIELDS + MODEL_METADATA_FIELDS
