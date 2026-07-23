import pytest
import torch

from pretrain.checkpoints import load_model_state_compat
from pretrain.configs import get_preset
from pretrain.models.tn_mjepa import TNMJEPA


def tiny_model() -> TNMJEPA:
    cfg = get_preset("tn-mjepa-base")
    cfg.model.embed_dim = 32
    cfg.model.num_heads = 4
    cfg.model.vel_hidden_dim = 16
    cfg.model.vel_num_heads = 4
    cfg.model.meta_hidden_dim = 8
    cfg.model.tf_depth = 1
    cfg.model.wave_depth = 1
    cfg.model.vel_depth = 1
    cfg.model.fusion_depth = 1
    cfg.model.predictor_depth = 1
    cfg.model.sga_num_freq_groups = 4
    return TNMJEPA(cfg)


def test_legacy_unused_target_branches_are_the_only_allowed_extras() -> None:
    model = tiny_model()
    state = dict(model.state_dict())
    state["target_fusion.legacy"] = torch.zeros(1)
    state["target_meta_mlp.legacy"] = torch.zeros(1)
    ignored = load_model_state_compat(model, state)
    assert ignored == ("target_fusion.legacy", "target_meta_mlp.legacy")


def test_unrelated_checkpoint_mismatch_is_rejected() -> None:
    model = tiny_model()
    state = dict(model.state_dict())
    state["unexpected_branch.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="unexpected_branch"):
        load_model_state_compat(model, state)

def test_missing_checkpoint_key_is_rejected() -> None:
    model = tiny_model()
    state = dict(model.state_dict())
    state.pop(next(iter(state)))
    with pytest.raises(RuntimeError, match="missing"):
        load_model_state_compat(model, state)
