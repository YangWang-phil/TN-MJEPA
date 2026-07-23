"""TN-MJEPA multimodal encoder, objectives, and EMA target path."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from pretrain.configs import ExperimentConfig
from pretrain.models.ema import create_ema_model
from pretrain.models.encoders import MetaMLP, TFEncoder, VelEncoder, WaveEncoder
from pretrain.metadata import AUXILIARY_TARGET_SLICE
from pretrain.models.fusion import MultimodalFusionTransformer
from pretrain.models.losses import aux_regression_loss, build_positive_sets, masked_smooth_l1, multi_positive_info_nce, vicreg_loss
from pretrain.models.masking import build_masks
from pretrain.models.predictor import JEPAPredictor
from pretrain.models.sga import SpeedGuidedAttention


def active_objective_paths(cfg: ExperimentConfig, return_extras: bool) -> tuple[bool, bool]:
    """Return whether JEPA and paired-view paths are needed for this forward pass."""
    need_jepa = return_extras or cfg.loss.lambda_jepa > 0
    need_pair = return_extras or cfg.loss.lambda_contrastive > 0 or cfg.loss.lambda_vicreg > 0 or cfg.model.use_vicreg
    return need_jepa, need_pair


class TNMJEPA(nn.Module):
    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        sga = SpeedGuidedAttention(
            vel_dim=cfg.model.vel_hidden_dim,
            feat_dim=cfg.model.embed_dim,
            num_heads=cfg.model.num_heads,
            num_freq_groups=cfg.model.sga_num_freq_groups,
            use_film=cfg.model.sga_film,
            use_attn_bias=cfg.model.sga_attn_bias,
            use_freq_gate=cfg.model.sga_freq_gate,
        )

        self.tf_encoder = TFEncoder(cfg.model, sga=sga)
        self.wave_encoder = WaveEncoder(cfg.model, sga=sga)
        self.vel_encoder = VelEncoder(cfg.model)
        self.meta_mlp = MetaMLP(cfg.model)
        self.fusion = MultimodalFusionTransformer(cfg.model)
        self.predictor = JEPAPredictor(cfg.model)
        self.proj_head = nn.Sequential(
            nn.Linear(cfg.model.embed_dim, cfg.model.embed_dim),
            nn.GELU(),
            nn.Linear(cfg.model.embed_dim, cfg.model.embed_dim),
        )
        self.aux_head = nn.Linear(cfg.model.embed_dim, 2)

        self.task_names = []
        if cfg.loss.lambda_jepa > 0:
            self.task_names.extend(["jepa_tf", "jepa_wave", "jepa_vel"])
        if cfg.loss.lambda_aux > 0:
            self.task_names.append("aux")
        if not cfg.loss.uncertainty_exclude_contrastive and cfg.loss.lambda_contrastive > 0:
            self.task_names.append("contrastive")
        if cfg.loss.use_uncertainty_weighting:
            self.log_sigma = nn.ParameterDict({name: nn.Parameter(torch.zeros(())) for name in self.task_names})
        else:
            self.log_sigma = None

        self.target_tf_encoder = create_ema_model(self.tf_encoder)
        self.target_wave_encoder = create_ema_model(self.wave_encoder)
        self.target_vel_encoder = create_ema_model(self.vel_encoder)

    def _encode_online(
        self,
        tf_log: torch.Tensor,
        pa: torch.Tensor,
        vel: torch.Tensor,
        time_axis: torch.Tensor,
        time_mask: torch.Tensor,
        freq_axis: torch.Tensor,
        meta: torch.Tensor,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        vel_out = self.vel_encoder(vel, time_axis, time_mask)
        tf_out = self.tf_encoder(tf_log, freq_axis, time_axis, time_mask, speed_hidden=vel_out["hidden"], speed_valid_mask=vel_out["mask"])
        wave_out = self.wave_encoder(pa, time_axis, time_mask, speed_hidden=vel_out["hidden"], speed_valid_mask=vel_out["mask"])
        meta_token = self.meta_mlp(meta)
        fusion_out = self.fusion(tf_out["tokens"], tf_out["mask"], wave_out["tokens"], wave_out["mask"], vel_out["tokens"], vel_out["mask"], meta_token)
        return {"tf": tf_out, "wave": wave_out, "vel": vel_out, "fusion": fusion_out}

    @torch.no_grad()
    def _encode_target(
        self,
        tf_log: torch.Tensor,
        pa: torch.Tensor,
        vel: torch.Tensor,
        time_axis: torch.Tensor,
        time_mask: torch.Tensor,
        freq_axis: torch.Tensor,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        vel_out = self.target_vel_encoder(vel, time_axis, time_mask)
        tf_out = self.target_tf_encoder(tf_log, freq_axis, time_axis, time_mask, speed_hidden=vel_out["hidden"], speed_valid_mask=vel_out["mask"])
        wave_out = self.target_wave_encoder(pa, time_axis, time_mask, speed_hidden=vel_out["hidden"], speed_valid_mask=vel_out["mask"])
        return {"tf": tf_out, "wave": wave_out, "vel": vel_out}

    def _jepa_loss(
        self,
        online: Dict[str, Dict[str, torch.Tensor]],
        target: Dict[str, Dict[str, torch.Tensor]],
        time_mask: torch.Tensor,
        return_extras: bool = False,
    ) -> Dict[str, torch.Tensor]:
        masks = build_masks(
            time_mask=time_mask,
            token_masks={"tf": online["tf"]["mask"], "wave": online["wave"]["mask"], "vel": online["vel"]["mask"]},
            cfg=self.cfg.model,
        )
        per_modality: Dict[str, torch.Tensor] = {}
        extras: Dict[str, Dict[str, torch.Tensor]] = {}
        for name in ["tf", "wave", "vel"]:
            pred = self.predictor(online[name]["tokens"], masks.visible_mask[name], masks.target_mask[name], online[name]["time_pos"])
            tgt = target[name]["tokens"].detach()
            per_modality[name] = masked_smooth_l1(pred, tgt, masks.target_mask[name])
            if return_extras:
                extras[name] = {"pred": pred, "target": tgt, "mask": masks.target_mask[name]}
        loss = sum(per_modality.values()) / max(len(per_modality), 1)
        result = {"loss": loss, "per_modality": per_modality, "masks": masks}
        if return_extras:
            result["extras"] = extras
        return result

    def _combine_losses(self, weighted_terms: Dict[str, torch.Tensor]) -> tuple:
        """Combines task losses either with static lambdas or learnable uncertainty weights.

        ``weighted_terms`` values are already lambda-scaled (lambdas act as priors).
        Returns (total_loss, effective_weights) where effective_weights are the
        0.5*exp(-s_i) factors (or 1.0 under static weighting) for logging.
        """
        if self.log_sigma is None:
            total = sum(weighted_terms.values())
            weights = {name: torch.tensor(1.0) for name in weighted_terms}
            return total, weights
        total = torch.tensor(0.0, device=next(iter(weighted_terms.values())).device)
        weights = {}
        for name, term in weighted_terms.items():
            s = self.log_sigma[name]
            w = 0.5 * torch.exp(-s)
            total = total + w * term + 0.5 * s
            weights[name] = w.detach()
        return total, weights

    def _contrastive_loss(self, pooled1: torch.Tensor, pooled2: torch.Tensor, records, return_extras: bool = False):
        z1 = self.proj_head(pooled1)
        z2 = self.proj_head(pooled2)
        embeddings = torch.cat([z1, z2], dim=0)
        positive_sets = build_positive_sets(records)
        loss = multi_positive_info_nce(embeddings, positive_sets, self.cfg.loss.contrastive_temperature)
        if return_extras:
            return loss, {"embeddings": embeddings, "positive_sets": positive_sets}
        return loss

    def forward(self, batch: Dict[str, torch.Tensor], return_extras: bool = False) -> Dict[str, torch.Tensor]:
        need_jepa, need_pair = active_objective_paths(self.cfg, return_extras)
        online_v1 = self._encode_online(
            tf_log=batch["tf_log_view1"],
            pa=batch["pa_view1"],
            vel=batch["vel_view1"],
            time_axis=batch["time_view1"],
            time_mask=batch["time_mask_view1"],
            freq_axis=batch["freq_axis"],
            meta=batch["meta"],
        )
        online_v2 = None
        if need_pair:
            online_v2 = self._encode_online(
                tf_log=batch["tf_log_view2"],
                pa=batch["pa_view2"],
                vel=batch["vel_view2"],
                time_axis=batch["time_view2"],
                time_mask=batch["time_mask_view2"],
                freq_axis=batch["freq_axis"],
                meta=batch["meta"],
            )

        pooled_v1 = online_v1["fusion"]["pooled"]
        zero = pooled_v1.sum() * 0.0
        if need_jepa:
            target_v1 = self._encode_target(
                tf_log=batch["tf_log_view1"],
                pa=batch["pa_view1"],
                vel=batch["vel_view1"],
                time_axis=batch["time_view1"],
                time_mask=batch["time_mask_view1"],
                freq_axis=batch["freq_axis"],
            )
            jepa = self._jepa_loss(online_v1, target_v1, batch["time_mask_view1"], return_extras=return_extras)
        else:
            jepa = {"loss": zero, "per_modality": {name: zero for name in ["tf", "wave", "vel"]}}

        contrastive = zero
        if need_pair:
            pooled_v2 = online_v2["fusion"]["pooled"]
            if return_extras:
                contrastive, contrastive_extras = self._contrastive_loss(
                    pooled_v1, pooled_v2, batch["records"], return_extras=True
                )
            else:
                contrastive = self._contrastive_loss(pooled_v1, pooled_v2, batch["records"])

        pred_meta = self.aux_head(pooled_v1)
        aux_target = batch["meta"][:, AUXILIARY_TARGET_SLICE]
        aux = aux_regression_loss(pred_meta, aux_target)

        weighted_terms = {}
        if self.cfg.loss.lambda_jepa > 0:
            weighted_terms.update({
                "jepa_tf": self.cfg.loss.lambda_jepa * jepa["per_modality"]["tf"],
                "jepa_wave": self.cfg.loss.lambda_jepa * jepa["per_modality"]["wave"],
                "jepa_vel": self.cfg.loss.lambda_jepa * jepa["per_modality"]["vel"],
            })
        if self.cfg.loss.lambda_aux > 0:
            weighted_terms["aux"] = self.cfg.loss.lambda_aux * aux
        if not self.cfg.loss.uncertainty_exclude_contrastive and self.cfg.loss.lambda_contrastive > 0:
            weighted_terms["contrastive"] = self.cfg.loss.lambda_contrastive * contrastive
        if weighted_terms:
            total, task_weights = self._combine_losses(weighted_terms)
        else:
            total = zero
            task_weights = {}
        if self.cfg.loss.uncertainty_exclude_contrastive and self.cfg.loss.lambda_contrastive > 0:
            total = total + self.cfg.loss.lambda_contrastive * contrastive
            task_weights["contrastive"] = torch.tensor(1.0, device=total.device)

        vic = zero
        if self.cfg.model.use_vicreg or self.cfg.loss.lambda_vicreg > 0:
            vic = vicreg_loss(pooled_v1, online_v2["fusion"]["pooled"], self.cfg.loss)
            total = total + self.cfg.loss.lambda_vicreg * vic

        output = {
            "loss": total,
            "loss_jepa": jepa["loss"],
            "loss_jepa_tf": jepa["per_modality"]["tf"],
            "loss_jepa_wave": jepa["per_modality"]["wave"],
            "loss_jepa_vel": jepa["per_modality"]["vel"],
            "loss_contrastive": contrastive,
            "loss_aux": aux,
            "loss_vicreg": vic,
            "task_weights": task_weights,
            "pooled_online": pooled_v1,
            "pooled_online_v2": online_v2["fusion"]["pooled"] if online_v2 is not None else None,
            "pred_meta": pred_meta,
            "aux_target": aux_target,
        }
        if return_extras:
            output["jepa_extras"] = jepa["extras"]
            output["contrastive_extras"] = contrastive_extras
        return output
