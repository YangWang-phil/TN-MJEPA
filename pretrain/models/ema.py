"""Exponential-moving-average target model utilities."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


def create_ema_model(module: nn.Module) -> nn.Module:
    ema_module = copy.deepcopy(module)
    for param in ema_module.parameters():
        param.requires_grad_(False)
    ema_module.eval()
    return ema_module


@torch.no_grad()
def update_ema_model(online: nn.Module, target: nn.Module, momentum: float) -> None:
    online_state = dict(online.named_parameters())
    for name, param_t in target.named_parameters():
        param_o = online_state[name]
        param_t.data.mul_(momentum).add_(param_o.data, alpha=1.0 - momentum)

    online_buffers = dict(online.named_buffers())
    for name, buffer_t in target.named_buffers():
        buffer_t.copy_(online_buffers[name])
