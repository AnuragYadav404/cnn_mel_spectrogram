from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class GeM(nn.Module):
    def __init__(self, p: float = 1.0, eps: float = 1e-6, trainable: bool = True):
        super().__init__()
        self.eps = eps
        raw_p = torch.log(torch.expm1(torch.tensor(float(p))))
        if trainable:
            self.raw_p = nn.Parameter(raw_p)
        else:
            self.register_buffer("raw_p", raw_p)

    @property
    def p(self):
        return F.softplus(self.raw_p) + self.eps

    def forward(self, x: torch.Tensor):
        x = x.clamp(min=self.eps)
        pooled = F.adaptive_avg_pool2d(x.pow(self.p), (1, 1))
        return pooled.pow(1.0 / self.p)

    def __repr__(self):
        return f"GeM(p={self.p.item():.4f}, eps={self.eps})"


class AvgPool(nn.Module):
    def forward(self, x: torch.Tensor):
        return F.adaptive_avg_pool2d(x, (1, 1))
