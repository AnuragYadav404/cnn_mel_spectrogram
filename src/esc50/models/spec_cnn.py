from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import timm
import torch
from torch import nn
from torchaudio.functional import amplitude_to_DB
from torchaudio.transforms import FrequencyMasking, MelSpectrogram, TimeMasking

from .pooling import AvgPool


class NormalizeMelSpec(nn.Module):
    def __init__(self, eps: float = 1e-6, normalize_standart: bool = True, normalize_minmax: bool = True):
        super().__init__()
        self.eps = eps
        self.normalize_standart = normalize_standart
        self.normalize_minmax = normalize_minmax

    def forward(self, x: torch.Tensor):
        if self.normalize_standart:
            mean = x.mean((-2, -1), keepdim=True)
            std = x.std((-2, -1), keepdim=True)
            x = (x - mean) / (std + self.eps)
        if self.normalize_minmax:
            norm_max = torch.amax(x, dim=(-2, -1), keepdim=True)
            norm_min = torch.amin(x, dim=(-2, -1), keepdim=True)
            x = (x - norm_min) / (norm_max - norm_min + self.eps)
        return x


class CustomMasking(nn.Module):
    def __init__(self, mask_max_length: int, mask_max_masks: int, p: float = 1.0, inplace: bool = True):
        super().__init__()
        assert isinstance(mask_max_masks, int) and mask_max_masks > 0
        self.mask_max_masks = mask_max_masks
        self.mask_max_length = mask_max_length
        self.mask_module = None
        self.p = p
        self.inplace = inplace

    def forward(self, x):
        if not self.inplace:
            output = x.clone()
        for i in range(x.shape[0]):
            if np.random.binomial(n=1, p=self.p):
                n_applies = np.random.randint(low=1, high=self.mask_max_masks + 1)
                for _ in range(n_applies):
                    if self.inplace:
                        x[i : i + 1] = self.mask_module(x[i : i + 1])
                    else:
                        output[i : i + 1] = self.mask_module(output[i : i + 1])
        return x if self.inplace else output


class CustomTimeMasking(CustomMasking):
    def __init__(self, mask_max_length: int, mask_max_masks: int, p: float = 1.0, inplace: bool = True):
        super().__init__(mask_max_length=mask_max_length, mask_max_masks=mask_max_masks, p=p, inplace=inplace)
        self.mask_module = TimeMasking(time_mask_param=mask_max_length)


class CustomFreqMasking(CustomMasking):
    def __init__(self, mask_max_length: int, mask_max_masks: int, p: float = 1.0, inplace: bool = True):
        super().__init__(mask_max_length=mask_max_length, mask_max_masks=mask_max_masks, p=p, inplace=inplace)
        self.mask_module = FrequencyMasking(freq_mask_param=mask_max_length)


class ChannelAgnosticAmplitudeToDB(nn.Module):
    __constants__ = ["multiplier", "amin", "ref_value", "db_multiplier"]

    def __init__(self, stype: str = "power", top_db: Optional[float] = None) -> None:
        super().__init__()
        self.stype = stype
        if top_db is not None and top_db < 0:
            raise ValueError("top_db must be positive value")
        self.top_db = top_db
        self.multiplier = 10.0 if stype == "power" else 20.0
        self.amin = 1e-10
        self.ref_value = 1.0
        self.db_multiplier = math.log10(max(self.amin, self.ref_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() in [3, 4], f"Expected 3D or 4D tensor, but got {x.dim()}D tensor"
        add_fake_channel = False
        if x.dim() == 3:
            x = x.unsqueeze(1)
            add_fake_channel = True

        x_db = amplitude_to_DB(x, self.multiplier, self.amin, self.db_multiplier, self.top_db)
        return x_db.squeeze(1) if add_fake_channel else x_db


class SpecCNNClassifier(nn.Module):
    def __init__(
        self,
        backbone: str,
        device: str,
        n_classes: int,
        classifier_dropout: float = 0.5,
        spec_paramms: Dict[str, Any] | None = None,
        top_db: float = 80.0,
        normalize_config: Dict[str, bool] | None = None,
        pretrained: bool = True,
        timm_kwargs: Optional[Dict] = None,
        spec_augment_config: Optional[Dict[str, Any]] = None,
        pool: Optional[nn.Module] = None,
    ):
        super().__init__()
        timm_kwargs = {} if timm_kwargs is None else timm_kwargs
        spec_paramms = {
            "sample_rate": 32000,
            "n_mels": 128,
            "f_min": 20,
            "n_fft": 1024,
            "hop_length": 512,
            "normalized": True,
        } if spec_paramms is None else spec_paramms
        normalize_config = {
            "normalize_standart": True,
            "normalize_minmax": True,
        } if normalize_config is None else normalize_config

        self.device = device
        self.spectogram_extractor = nn.Sequential(
            MelSpectrogram(**spec_paramms),
            ChannelAgnosticAmplitudeToDB(top_db=top_db),
            NormalizeMelSpec(**normalize_config),
        )

        if spec_augment_config is not None:
            augment_layers = []
            if "freq_mask" in spec_augment_config:
                augment_layers.append(CustomFreqMasking(**spec_augment_config["freq_mask"]))
            if "time_mask" in spec_augment_config:
                augment_layers.append(CustomTimeMasking(**spec_augment_config["time_mask"]))
            self.spec_augment = nn.Sequential(*augment_layers) if augment_layers else None
        else:
            self.spec_augment = None

        self.backbone = timm.create_model(
            backbone,
            features_only=True,
            pretrained=pretrained,
            in_chans=1,
            exportable=True,
            **timm_kwargs,
        )

        self.pool = pool if pool is not None else AvgPool()
        self.classifier = nn.Sequential(
            nn.Dropout(p=classifier_dropout),
            nn.Linear(self.backbone.feature_info.channels()[-1], n_classes),
        )

        self.to(self.device)

    def forward(self, input, return_spec_feature: bool = False, return_cnn_emb: bool = False):
        specs = self.spectogram_extractor(input)
        if self.spec_augment is not None and self.training:
            specs = self.spec_augment(specs)
        if return_spec_feature:
            return specs

        emb = self.backbone(specs.unsqueeze(1))[-1]
        if return_cnn_emb:
            return emb

        bs, ch, _, _ = emb.shape
        emb = self.pool(emb)
        emb = emb.view(bs, ch)
        logits = self.classifier(emb)
        return {"logits": logits}


SpecCNNClasifier = SpecCNNClassifier
