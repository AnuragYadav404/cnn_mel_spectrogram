from __future__ import annotations

import random
from typing import Optional

import librosa
import numpy as np
import torch


class AudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_df,
        filenpath_col: str = "filepath",
        target_col: str = "target",
        sample_rate: int = 16000,
        normalize_audio: bool = True,
        audio_transforms=None,
        mixup_prob: float = 0.0,
        mixup_alpha: float = 0.2,
        num_classes: Optional[int] = None,
        training: bool = False,
    ):
        self.df = input_df.reset_index(drop=True)
        self.filenpath_col = filenpath_col
        self.target_col = target_col
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio
        self.audio_transforms = audio_transforms
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha
        self.num_classes = num_classes
        self.training = training

        if self.mixup_prob > 0:
            assert self.num_classes is not None, "You must provide num_classes to mix targets."

    def __len__(self):
        return len(self.df)

    def _load_single_item(self, idx: int):
        waveform, sample_rate = librosa.load(self.df[self.filenpath_col].iloc[idx], sr=self.sample_rate)
        assert sample_rate == self.sample_rate
        assert len(waveform.shape) == 1
        assert waveform.shape[0] == self.sample_rate * 5

        target_idx = self.df[self.target_col].iloc[idx]

        if self.audio_transforms is not None:
            waveform = self.audio_transforms(samples=waveform, sample_rate=sample_rate)

        if self.normalize_audio:
            waveform = librosa.util.normalize(waveform)

        return waveform, target_idx

    def _to_soft_target(self, target_idx: int):
        target = np.zeros(self.num_classes, dtype=np.float32)
        target[target_idx] = 1.0
        return target

    def _prepare_sample(self, idx: int):
        waveform, target_idx = self._load_single_item(idx)

        if self.mixup_prob <= 0 or not self.training:
            target = self._to_soft_target(target_idx) if self.num_classes is not None else target_idx
            return torch.from_numpy(waveform).float(), torch.as_tensor(target)

        target = self._to_soft_target(target_idx)
        if random.random() >= self.mixup_prob:
            return torch.from_numpy(waveform).float(), torch.from_numpy(target).float()

        idx2 = random.randint(0, len(self.df) - 1)
        waveform2, target_idx2 = self._load_single_item(idx2)
        target2 = self._to_soft_target(target_idx2)

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        waveform = lam * waveform + (1 - lam) * waveform2
        target = lam * target + (1 - lam) * target2

        return torch.from_numpy(waveform).float(), torch.from_numpy(target).float()

    def __getitem__(self, idx: int):
        return self._prepare_sample(idx)
