
import torch
import librosa

class AudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_df,
        filenpath_col="filepath",
        target_col="target",
        sample_rate=16000,
        normalize_audio=True,
        audio_transforms=None,
    ):        
        self.df = input_df.reset_index(drop=True)

        self.filenpath_col = filenpath_col
        self.target_col = target_col

        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio

        self.audio_transforms = audio_transforms

    def __len__(self):
        return len(self.df)

    def _prepare_sample(self, idx: int):
        au, sr = librosa.load(self.df[self.filenpath_col].iloc[idx], sr=self.sample_rate)
        assert sr == self.sample_rate
        # We know that all samples are of the same length = 5 sec and contains only one channel
        assert len(au.shape) == 1
        assert au.shape[0] == self.sample_rate * 5

        target_idx = self.df[self.target_col].iloc[idx]

        if self.audio_transforms is not None:
            au = self.audio_transforms(samples=au, sample_rate=sr)

        if self.normalize_audio:
            au = librosa.util.normalize(au)

        return torch.from_numpy(au).float(), target_idx

    def __getitem__(self, idx: int):
        return self._prepare_sample(idx)