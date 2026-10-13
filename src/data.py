from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F


class SpecScaler(torch.nn.Module):
    def forward(self, spectrogram):
        return torch.log(spectrogram.clamp(1e-9, 1e9))


class WaveformAugment(torch.nn.Module):
    """Low-cost waveform augmentation that does not need external data."""

    def __init__(
        self,
        sample_rate: int = 16000,
        max_shift_ms: float = 100.0,
        max_gain_db: float = 6.0,
        noise_probability: float = 0.5,
        min_snr_db: float = 10.0,
        max_snr_db: float = 30.0,
    ):
        super().__init__()
        self.max_shift = round(sample_rate * max_shift_ms / 1000)
        self.max_gain_db = max_gain_db
        self.noise_probability = noise_probability
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.max_shift > 0:
            shift = int(
                torch.randint(
                    -self.max_shift,
                    self.max_shift + 1,
                    size=(1,),
                ).item()
            )
            if shift > 0:
                waveform = F.pad(waveform[..., :-shift], (shift, 0))
            elif shift < 0:
                waveform = F.pad(waveform[..., -shift:], (0, -shift))

        gain_db = (
            torch.rand((), device=waveform.device) * 2.0 - 1.0
        ) * self.max_gain_db
        waveform = waveform * torch.pow(10.0, gain_db / 20.0)

        if torch.rand(()).item() < self.noise_probability:
            snr_db = torch.empty(()).uniform_(
                self.min_snr_db,
                self.max_snr_db,
            )
            signal_rms = waveform.square().mean().sqrt().clamp_min(1e-6)
            noise = torch.randn_like(waveform)
            noise_rms = noise.square().mean().sqrt().clamp_min(1e-6)
            target_noise_rms = signal_rms / torch.pow(10.0, snr_db / 20.0)
            waveform = waveform + noise * (target_noise_rms / noise_rms)

        return waveform.clamp(-1.0, 1.0)


class SpotterDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        manifest_path: str,
        idx_to_keyword: List[str],
        transforms: List[torch.nn.Module],
        test: Optional[bool] = False,
        waveform_transform: Optional[torch.nn.Module] = None,
        sample_rate: int = 16000,
        num_samples: int = 16000,
    ):
        super().__init__()

        self.transform = torch.nn.Sequential(*transforms)
        self.waveform_transform = waveform_transform
        self.sample_rate = sample_rate
        self.num_samples = num_samples

        manifest = pd.read_csv(manifest_path)
        parent = Path(manifest_path).parent

        self.wav_files = [
            Path(wav_path)
            if Path(wav_path).is_absolute()
            else parent / Path(wav_path)
            for wav_path in manifest.path
        ]

        keyword_to_idx = {keyword: idx for idx, keyword in enumerate(idx_to_keyword)}
        self.labels = (
            [-1] * manifest.shape[0]
            if test
            else [keyword_to_idx[keyword] for keyword in manifest.label]
        )

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        wav, sample_rate = torchaudio.load(self.wav_files[idx])
        wav = wav.mean(dim=0, keepdim=True)

        if sample_rate != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sample_rate, self.sample_rate)

        if wav.shape[-1] < self.num_samples:
            total_padding = self.num_samples - wav.shape[-1]
            left_padding = total_padding // 2
            wav = F.pad(wav, (left_padding, total_padding - left_padding))
        elif wav.shape[-1] > self.num_samples:
            offset = (wav.shape[-1] - self.num_samples) // 2
            wav = wav[..., offset : offset + self.num_samples]

        if self.waveform_transform is not None:
            wav = self.waveform_transform(wav)

        features = self.transform(wav)
        return idx, wav[0], features, self.labels[idx]


def collator(
    data: List[Tuple[int, torch.Tensor, torch.Tensor, int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    ids, _, specs, labels = zip(*data)

    ids_tensor = torch.Tensor(ids).long()
    spec_tensor = torch.cat(specs)
    label_tensor = torch.Tensor(labels).long()

    return ids_tensor, spec_tensor, label_tensor
