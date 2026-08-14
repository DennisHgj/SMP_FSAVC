"""Audio-visual dataset and preprocessing used by SMP."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class AudioVisualDataset(Dataset):
    """Load audio, frames, a semantic prompt, and a label from CSV rows.

    The first three columns are ``clip_id``, integer ``label``, and
    ``semantic_prompt``. The prompt may be a VLM-generated caption or the same
    fixed literal string ``a video of [label]`` for every sample; ``[label]``
    is not replaced with a class name. An optional fourth class-name column,
    used by VGGSound100 annotations, is accepted but not fed to SMP.
    """

    def __init__(
        self,
        annotations: str | Path | pd.DataFrame,
        audio_dir: str | Path,
        visual_dir: str | Path,
        num_frames: int = 8,
        spectrogram_mean: float = 2.5812705,
        spectrogram_std: float = 24.051544,
    ) -> None:
        super().__init__()
        if isinstance(annotations, (str, Path)):
            self.annotations = pd.read_csv(annotations, header=None)
        else:
            self.annotations = annotations.reset_index(drop=True).copy()
        if self.annotations.shape[1] < 3:
            raise ValueError(
                "Annotations require at least three columns: "
                "clip_id, label, semantic_prompt[, class_name]"
            )

        self.audio_dir = Path(audio_dir).expanduser()
        self.visual_dir = Path(visual_dir).expanduser()
        self.num_frames = num_frames
        self.sample_rate = 16_000
        self.spectrogram_mean = spectrogram_mean
        self.spectrogram_std = spectrogram_std

        self.image_transform = transforms.Compose(
            (
                transforms.Resize(224, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            )
        )
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=128,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self) -> int:
        return len(self.annotations)

    def _load_audio(self, clip_id: str) -> torch.Tensor:
        path = self.audio_dir / f"{clip_id}.wav"
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        waveform, sample_rate = torchaudio.load(path, normalize=True)
        # Match the paper experiments: use the first channel for non-mono files.
        waveform = waveform[0]
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.sample_rate
            )
        waveform = (waveform - waveform.mean()) / waveform.std().clamp_min(1e-6)
        spectrogram = self.amplitude_to_db(self.mel_transform(waveform))
        spectrogram = (
            spectrogram - self.spectrogram_mean
        ) / self.spectrogram_std
        return spectrogram.to(torch.float32)

    def _load_frames(self, clip_id: str) -> torch.Tensor:
        directory = self.visual_dir / clip_id
        if not directory.is_dir():
            raise FileNotFoundError(f"Frame directory not found: {directory}")
        frame_paths = sorted(
            path
            for path in directory.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if not frame_paths:
            raise FileNotFoundError(f"No image frames found in: {directory}")

        indices = np.linspace(0, len(frame_paths) - 1, self.num_frames, dtype=int)
        frames = []
        for index in indices:
            with Image.open(frame_paths[index]) as image:
                frames.append(self.image_transform(image.convert("RGB")))
        return torch.stack(frames).to(torch.float32)

    def __getitem__(self, index: int):
        row = self.annotations.iloc[index]
        clip_id = str(row.iloc[0])
        label = int(row.iloc[1])
        # Column 2 is deliberately consumed verbatim: users can provide either
        # generated captions or precomputed static prompts in the same schema.
        prompt = str(row.iloc[2])
        return self._load_audio(clip_id), self._load_frames(clip_id), prompt, label
