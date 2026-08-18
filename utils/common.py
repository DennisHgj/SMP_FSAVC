"""Common helpers shared by pretraining and few-shot experiments."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable task sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_audio_visual(batch):
    """Pad variable-duration spectrograms and stack frames and labels."""
    spectrograms, frames, prompts, labels = zip(*batch)
    transposed = [spectrogram.transpose(0, 1) for spectrogram in spectrograms]
    padded = torch.nn.utils.rnn.pad_sequence(
        transposed, batch_first=True, padding_value=0.0
    ).transpose(1, 2)
    return (
        padded,
        torch.stack(frames),
        list(prompts),
        torch.as_tensor(labels, dtype=torch.long),
    )


def resolve_device(requested: str) -> torch.device:
    """Validate a requested device and provide a clear CUDA error."""
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Device {requested!r} was requested, but CUDA is unavailable. "
            "Use --device cpu or install a CUDA-enabled PyTorch build."
        )
    return device


def save_json(path: str | Path, payload: Any) -> None:
    """Write JSON atomically enough for a single-process experiment."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def trainable_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
