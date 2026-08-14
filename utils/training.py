"""Training and evaluation loops for the paper configuration of SMP."""

from collections.abc import Iterable

import torch
from torch import nn

from .prototypes import PrototypeSet, prompt_tuned_regularization


def train_one_epoch(
    model: nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    prototypes: PrototypeSet,
    device: torch.device,
    alpha: float,
) -> dict[str, float]:
    model.train()
    total_loss = total_correct = total_samples = 0
    metric_totals = {
        "prompt_weight": 0.0,
        "audio_visual_ratio": 0.0,
        "audio_weight": 0.0,
        "visual_weight": 0.0,
    }
    num_batches = 0

    for spectrograms, frames, prompts, labels in data_loader:
        spectrograms = spectrograms.to(device)
        frames = frames.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        audio, visual, logits, text = model(spectrograms, frames, prompts)
        ppr_loss, ppr_metrics = prompt_tuned_regularization(
            audio, visual, text, labels, prototypes, criterion, alpha
        )
        loss = criterion(logits, labels) + ppr_loss
        loss.backward()
        optimizer.step()

        batch_size = labels.numel()
        total_loss += float(loss.detach()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum())
        total_samples += batch_size
        num_batches += 1
        for name, value in ppr_metrics.items():
            metric_totals[name] += value

    if total_samples == 0:
        raise ValueError("Cannot train on an empty data loader")
    result = {
        "loss": total_loss / total_samples,
        "accuracy": 100.0 * total_correct / total_samples,
    }
    result.update(
        {name: value / num_batches for name, value in metric_totals.items()}
    )
    return result


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: Iterable,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = total_correct = total_samples = 0
    for spectrograms, frames, prompts, labels in data_loader:
        labels = labels.to(device)
        _, _, logits, _ = model(
            spectrograms.to(device), frames.to(device), prompts
        )
        batch_size = labels.numel()
        total_loss += float(criterion(logits, labels)) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum())
        total_samples += batch_size
    if total_samples == 0:
        raise ValueError("Cannot evaluate an empty data loader")
    return {
        "loss": total_loss / total_samples,
        "accuracy": 100.0 * total_correct / total_samples,
    }
