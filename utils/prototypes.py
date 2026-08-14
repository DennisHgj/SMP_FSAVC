"""Prototype estimation and Prompt-tuned Prototypical Regularization."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class PrototypeSet:
    audio: torch.Tensor
    visual: torch.Tensor
    text: torch.Tensor


def pairwise_euclidean(inputs: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Vectorized Euclidean distance between two sets of embeddings."""
    return torch.cdist(inputs, prototypes, p=2)


def class_prototypes(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Return class means and reject annotation sets with missing classes."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [samples, dimensions]")
    labels = labels.to(device=embeddings.device, dtype=torch.long)
    totals = embeddings.new_zeros((num_classes, embeddings.shape[1]))
    totals.index_add_(0, labels, embeddings)
    counts = torch.bincount(labels, minlength=num_classes).to(embeddings.dtype)
    missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
    if missing:
        raise ValueError(f"No samples were available for classes: {missing}")
    return totals / counts.unsqueeze(1)


@torch.no_grad()
def estimate_prototypes(
    model: nn.Module,
    data_loader,
    num_classes: int,
    device: torch.device,
    previous: PrototypeSet | None = None,
    momentum: float = 0.0,
) -> PrototypeSet:
    """Estimate audio, visual, and prompt prototypes over a data loader."""
    if not 0.0 <= momentum < 1.0:
        raise ValueError("momentum must be in [0, 1)")
    was_training = model.training
    model.eval()
    audio_embeddings, visual_embeddings, text_embeddings, labels = [], [], [], []
    for spectrograms, frames, prompts, batch_labels in data_loader:
        audio, visual, _, text = model(
            spectrograms.to(device), frames.to(device), prompts
        )
        audio_embeddings.append(audio)
        visual_embeddings.append(visual)
        text_embeddings.append(text)
        labels.append(batch_labels.to(device))

    if was_training:
        model.train()
    if not labels:
        raise ValueError("Cannot estimate prototypes from an empty data loader")

    all_labels = torch.cat(labels)
    current = PrototypeSet(
        class_prototypes(torch.cat(audio_embeddings), all_labels, num_classes),
        class_prototypes(torch.cat(visual_embeddings), all_labels, num_classes),
        class_prototypes(torch.cat(text_embeddings), all_labels, num_classes),
    )
    if previous is None or momentum == 0.0:
        return current
    return PrototypeSet(
        (1.0 - momentum) * current.audio + momentum * previous.audio,
        (1.0 - momentum) * current.visual + momentum * previous.visual,
        (1.0 - momentum) * current.text + momentum * previous.text,
    )


def prompt_tuned_regularization(
    audio_embeddings: torch.Tensor,
    visual_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    labels: torch.Tensor,
    prototypes: PrototypeSet,
    criterion: nn.Module,
    alpha: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the paper's P-PR term for one training batch.

    Prompt quality determines how strongly text prototypes tune audio/visual
    prototypes. The relative correct-class confidence then regularizes only the
    weaker modality, with the coefficient clipped to [0, alpha].
    """
    text_logits = -pairwise_euclidean(text_embeddings, prototypes.text)
    correct_text = torch.softmax(text_logits, dim=1).gather(
        1, labels[:, None]
    ).mean()
    prompt_weight = (torch.sin(torch.pi * correct_text - torch.pi / 2) + 1) / 2

    tuned_audio = prototypes.audio + prompt_weight * prototypes.text
    tuned_visual = prototypes.visual + prompt_weight * prototypes.text
    audio_logits = -pairwise_euclidean(audio_embeddings, tuned_audio)
    visual_logits = -pairwise_euclidean(visual_embeddings, tuned_visual)

    audio_confidence = torch.softmax(audio_logits, dim=1).gather(
        1, labels[:, None]
    ).sum()
    visual_confidence = torch.softmax(visual_logits, dim=1).gather(
        1, labels[:, None]
    ).sum()
    ratio = audio_confidence / visual_confidence.clamp_min(1e-8)

    audio_weight = torch.clamp(ratio.reciprocal() - 1.0, 0.0, 1.0) * alpha
    visual_weight = torch.clamp(ratio - 1.0, 0.0, 1.0) * alpha
    regularization = (
        audio_weight * criterion(audio_logits, labels)
        + visual_weight * criterion(visual_logits, labels)
    )
    metrics = {
        "prompt_weight": float(prompt_weight.detach()),
        "audio_visual_ratio": float(ratio.detach()),
        "audio_weight": float(audio_weight.detach()),
        "visual_weight": float(visual_weight.detach()),
    }
    return regularization, metrics
