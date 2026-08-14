"""Late-fusion classification head used by SMP."""

from collections.abc import Sequence

import torch
from torch import nn


class ConcatClassificationHead(nn.Module):
    """Classify the concatenated audio and visual CLS embeddings."""

    def __init__(
        self,
        feature_dim: int,
        num_modalities: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(num_modalities * feature_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # Keep the historical attribute name for checkpoint compatibility.
        self.fc_action = nn.Linear(hidden_dim, num_classes)

        nn.init.normal_(self.fc1.weight, std=0.001)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc_action.weight, std=0.001)
        nn.init.zeros_(self.fc_action.bias)

    def forward(self, embeddings: Sequence[torch.Tensor]) -> torch.Tensor:
        features = torch.cat(tuple(embeddings), dim=1)
        return self.fc_action(self.dropout(self.activation(self.fc1(features))))
