"""Prompt-refined Audio-Visual efficient Learner (P-AVeL)."""

from __future__ import annotations

import torch
from torch import nn

from .prompt_attention import PromptGuidedLatentAttention


class QuickGELU(nn.Module):
    """GELU approximation used by CLIP."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * torch.sigmoid(1.702 * inputs)


class PromptRefinedAVLearner(nn.Module):
    """A bottleneck adapter that aligns audio, video, and prompt tokens."""

    def __init__(
        self,
        hidden_dim: int = 768,
        adapter_dim: int = 16,
        attention_locations: str = "before,after",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        locations = {item.strip() for item in attention_locations.split(",") if item}
        invalid = locations - {"before", "after"}
        if invalid:
            raise ValueError(f"Unknown latent-attention locations: {sorted(invalid)}")

        self.adapter_dim = adapter_dim
        self.use_attention_before = "before" in locations
        self.use_attention_after = "after" in locations
        self.activation = QuickGELU()
        self.dropout = nn.Dropout(dropout)

        self.spec_down = nn.Linear(hidden_dim, adapter_dim)
        self.spec_up = nn.Linear(adapter_dim, hidden_dim)
        self.spec_conv = nn.Conv2d(
            adapter_dim, adapter_dim, kernel_size=3, padding=1
        )

        self.rgb_down = nn.Linear(hidden_dim, adapter_dim)
        self.rgb_up = nn.Linear(adapter_dim, hidden_dim)
        self.rgb_conv = nn.Conv3d(
            adapter_dim,
            adapter_dim,
            kernel_size=3,
            padding=1,
            groups=adapter_dim,
        )

        self.text_down = nn.Linear(hidden_dim, adapter_dim)
        self.text_up = nn.Linear(adapter_dim, hidden_dim)

        self.spec_scale = nn.Parameter(torch.zeros(1))
        self.rgb_scale = nn.Parameter(torch.zeros(1))
        self.text_scale = nn.Parameter(torch.zeros(1))

        if self.use_attention_before:
            self.latent_fusion1 = PromptGuidedLatentAttention()
        if self.use_attention_after:
            self.latent_fusion2 = PromptGuidedLatentAttention()

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for layer in (
            self.spec_down,
            self.spec_up,
            self.rgb_down,
            self.rgb_up,
            self.text_down,
            self.text_up,
        ):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        for layer in (self.spec_conv, self.rgb_conv):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _convolve_audio(
        self,
        tokens: torch.Tensor,
        frequency_bins: int,
        time_bins: int,
    ) -> torch.Tensor:
        batch_size = tokens.shape[0]
        patches = tokens[:, 1:].reshape(
            batch_size, frequency_bins, time_bins, self.adapter_dim
        )
        patches = self.spec_conv(patches.permute(0, 3, 1, 2))
        patches = patches.permute(0, 2, 3, 1).reshape(
            batch_size, frequency_bins * time_bins, self.adapter_dim
        )
        cls_token = tokens[:, :1].transpose(1, 2).unsqueeze(-1)
        cls_token = self.spec_conv(cls_token).flatten(2).transpose(1, 2)
        return self.dropout(self.activation(torch.cat((cls_token, patches), dim=1)))

    def _convolve_visual(
        self,
        tokens: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        batch_size = tokens.shape[0]
        patches = tokens[:, 1:].transpose(1, 2).reshape(
            batch_size, self.adapter_dim, num_frames, height, width
        )
        patches = self.rgb_conv(patches).flatten(2).transpose(1, 2)
        cls_token = tokens[:, :1].transpose(1, 2).reshape(
            batch_size, self.adapter_dim, 1, 1, 1
        )
        cls_token = self.rgb_conv(cls_token).flatten(2).transpose(1, 2)
        return self.dropout(self.activation(torch.cat((cls_token, patches), dim=1)))

    def forward(
        self,
        audio_tokens: torch.Tensor,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        shapes: tuple[int, int, int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        frequency_bins, time_bins, num_frames, height, width = shapes
        audio = self.activation(self.spec_down(audio_tokens))
        visual = self.activation(self.rgb_down(visual_tokens))
        text = self.activation(self.text_down(text_tokens))

        if self.use_attention_before:
            audio, visual, text = self.latent_fusion1(audio, visual, text)

        audio = self._convolve_audio(audio, frequency_bins, time_bins)
        visual = self._convolve_visual(visual, num_frames, height, width)
        text = self.dropout(self.activation(text))

        if self.use_attention_after:
            audio, visual, text = self.latent_fusion2(audio, visual, text)

        return (
            self.spec_scale * self.spec_up(audio),
            self.rgb_scale * self.rgb_up(visual),
            self.text_scale * self.text_up(text),
        )
