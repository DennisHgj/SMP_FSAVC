"""Prompt-guided latent attention from the P-AVeL adapter."""

import torch
from torch import nn


class PromptGuidedLatentAttention(nn.Module):
    """Perform text-aware attention, modality fusion, and token summation.

    Text queries first compress the audio and visual sequences to the text-token
    length. The compressed representations then carry cross-modal information
    back to the full audio and visual sequences. Learnable scalar gates implement
    the weighting coefficients described in the paper.
    """

    def __init__(self) -> None:
        super().__init__()
        # Attribute names intentionally match the paper experiment checkpoint.
        self.T_A_gate = nn.Parameter(torch.zeros(1))
        self.T_V_gate = nn.Parameter(torch.zeros(1))
        self.A_T_V_gate = nn.Parameter(torch.zeros(1))
        self.V_T_A_gate = nn.Parameter(torch.zeros(1))
        self.latent_gate1 = nn.Parameter(torch.full((1,), 0.5))
        self.latent_gate2 = nn.Parameter(torch.full((1,), 0.5))

    @staticmethod
    def cross_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Parameter-free scaled dot-product cross-attention."""
        scale = query.shape[-1] ** -0.5
        weights = torch.softmax((query @ key.transpose(-2, -1)) * scale, dim=-1)
        return weights @ value

    def forward(
        self,
        audio_tokens: torch.Tensor,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Phase 1: text-aware compression of the two long modality sequences.
        audio_latent = self.cross_attention(text_tokens, audio_tokens, audio_tokens)
        visual_latent = self.cross_attention(text_tokens, visual_tokens, visual_tokens)

        text_audio = text_tokens + self.latent_gate1 * audio_latent
        text_visual = text_tokens + self.latent_gate2 * visual_latent

        # Phase 2: cross-modal information returns to the full token sequences.
        audio_update = self.cross_attention(audio_tokens, text_visual, text_visual)
        visual_update = self.cross_attention(visual_tokens, text_audio, text_audio)
        audio_tokens = audio_tokens + self.A_T_V_gate * audio_update
        visual_tokens = visual_tokens + self.V_T_A_gate * visual_update

        # Phase 3: sum prompt and modality-aware tokens into the text stream.
        text_tokens = (
            text_tokens
            + self.T_A_gate * audio_latent
            + self.T_V_gate * visual_latent
        )
        return audio_tokens, visual_tokens, text_tokens
