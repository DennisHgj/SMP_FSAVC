"""Tri-stream Transformer block with P-AVeL in the selected location."""

from torch import nn

from .p_avel import PromptRefinedAVLearner


class SMPFusionBlock(nn.Module):
    """Wrap paired ViT blocks and a CLIP text block with P-AVeL adapters."""

    def __init__(
        self,
        audio_block: nn.Module,
        visual_block: nn.Module,
        text_block: nn.Module,
        hidden_dim: int,
        adapter_dim: int,
        adapter_location: str,
        attention_locations: str,
    ) -> None:
        super().__init__()
        if adapter_location not in {"attention", "mlp", "both"}:
            raise ValueError(
                "adapter_location must be one of: attention, mlp, both"
            )

        self.audio_norm1 = audio_block.norm1
        self.audio_attn = audio_block.attn
        self.audio_norm2 = audio_block.norm2
        self.audio_mlp = audio_block.mlp

        self.video_norm1 = visual_block.norm1
        self.video_attn = visual_block.attn
        self.video_norm2 = visual_block.norm2
        self.video_mlp = visual_block.mlp

        # The alias preserves state-dict compatibility with the research model.
        self.caption_encoder = text_block
        self.caption_norm1 = text_block.layer_norm1
        self.caption_attn = text_block.self_attn
        self.caption_norm2 = text_block.layer_norm2
        self.caption_mlp = text_block.mlp

        self.adapter_location = adapter_location
        if adapter_location in {"attention", "both"}:
            self.AVT_adapter_1 = PromptRefinedAVLearner(
                hidden_dim, adapter_dim, attention_locations
            )
        if adapter_location in {"mlp", "both"}:
            self.AVT_adapter_2 = PromptRefinedAVLearner(
                hidden_dim, adapter_dim, attention_locations
            )

    def _text_attention_forward(self, hidden_states, attention_mask, causal_mask):
        return self.caption_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_mask,
        )[0]

    def forward(self, audio, visual, text_state, shapes):
        text, attention_mask, causal_mask = text_state

        if self.adapter_location in {"attention", "both"}:
            audio_delta, visual_delta, text_delta = self.AVT_adapter_1(
                self.audio_norm1(audio),
                self.video_norm1(visual),
                self.caption_norm1(text),
                shapes,
            )
        else:
            audio_delta = visual_delta = text_delta = 0.0

        audio = audio + self.audio_attn(self.audio_norm1(audio)) + audio_delta
        visual = visual + self.video_attn(self.video_norm1(visual)) + visual_delta
        text = (
            text
            + self._text_attention_forward(
                self.caption_norm1(text), attention_mask, causal_mask
            )
            + text_delta
        )

        if self.adapter_location in {"mlp", "both"}:
            audio_delta, visual_delta, text_delta = self.AVT_adapter_2(
                self.audio_norm2(audio),
                self.video_norm2(visual),
                self.caption_norm2(text),
                shapes,
            )
        else:
            audio_delta = visual_delta = text_delta = 0.0

        audio = audio + self.audio_mlp(self.audio_norm2(audio)) + audio_delta
        visual = visual + self.video_mlp(self.video_norm2(visual)) + visual_delta
        text = text + self.caption_mlp(self.caption_norm2(text)) + text_delta
        return audio, visual, (text, attention_mask, causal_mask)
