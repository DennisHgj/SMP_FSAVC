"""Complete Semantic Modulated Prompting model."""

from pathlib import Path

import timm
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoProcessor, CLIPModel

from .classification_head import ConcatClassificationHead
from .smp_fusion_block import SMPFusionBlock


DEFAULT_VIT_MODEL = "vit_base_patch16_224.augreg_in21k"
DEFAULT_TEXT_MODEL = "openai/clip-vit-large-patch14"


class SMPModel(nn.Module):
    """Dual-stream ViT model implementing P-AVeL; P-PR is applied in the loss."""

    def __init__(
        self,
        num_classes: int,
        adapter_dim: int = 16,
        begin_layer: int = 4,
        adapter_location: str = "mlp",
        attention_locations: str = "before,after",
        tuning: str = "bias",
        vit_model: str = DEFAULT_VIT_MODEL,
        vit_checkpoint: str | None = None,
        text_model: str = DEFAULT_TEXT_MODEL,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        checkpoint_path = str(Path(vit_checkpoint).expanduser()) if vit_checkpoint else ""
        create_options = {
            "pretrained": pretrained_backbone and not checkpoint_path,
            "num_classes": 0,
        }
        if checkpoint_path:
            create_options["checkpoint_path"] = checkpoint_path

        self.v1 = timm.create_model(vit_model, **create_options)
        self.v2 = timm.create_model(vit_model, **create_options)
        self.processor = AutoProcessor.from_pretrained(text_model)
        self.text_model = CLIPModel.from_pretrained(text_model).text_model

        hidden_dim = self.v1.embed_dim
        if hidden_dim != self.v2.embed_dim or hidden_dim != self.text_model.config.hidden_size:
            raise ValueError(
                "Audio ViT, visual ViT, and CLIP text hidden dimensions must match"
            )
        if not 0 <= begin_layer < len(self.v1.blocks):
            raise ValueError("begin_layer must index an existing Transformer block")

        self.begin_layer = begin_layer
        self.num_encoder_layers = len(self.v1.blocks)
        self._configure_trainable_parameters(tuning)

        # Retain historical names so checkpoints from the research code load.
        self.spec_conv = self.v1.patch_embed.proj
        self.rgb_conv = self.v2.patch_embed.proj
        self.spec_pos_embed = self.v1.pos_embed
        self.rgb_pos_embed = self.v2.pos_embed
        self.spec_cls_token = self.v1.cls_token
        self.rgb_cls_token = self.v2.cls_token

        for index in range(begin_layer, self.num_encoder_layers):
            self.v2.blocks[index] = SMPFusionBlock(
                self.v1.blocks[index],
                self.v2.blocks[index],
                self.text_model.encoder.layers[index],
                hidden_dim=hidden_dim,
                adapter_dim=adapter_dim,
                adapter_location=adapter_location,
                attention_locations=attention_locations,
            )

        self.spec_post_norm = self.v1.norm
        self.rgb_post_norm = self.v2.norm
        self.fusion_classification_net = ConcatClassificationHead(
            hidden_dim, num_modalities=2, num_classes=num_classes
        )

    def _configure_trainable_parameters(self, tuning: str) -> None:
        if tuning not in {"none", "norm", "bias", "all"}:
            raise ValueError("tuning must be one of: none, norm, bias, all")

        self.text_model.requires_grad_(False)
        self.v1.requires_grad_(False)
        self.v2.requires_grad_(False)
        # The audio branch adapts an ImageNet projection to spectrograms.
        self.v1.patch_embed.proj.requires_grad_(True)
        self.v1.pos_embed.requires_grad_(True)
        self.v1.cls_token.requires_grad_(True)

        if tuning == "all":
            self.v1.requires_grad_(True)
            self.v2.requires_grad_(True)
        elif tuning in {"norm", "bias"}:
            for backbone in (self.v1, self.v2):
                for block in backbone.blocks:
                    block.norm1.requires_grad_(True)
                    block.norm2.requires_grad_(True)
                    if tuning == "bias":
                        for name, parameter in block.named_parameters():
                            if name.endswith("bias"):
                                parameter.requires_grad_(True)

    @staticmethod
    def _interpolate_position_embedding(
        position_embedding: torch.Tensor, token_count: int
    ) -> torch.Tensor:
        return F.interpolate(
            position_embedding.transpose(1, 2),
            size=token_count,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def _audio_tokens(self, spectrogram: torch.Tensor):
        features = self.spec_conv(spectrogram.unsqueeze(1).repeat(1, 3, 1, 1))
        batch_size, channels, frequency_bins, time_bins = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        tokens = torch.cat((self.spec_cls_token.expand(batch_size, -1, -1), tokens), 1)
        tokens = tokens + self._interpolate_position_embedding(
            self.spec_pos_embed, tokens.shape[1]
        )
        return tokens, frequency_bins, time_bins

    def _visual_tokens(self, frames: torch.Tensor):
        batch_size, num_frames, channels, height, width = frames.shape
        features = self.rgb_conv(
            frames.reshape(batch_size * num_frames, channels, height, width)
        )
        _, feature_dim, patch_height, patch_width = features.shape
        features = features.reshape(
            batch_size, num_frames, feature_dim, patch_height, patch_width
        )
        tokens = features.permute(0, 2, 1, 3, 4).flatten(2).transpose(1, 2)
        tokens = torch.cat((self.rgb_cls_token.expand(batch_size, -1, -1), tokens), 1)
        tokens = tokens + self._interpolate_position_embedding(
            self.rgb_pos_embed, tokens.shape[1]
        )
        return tokens, num_frames, patch_height, patch_width

    @staticmethod
    def _attention_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        expanded = mask[:, None, None, :].to(dtype=dtype)
        return (1.0 - expanded) * torch.finfo(dtype).min

    @staticmethod
    def _causal_mask(
        batch_size: int,
        sequence_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        mask = torch.full(
            (sequence_length, sequence_length),
            torch.finfo(dtype).min,
            dtype=dtype,
            device=device,
        )
        mask = torch.triu(mask, diagonal=1)
        return mask[None, None].expand(batch_size, 1, -1, -1)

    def _text_tokens(self, captions: list[str], device: torch.device):
        encoded = self.processor(text=captions, return_tensors="pt", padding=True)
        encoded = {name: value.to(device) for name, value in encoded.items()}
        input_ids = encoded["input_ids"]
        attention = encoded.get("attention_mask")
        hidden = self.text_model.embeddings(input_ids=input_ids)
        batch_size, sequence_length = input_ids.shape
        pooling_indices = input_ids.to(torch.int).argmax(dim=-1)
        attention_mask = (
            self._attention_mask(attention, hidden.dtype) if attention is not None else None
        )
        causal_mask = self._causal_mask(
            batch_size, sequence_length, hidden.dtype, hidden.device
        )
        return (hidden, attention_mask, causal_mask), pooling_indices

    def forward(
        self,
        spectrogram: torch.Tensor,
        frames: torch.Tensor,
        captions: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        audio, frequency_bins, time_bins = self._audio_tokens(spectrogram)
        visual, num_frames, height, width = self._visual_tokens(frames)
        text_state, pooling_indices = self._text_tokens(captions, spectrogram.device)

        for index in range(self.begin_layer):
            audio = self.v1.blocks[index](audio)
            visual = self.v2.blocks[index](visual)
            text_state = (
                self.text_model.encoder.layers[index](
                    text_state[0],
                    attention_mask=text_state[1],
                    causal_attention_mask=text_state[2],
                )[0],
                text_state[1],
                text_state[2],
            )

        shapes = (frequency_bins, time_bins, num_frames, height, width)
        for index in range(self.begin_layer, self.num_encoder_layers):
            audio, visual, text_state = self.v2.blocks[index](
                audio, visual, text_state, shapes
            )

        audio_embedding = self.spec_post_norm(audio)[:, 0]
        visual_embedding = self.rgb_post_norm(visual)[:, 0]
        text_tokens = self.text_model.final_layer_norm(text_state[0])
        batch_indices = torch.arange(text_tokens.shape[0], device=text_tokens.device)
        text_embedding = text_tokens[batch_indices, pooling_indices]
        logits = self.fusion_classification_net(
            (audio_embedding, visual_embedding)
        )
        return audio_embedding, visual_embedding, logits, text_embedding
