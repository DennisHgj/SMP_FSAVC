import torch.nn as nn
from models.T_AVeL import T_AVeL

class FusionBlock(nn.Module):
    def __init__(self, Encoder1, Encoder2, caption_encoder, dim, latent_attention_loc, T_AVeL_loc):
        super(FusionBlock, self).__init__()

        # Fusion Parameters

        self.caption_encoder = caption_encoder

        # Attention Layer
        self.audio_norm1 = Encoder1.norm1
        self.audio_attn = Encoder1.attn
        self.audio_norm2 = Encoder1.norm2  # Feed Forward Layers
        self.audio_mlp = Encoder1.mlp

        self.video_norm1 = Encoder2.norm1
        self.video_attn = Encoder2.attn
        self.video_norm2 = Encoder2.norm2
        self.video_mlp = Encoder2.mlp

        self.caption_norm1 = caption_encoder.layer_norm1
        self.caption_attn = caption_encoder.self_attn
        self.caption_norm2 = caption_encoder.layer_norm2
        self.caption_mlp = caption_encoder.mlp

        self.T_AVeL_loc = T_AVeL_loc

        if "1" in self.T_AVeL_loc:
            self.AVT_adapter_1 = T_AVeL(dim, latent_attention_loc)

        if "2" in self.T_AVeL_loc:
            self.AVT_adapter_2 = T_AVeL(dim, latent_attention_loc)

    def forward(self, x, y, z, shapes):
        # Attn skip connections

        z_hidden_states = z[0]
        z_attention_mask = z[1]
        z_causal_am = z[2]


        if "1" in self.T_AVeL_loc:

            x_f, y_f, z_f = self.AVT_adapter_1(self.audio_norm1(x), self.video_norm1(y),
                                               self.caption_norm1(z_hidden_states), shapes)
            x = x + self.audio_attn(self.audio_norm1(x)) + x_f
            y = y + self.video_attn(self.video_norm1(y)) + y_f
            z_hidden_states = z_hidden_states + self.caption_attn(
                hidden_states=self.caption_norm1(z_hidden_states),
                attention_mask=z_attention_mask,
                causal_attention_mask=z_causal_am,
            )[0] + z_f

        else:
            x = x + self.audio_attn(self.audio_norm1(x))
            y = y + self.video_attn(self.video_norm1(y))
            z_hidden_states = z_hidden_states + self.caption_attn(
                hidden_states=self.caption_norm1(z_hidden_states),
                attention_mask=z_attention_mask,
                causal_attention_mask=z_causal_am,
            )[0]

        if "2" in self.T_AVeL_loc:
            x_f, y_f, z_f = self.AVT_adapter_2(self.audio_norm2(x), self.video_norm2(y),
                                               self.caption_norm2(z_hidden_states), shapes)
            x = x + self.audio_mlp(self.audio_norm2(x)) + x_f
            y = y + self.video_mlp(self.video_norm2(y)) + y_f
            z_hidden_states = z_hidden_states + self.caption_mlp(self.caption_norm2(z_hidden_states)) + z_f

        else:
            x = x + self.audio_mlp(self.audio_norm2(x))
            y = y + self.video_mlp(self.video_norm2(y))
            z_hidden_states = z_hidden_states + self.caption_mlp(self.caption_norm2(z_hidden_states))

        z = (z_hidden_states, z_attention_mask, z_causal_am)
        return x, y, z
