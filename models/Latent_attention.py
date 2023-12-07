import torch
import torch.nn as nn


class LatentAttention(nn.Module):
    def __init__(self):
        super(LatentAttention, self).__init__()
        self.T_A_gate = nn.Parameter(torch.zeros(1))
        self.T_V_gate = nn.Parameter(torch.zeros(1))

        self.latent_gate1 = nn.Parameter(torch.ones(1) * 0.5)
        self.latent_gate2 = nn.Parameter(torch.ones(1) * 0.5)

        self.A_T_V_gate = nn.Parameter(torch.zeros(1))
        self.V_T_A_gate = nn.Parameter(torch.zeros(1))
        self.one = torch.ones(1)

    def cross_attention(self, q, k, v):  # requires q,k,v to have same dimension
        B, N, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * (C ** -0.5)  # scaling
        attn = attn.softmax(dim=-1)
        x = (attn @ v).reshape(B, N, C)
        return x

    def forward(self, audio_tokens, visual_tokens, text_tokens):
        audio_latent = self.cross_attention(text_tokens, audio_tokens, audio_tokens)
        visual_latent = self.cross_attention(text_tokens, visual_tokens, visual_tokens)

        one = self.one.to(audio_latent.device)

        if self.latent_gate1.item() >= 1:
            T_A_latent = self.latent_gate1 * audio_latent
        elif self.latent_gate1.item() <= 0:
            T_A_latent = text_tokens
        else:
            T_A_latent = (one - self.latent_gate1) * text_tokens + self.latent_gate1 * audio_latent

        if self.latent_gate2.item() >= 1:
            T_V_latent = self.latent_gate2 * visual_latent
        elif self.latent_gate2.item() <= 0:
            T_V_latent = text_tokens
        else:
            T_V_latent = (one - self.latent_gate2) * text_tokens + self.latent_gate2 * visual_latent

        audio_tokens = audio_tokens + self.A_T_V_gate * self.cross_attention(audio_tokens, T_V_latent, T_V_latent)

        visual_tokens = visual_tokens + self.V_T_A_gate * self.cross_attention(visual_tokens, T_A_latent, T_A_latent)

        text_tokens = text_tokens + self.T_A_gate * audio_latent + self.T_V_gate * visual_latent

        return audio_tokens, visual_tokens, text_tokens
