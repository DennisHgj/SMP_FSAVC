import torch
import torch.nn as nn

from models.Latent_attention import LatentAttention
from utils.functions import QuickGELU


class T_AVeL(nn.Module):
    def __init__(self, dim, latent_attention_loc):
        super(T_AVeL, self).__init__()
        self.latent_attention_loc = latent_attention_loc
        self.dim = dim

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)

        # 2D-CNN for spectrogram
        self.spec_down = nn.Linear(768, dim)
        self.spec_up = nn.Linear(dim, 768)
        nn.init.xavier_uniform_(self.spec_down.weight)
        nn.init.zeros_(self.spec_down.bias)
        nn.init.xavier_uniform_(self.spec_up.weight)
        nn.init.zeros_(self.spec_up.bias)

        self.spec_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.spec_conv.weight)
        nn.init.zeros_(self.spec_conv.bias)

        # 3D-CNN for RGB images
        self.rgb_down = nn.Linear(768, dim)
        self.rgb_up = nn.Linear(dim, 768)
        nn.init.xavier_uniform_(self.rgb_down.weight)
        nn.init.zeros_(self.rgb_down.bias)
        nn.init.xavier_uniform_(self.rgb_up.weight)
        nn.init.zeros_(self.rgb_up.bias)

        self.rgb_conv = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1,
                                  groups=dim)  # depthwise convolution
        nn.init.xavier_uniform_(self.rgb_conv.weight)
        nn.init.zeros_(self.rgb_conv.bias)

        self.text_down = nn.Linear(768, dim)
        self.text_up = nn.Linear(dim, 768)
        nn.init.xavier_uniform_(self.text_down.weight)
        nn.init.zeros_(self.text_down.bias)
        nn.init.xavier_uniform_(self.text_up.weight)
        nn.init.zeros_(self.text_up.bias)

        self.spec_scale = nn.Parameter(torch.zeros(1))
        self.rgb_scale = nn.Parameter(torch.zeros(1))
        self.text_scale = nn.Parameter(torch.zeros(1))

        if 'cma_1' in self.latent_attention_loc:
            self.latent_attention1 = LatentAttention()
        if 'cma_2' in self.latent_attention_loc:
            self.latent_attention2 = LatentAttention()

    def conv_spec(self, x, shapes):
        # Audio 2D Conv
        f_dim, t_dim, no_of_frames, rgb_h, rgb_w = shapes
        B, _, C = x.shape

        x_patch = x[:, 1:].reshape(B, f_dim, t_dim, self.dim).permute(0, 3, 1, 2)
        x_patch = self.spec_conv(x_patch)
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, f_dim * t_dim, self.dim)

        x_cls = x[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        x_cls = self.spec_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_ = torch.cat([x_cls, x_patch], dim=1)
        x_ = self.act(x_)
        x_ = self.dropout(x_)
        return x_

    def conv_imgs(self, y_down, shapes):
        f_dim, t_dim, no_of_frames, rgb_h, rgb_w = shapes
        B = y_down.shape[0]
        # Visual 3D Conv
        y_patch = y_down[:, 1:].permute(0, 2, 1).reshape(B, self.dim, no_of_frames, rgb_h, rgb_w)
        y_patch = self.rgb_conv(y_patch)
        y_patch = y_patch.reshape(B, self.dim, no_of_frames * rgb_h * rgb_w).permute(0, 2, 1)

        y_cls = y_down[:, :1].permute(0, 2, 1).reshape(B, self.dim, 1, 1, 1)
        y_cls = self.rgb_conv(y_cls)
        y_cls = y_cls.reshape(B, self.dim, 1).permute(0, 2, 1)

        y_ = torch.cat([y_cls, y_patch], dim=1)
        y_ = self.act(y_)
        y_ = self.dropout(y_)
        return y_

    def forward(self, x, y, z, shapes):

        B, _, C = x.shape

        if 'cma_fusion_0' in self.latent_attention_loc:
            x, y, z = self.latent_fusion0(x, y, z)

        x_down = self.spec_down(x)
        x_down = self.act(x_down)

        y_down = self.rgb_down(y)
        y_down = self.act(y_down)

        z_down = self.text_down(z)
        z_down = self.act(z_down)

        if 'cma_1' in self.latent_attention_loc:  # cross modal attention at low rank and before convolution
            x_down, y_down, z_down = self.latent_attention1(x_down, y_down, z_down)

        x_ = self.conv_spec(x_down, shapes)
        y_ = self.conv_imgs(y_down, shapes)

        z_ = self.act(z_down)
        z_ = self.dropout(z_)
        ########################################################################################
        if 'cma_2' in self.latent_attention_loc:  # cross modal attention at low rank and after convolution
            x_, y_, z_ = self.latent_attention2(x_, y_, z_)

        # Adapter up
        x_up = self.spec_up(x_)
        y_up = self.rgb_up(y_)
        z_up = self.text_up(z_)

        return x_up * self.spec_scale, y_up * self.rgb_scale, z_up * self.text_scale
