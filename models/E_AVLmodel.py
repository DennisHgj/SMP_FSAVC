import timm
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from transformers.models.clip.modeling_clip import _expand_mask

from models.ConcatHead import ConcatHead
from models.Fusion_block import FusionBlock


class E_AVLmodel(nn.Module):
    def __init__(self, num_classes, T_AVeL_dim=16, tune="LNBT", latent_attention_loc="cma_1cma_2", pooling=True, T_AVeL_loc='2',
                 begin_layer=4):
        super(E_AVLmodel, self).__init__()

        self.v1 = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=False,
                                    checkpoint_path='/home/guanjiehuang/.cache/huggingface/hub/models--timm--vit_base_patch16_224.augreg_in21k/snapshots/pytorch_model.bin'
                                    )
        self.v2 = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=False,
                                    checkpoint_path='/home/guanjiehuang/.cache/huggingface/hub/models--timm--vit_base_patch16_224.augreg_in21k/snapshots/pytorch_model.bin'
                                    )

        self.processor = CLIPProcessor.from_pretrained("/mnt/sdb/guanjiehuang/clip-vit-large-patch14")
        self.text_model = CLIPModel.from_pretrained("/mnt/sdb/guanjiehuang/clip-vit-large-patch14").text_model
        self.pooling = pooling

        """
        discard unnecessary layers and save parameters
        """
        self.v1.pre_logits = nn.Identity()
        self.v2.pre_logits = nn.Identity()
        self.v1.head = nn.Identity()
        self.v2.head = nn.Identity()
        # discard the Encoder block of v1 completely
        # self.v1.blocks = nn.Identity()

        """
        Freeze parameters
        """
        # Trainable = Spec Conv, Spec+RGB pos embed and cls token, linear classifier

        for p in self.text_model.parameters(): p.requires_grad = False

        # self.processor.required_grad = False
        for p in self.v2.patch_embed.proj.parameters(): p.requires_grad = False
        for p in self.v2.blocks.parameters(): p.requires_grad = False

        for p in self.v1.blocks.parameters(): p.requires_grad = False
        # for p in self.v1.patch_embed.proj.parameters(): p.requires_grad = False

        if tune == 'LNBT':
            for n in range(12):
                for p in self.v2.blocks[n].norm1.parameters(): p.requires_grad = True
                for p in self.v2.blocks[n].norm2.parameters(): p.requires_grad = True

                for p in self.v1.blocks[n].norm1.parameters(): p.requires_grad = True
                for p in self.v1.blocks[n].norm2.parameters(): p.requires_grad = True
            for n in range(12):
                self.v2.blocks[n].attn.qkv.bias.requires_grad = True
                self.v2.blocks[n].attn.proj.bias.requires_grad = True
                self.v2.blocks[n].mlp.fc1.bias.requires_grad = True
                self.v2.blocks[n].mlp.fc2.bias.requires_grad = True

                self.v1.blocks[n].attn.qkv.bias.requires_grad = True
                self.v1.blocks[n].attn.proj.bias.requires_grad = True
                self.v1.blocks[n].mlp.fc1.bias.requires_grad = True
                self.v1.blocks[n].mlp.fc2.bias.requires_grad = True


        elif tune == 'ALL':
            for p in self.v2.patch_embed.proj.parameters(): p.requires_grad = True
            for p in self.v2.blocks.parameters(): p.requires_grad = True

            # for p in self.v1.patch_embed.proj.parameters(): p.requires_grad = True
            for p in self.v1.blocks.parameters(): p.requires_grad = True

        """
        Initialize conv projection, cls token, pos embed and encoders for audio and visual modality
        """
        # conv projection
        self.spec_conv = self.v1.patch_embed.proj
        self.rgb_conv = self.v2.patch_embed.proj

        # cls token and pos embedding
        self.spec_pos_embed = self.v1.pos_embed
        self.rgb_pos_embed = self.v2.pos_embed

        self.spec_cls_token = self.v1.cls_token
        self.rgb_cls_token = self.v2.cls_token

        """
        Initialize Encoder and Final Norm
        """
        self.begin_layer = begin_layer

        for i in range(begin_layer, 12):
            self.v2.blocks[i] = FusionBlock(self.v1.blocks[i], self.v2.blocks[i], self.text_model.encoder.layers[i],
                                            dim=T_AVeL_dim, latent_attention_loc=latent_attention_loc, T_AVeL_loc=T_AVeL_loc)

        # self.audio_visual_blocks = self.v2.blocks

        # final norm
        self.spec_post_norm = self.v1.norm
        self.rgb_post_norm = self.v2.norm

        self.fusion_classification_head = ConcatHead(768, ['RGB', 'Spec'], num_classes, 0.5)

    def forward_spec_features(self, x):  # shape = (bs, 128, 1006)

        x = x.unsqueeze(1)  # shape = (bs, 1, 128, 1006)
        x = x.repeat(1, 3, 1, 1)  # shape = (bs, 3, 128, 1006)
        x = self.spec_conv(x)
        B, dim, f_dim, t_dim = x.shape  # shape = (bs, 768, 8, 62)
        x = torch.reshape(x, (B, dim, f_dim * t_dim))  # shape = (bs, 768, 496)
        x = x.permute(0, 2, 1)  # shape = (bs, 496, 768)

        x = torch.cat((self.spec_cls_token.expand(x.shape[0], -1, -1), x), dim=1)  # shape = (bs, 1+496, 768)
        # interplate pos embedding and add
        x = x + nn.functional.interpolate(self.spec_pos_embed.permute(0, 2, 1), x.shape[1], mode='linear').permute(0, 2,
                                                                                                                   1)
        return x, f_dim, t_dim

    def forward_rgb_features(self, x):
        B, no_of_frames, C, H, W = x.shape  # shape = (bs, no_of_frames, 3, 224, 224)
        x = torch.reshape(x, (B * no_of_frames, C, H, W))  # shape = (bs*no_of_frames, 3, 224, 224)
        x = self.rgb_conv(x)  # shape = (bs*no_of_frames, 768, 14, 14)

        _, dim, h, w = x.shape
        x = torch.reshape(x, (B, no_of_frames, dim, h, w))  # shape = (bs, no_of_frames, 768, 14, 14)
        x = x.permute(0, 2, 1, 3, 4)  # shape = (bs, 768, no_of_frames, 14, 14)
        x = torch.reshape(x, (B, dim, no_of_frames * h * w))  # shape = (bs, 768, no_of_frames*14*14) = (bs, 768, 1568)
        x = x.permute(0, 2, 1)  # shape = (bs, 1568, 768); 1568 = spatio-temporal tokens for 8 RGB images

        x = torch.cat((self.rgb_cls_token.expand(x.shape[0], -1, -1), x), dim=1)  # shape = (bs, 1+1568, 768)
        # interplate pos embedding and add
        x = x + nn.functional.interpolate(self.rgb_pos_embed.permute(0, 2, 1), x.shape[1], mode='linear').permute(0, 2,
                                                                                                                  1)
        return x, no_of_frames, h, w

    def forward_text_features(self, input_ids, attention_mask):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        pooling_idx = input_ids.to(torch.int).argmax(dim=-1)

        hidden_states = self.text_model.embeddings(input_ids=input_ids, position_ids=None)

        bsz, seq_len = input_shape
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self.text_model._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
            hidden_states.device
        )
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        return hidden_states, attention_mask, causal_attention_mask, pooling_idx

    def forward_encoder(self, x, y, z, shapes):

        # encoder forward pass

        attention_mask = z[1]
        causal_attention_mask = z[2]
        pooling_idx = z[3]

        for i in range(self.begin_layer):
            x = self.v1.blocks[i](x)
            y = self.v2.blocks[i](y)
            z = self.text_model.encoder.layers[i](
                z[0],
                attention_mask,
                causal_attention_mask,
            )
        z = (z[0], attention_mask, causal_attention_mask, pooling_idx)

        for j in range(self.begin_layer, 12):
            x, y, z = self.v2.blocks[j](x, y, z, shapes)

        z = z[0]

        x = self.spec_post_norm(x)
        y = self.rgb_post_norm(y)
        z = self.text_model.final_layer_norm(z)
        # return class token alone

        x = x[:, 0]
        y = y[:, 0]

        if self.pooling:
            z = z[torch.arange(z.shape[0], device=z.device), pooling_idx]
        else:
            z = z[:, 0]
        return x, y, z

    def forward(self, x, y, z, device):

        x, f_dim, t_dim = self.forward_spec_features(x)
        y, no_of_frames, rgb_h, rgb_w = self.forward_rgb_features(y)
        z = self.processor(text=z, return_tensors="pt", padding=True)
        z.to(device)
        z = self.forward_text_features(**z)
        x, y, z = self.forward_encoder(x, y, z, (f_dim, t_dim, no_of_frames, rgb_h, rgb_w))
        out = self.fusion_classification_head([x, y])
        a = x
        v = y

        return a, v, out, z
