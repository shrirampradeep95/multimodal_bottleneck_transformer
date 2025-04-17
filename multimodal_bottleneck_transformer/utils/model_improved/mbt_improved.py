# Fully Improved Multimodal Transformer with Bottleneck Tokens
# ============================================================================
# Improvements:
# - Extract token-level features from AST model
# - Added positional encoding to audio and video embeddings
# - Improved fusion with robust multi-head cross-attention via bottleneck tokens
# - Normalization and projection layers for better modality alignment

import torch
import torch.nn as nn
import timm
from utils.model.ast.src.models.ast_models import ASTModel
from einops import rearrange


class ASTEncoder(nn.Module):
    def __init__(self, model_size='base224'):
        super().__init__()
        self.ast = ASTModel(
            label_dim=527,  # Placeholder
            fstride=10, tstride=10,
            input_fdim=128, input_tdim=800,
            model_size=model_size
        )

    def forward(self, x):
        B, _, _, _ = x.shape
        x = self.ast.v.patch_embed(x)
        cls_tokens = self.ast.v.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.ast.v.pos_embed[:, :x.size(1), :]
        for blk in self.ast.v.blocks:
            x = blk(x)
        x = self.ast.v.norm(x)
        return x


class ViTEncoder(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained)
        self.patch_embed = self.vit.patch_embed
        self.cls_token = self.vit.cls_token
        self.pos_embed = self.vit.pos_embed
        self.blocks = self.vit.blocks
        self.norm = self.vit.norm

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed[:, :x.size(1)]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = rearrange(x, '(b t) n d -> b (t n) d', b=B, t=T)
        return x


class BottleneckCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, latents, tokens):
        attended, _ = self.attn(latents, tokens, tokens)
        latents = self.norm1(latents + attended)
        latents = self.norm2(latents + self.mlp(latents))
        return latents


class FusionTransformer(nn.Module):
    def __init__(self, dim=768, num_latents=16, num_layers=3):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_latents, dim))
        self.audio_proj = nn.Linear(dim, dim)
        self.video_proj = nn.Linear(dim, dim)

        self.audio_blocks = nn.ModuleList([BottleneckCrossAttention(dim) for _ in range(num_layers)])
        self.video_blocks = nn.ModuleList([BottleneckCrossAttention(dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, audio_tokens, video_tokens):
        B = audio_tokens.size(0)
        latents = self.latents.expand(B, -1, -1) + self.pos_embedding

        audio_tokens = self.audio_proj(audio_tokens)
        video_tokens = self.video_proj(video_tokens)

        for audio_blk, video_blk in zip(self.audio_blocks, self.video_blocks):
            latents = audio_blk(latents, audio_tokens)
            latents = video_blk(latents, video_tokens)

        return self.norm(latents)


class AVClassificationModel(nn.Module):
    def __init__(self, num_classes, num_latents=16, num_layers=3):
        super().__init__()
        self.audio_encoder = ASTEncoder()
        self.video_encoder = ViTEncoder()
        self.fusion = FusionTransformer(dim=768, num_latents=num_latents, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, audio_input, video_input):
        audio_tokens = self.audio_encoder(audio_input)
        video_tokens = self.video_encoder(video_input)
        latent_tokens = self.fusion(audio_tokens, video_tokens)
        pooled = latent_tokens.mean(dim=1)
        return self.classifier(pooled)
