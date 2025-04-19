# Import relevant libraries
import torch
import torch.nn as nn
import timm
from utils.model.ast.src.models.ast_models import ASTModel
from einops import rearrange


class ASTEncoder(nn.Module):
    """
    Audio encoder using the Audio Spectrogram Transformer (AST) backbone
    Adapts ASTModel to extract token-level features from log-mel spectrogram
    """
    def __init__(self, model_size='base224'):
        """
        Initializes AST model for audio encoding.
        Args:
            model_size: Model variant (default 'base224')
        """
        super().__init__()
        self.ast = ASTModel(
            label_dim=527,
            fstride=10, tstride=10,
            input_fdim=128, input_tdim=800,
            model_size=model_size
        )

    def forward(self, x):
        """
        Forward pass for extracting audio tokens from log-mel spectrogram.
        Returns:
            Tensor: Audio tokens
        """
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
    """
    Video encoder using Vision Transformer (ViT) backbone.
    Converts a sequence of video frames into a sequence of patch tokens.
    """
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        """
        Initializes the ViT model for video encoding.
        Args:
            model_name: ViT model variant
            pretrained: If True, loads pretrained weights
        """
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained)
        self.patch_embed = self.vit.patch_embed
        self.cls_token = self.vit.cls_token
        self.pos_embed = self.vit.pos_embed
        self.blocks = self.vit.blocks
        self.norm = self.vit.norm

    def forward(self, x):
        """
        Forward pass for encoding a batch of video clips.
        Returns:
            Tensor: Video tokens
        """
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
    """
    Multi-head cross-attention block used for fusing tokens with bottleneck latents.
    """
    def __init__(self, dim, num_heads=12):
        """
        Args:
            dim: Embedding dimension
            num_heads: Number of attention heads
        """
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
        """
        Forward pass for cross-attention between latent and modality tokens.
        Args:
            latents: Latent bottlenecks
            tokens: Modality tokens
        Returns:
            Updated latent tokens
        """
        attended, _ = self.attn(latents, tokens, tokens)
        latents = self.norm1(latents + attended)
        latents = self.norm2(latents + self.mlp(latents))
        return latents


class FusionTransformer(nn.Module):
    """
    Stacks multiple BottleneckCrossAttention blocks for iterative cross-modal fusion.
    """
    def __init__(self, dim=768, num_latents=16, num_layers=3):
        """
        Args:
            dim: Embedding dimension
            num_latents: Number of bottleneck latent tokens
            num_layers: Number of fusion layers
        """
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_latents, dim))
        self.audio_proj = nn.Linear(dim, dim)
        self.video_proj = nn.Linear(dim, dim)

        self.audio_blocks = nn.ModuleList([BottleneckCrossAttention(dim) for _ in range(num_layers)])
        self.video_blocks = nn.ModuleList([BottleneckCrossAttention(dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, audio_tokens, video_tokens):
        """
        Fuses audio and video tokens using stacked bottleneck blocks.
        Returns:
            Pooled latent representation
        """
        B = audio_tokens.size(0)
        latents = self.latents.expand(B, -1, -1) + self.pos_embedding

        # Project modalities to common space
        audio_tokens = self.audio_proj(audio_tokens)
        video_tokens = self.video_proj(video_tokens)

        # Iterative cross-attention updates
        for audio_blk, video_blk in zip(self.audio_blocks, self.video_blocks):
            latents = audio_blk(latents, audio_tokens)
            latents = video_blk(latents, video_tokens)

        return self.norm(latents)


class AVClassificationModel(nn.Module):
    """
    Full Audio-Visual Classification Model using bottleneck transformer fusion.
    Combines AST audio encoder, ViT video encoder, fusion layers, and a classifier head.
    """
    def __init__(self, num_classes, num_latents=16, num_layers=3):
        """
        Args:
            num_classes: Number of target classes
            num_latents: Number of bottleneck tokens
            num_layers: Number of fusion layers
        """
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
        """
        Forward pass for multimodal classification.
        Args:
            audio_input: log-mel spectrogram
            video_input: video frames
        Returns:
            Class logits
        """
        audio_tokens = self.audio_encoder(audio_input)
        video_tokens = self.video_encoder(video_input)
        latent_tokens = self.fusion(audio_tokens, video_tokens)
        pooled = latent_tokens.mean(dim=1)
        return self.classifier(pooled)