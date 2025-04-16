# Import relevant libraries
from utils.model.ast.src.models.ast_models import ASTModel
import torch
import torch.nn as nn
import timm
from einops import rearrange


class ASTEncoder(nn.Module):
    """
    ASTEncoder using torchaudio built-in AST backbone.

    Loads a pretrained Audio Spectrogram Transformer (AST) from torchaudio.pipelines
    and removes the classification head to use it as a feature extractor.

    Input:
        x (torch.Tensor): (B, 1, 128, T) log-mel spectrograms

    Output:
        torch.Tensor: (B, N, D) token embeddings
    """
    def __init__(self, num_labels, model_size='base224'):
        super().__init__()
        self.ast = ASTModel(
            label_dim=num_labels,
            fstride=10, tstride=10,
            input_fdim=128, input_tdim=800,
            model_size=model_size
        )

        # Remove classification head
        self.ast.mlp_head = nn.Identity()

    def forward(self, x):
        """
        Forward pass through AST model.

        Args:
            x (torch.Tensor): Input spectrogram of shape (B, 1, 128, T)

        Returns:
            torch.Tensor: Token embeddings of shape (B, N, D)
        """
        if x.dim() == 4:
            return torch.stack([self.ast(sample) for sample in x])
        elif x.dim() == 3:
            return self.ast(x)
        else:
            raise ValueError(f"Unexpected input shape for AST: {x.shape}")


class ViTEncoder(nn.Module):
    """
    Vision Transformer (ViT) Encoder for extracting patch-level embeddings from video frames.

    This class wraps a pretrained ViT and processes sequences of video frames.
    It extracts token embeddings for each frame using patch embedding and transformer layers.

    Args:
        model_name (str): Name of the ViT model from the `timm` library. Default is 'vit_base_patch16_224'.
        pretrained (bool): Whether to load ImageNet-21K pretrained weights. Default is True.

    Input:
        x (torch.Tensor): A tensor of shape (B, T, 3, 224, 224) representing a batch of B video clips,
                          each with T RGB frames resized to 224×224.

    Output:
        torch.Tensor: A tensor of shape (B, T, N, D), where:
            - B is the batch size
            - T is the number of frames
            - N is the number of tokens per frame (including [CLS] if present)
            - D is the embedding dimension (typically 768 for ViT-Base)
    """
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super().__init__()
        # Load the full ViT model from timm
        self.vit = timm.create_model(model_name, pretrained=pretrained)

        # Extract the essential components of ViT
        # Converts 2D frame to patch tokens
        self.patch_embed = self.vit.patch_embed
        # Learnable [CLS] token
        self.cls_token = self.vit.cls_token
        # Positional embeddings
        self.pos_embed = self.vit.pos_embed
        # Transformer encoder layers
        self.blocks = self.vit.blocks
        # Final layer normalization
        self.norm = self.vit.norm

    def forward(self, x):
        # x shape: (B, T, 3, 224, 224)
        B, T, C, H, W = x.shape

        # Flatten time dimension into batch: (B*T, 3, 224, 224)
        x = rearrange(x, 'b t c h w -> (b t) c h w')

        # Apply patch embedding: (B*T, N_patches, D)
        x = self.patch_embed(x)

        # Expand [CLS] token to match batch size and prepend
        # (B*T, 1, D)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        # (B*T, 1 + N_patches, D)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embeddings
        x = x + self.pos_embed[:, :x.size(1)]

        # Pass through transformer encoder blocks
        for blk in self.blocks:
            x = blk(x)

        # Apply final layer normalization
        x = self.norm(x)

        # Reshape back to (B, T, N, D)
        x = rearrange(x, '(b t) n d -> b (t n) d', b=B, t=T)
        return x


class LatentCrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, query: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(query, source, source)
        query = self.norm1(query + attended)
        query = self.norm2(query + self.mlp(query))
        return query


class BottleneckFusionTransformer(nn.Module):
    def __init__(self, dim: int = 768, num_latents: int = 8, num_layers: int = 2):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, dim) * 0.02)
        self.audio_blocks = nn.ModuleList([
            LatentCrossAttentionBlock(dim) for _ in range(num_layers)
        ])
        self.video_blocks = nn.ModuleList([
            LatentCrossAttentionBlock(dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, audio_tokens: torch.Tensor, video_tokens: torch.Tensor) -> torch.Tensor:
        B = audio_tokens.shape[0]
        latents = self.latents.expand(B, -1, -1)
        for audio_blk, video_blk in zip(self.audio_blocks, self.video_blocks):
            latents = audio_blk(latents, audio_tokens)
            latents = video_blk(latents, video_tokens)
        return self.norm(latents)


class AVClassificationModel(nn.Module):
    def __init__(self, num_classes: int, num_latents: int = 8, num_layers: int = 2):
        super().__init__()
        self.audio_encoder = ASTEncoder(num_classes)
        self.video_encoder = ViTEncoder()
        self.fusion = BottleneckFusionTransformer(dim=768, num_latents=num_latents, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, audio_input: torch.Tensor, video_input: torch.Tensor) -> torch.Tensor:
        audio_tokens = self.audio_encoder.forward(audio_input)
        video_tokens = self.video_encoder.forward(video_input)
        latent_tokens = self.fusion(audio_tokens, video_tokens)
        pooled = latent_tokens.mean(dim=1)
        return self.classifier(pooled)
