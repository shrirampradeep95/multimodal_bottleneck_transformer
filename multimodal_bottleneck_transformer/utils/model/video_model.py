# Import relevant libraries
import torch
import torch.nn as nn
import timm
from einops import rearrange


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
