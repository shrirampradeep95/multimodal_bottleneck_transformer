# Import relevant libraries
import torch
import torch.nn as nn


class BottleneckFusion(nn.Module):
    """
    BottleneckFusion performs attention-based fusion of two modalities
    (e.g., vision and audio tokens) via a shared set of bottleneck tokens.

    This module follows the architecture proposed in the "Attention Bottlenecks
    for Multimodal Fusion" paper, where bottleneck tokens allow for controlled
    information exchange between modalities.

    Args:
        d_model (int): Dimensionality of input and bottleneck tokens (default: 768)
        n_bottlenecks (int): Number of bottleneck tokens to insert (default: 4)
        n_heads (int): Number of attention heads in each Transformer layer (default: 12)
        depth (int): Number of TransformerEncoderLayer blocks (default: 2)

    Input:
        vision_tokens (Tensor): Shape (B, N1, D) - token embeddings from vision modality
        audio_tokens  (Tensor): Shape (B, N2, D) - token embeddings from audio modality

    Output:
        fused_tokens  (Tensor): Shape (B, N1 + N2 + n_bottlenecks, D) - fused representation
    """

    def __init__(self, d_model=768, n_bottlenecks=4, n_heads=12, depth=2):
        super().__init__()

        # Learnable bottleneck tokens (shared across all samples)
        self.bottlenecks = nn.Parameter(torch.randn(1, n_bottlenecks, d_model) * 0.02)

        # Stack of TransformerEncoderLayers for fusion
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=0.1,
                batch_first=True
            ) for _ in range(depth)
        ])

    def forward(self, vision_tokens, audio_tokens):
        B = vision_tokens.size(0)  # Batch size
        V, A = vision_tokens, audio_tokens

        # Repeat bottlenecks for each sample in the batch
        B_tokens = self.bottlenecks.expand(B, -1, -1)  # (B, Bn, D)

        # Concatenate V + B + A along token dimension
        x = torch.cat([V, B_tokens, A], dim=1)  # (B, N1 + Bn + N2, D)

        # Create attention mask to prevent bottlenecks from attending to each other
        seq_len = x.size(1)
        Bn = B_tokens.size(1)
        N1 = V.size(1)

        attn_mask = torch.zeros((seq_len, seq_len), device=x.device)  # (S, S)
        b_start = N1
        b_end = N1 + Bn

        # Block bottleneck-to-bottleneck attention
        attn_mask[b_start:b_end, b_start:b_end] = float('-inf')

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, src_mask=attn_mask)

        return x
