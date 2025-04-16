# Import relevant libraries
import torch
import torch.nn as nn


class BottleneckEncoder(nn.Module):
    def __init__(self, num_bottlenecks, spec_enc, rgb_enc, fusion_start_layer=8):
        """
        Initializes the Bottleneck Encoder that fuses audio and visual token representations
        using learnable latent tokens as a bottleneck for cross-modal interaction.
        """
        super(BottleneckEncoder, self).__init__()

        # Audio modality transformer components
        self.spec_norm1 = spec_enc.norm1
        self.spec_attn = spec_enc.attn
        self.spec_norm2 = spec_enc.norm2
        self.spec_mlp = spec_enc.mlp

        # Visual modality transformer components
        self.rgb_norm1 = rgb_enc.norm1
        self.rgb_attn = rgb_enc.attn
        self.rgb_norm2 = rgb_enc.norm2
        self.rgb_mlp = rgb_enc.mlp

        # Cross-modal latent bottleneck tokens
        self.num_latents = num_bottlenecks
        self.latents = nn.Parameter(torch.empty(1, num_bottlenecks, 768).normal_(std=0.02))

        # Learnable scaling parameters for controlling fusion strength - gating values
        self.scale_a = nn.Parameter(torch.ones(1) * 0.1)
        self.scale_v = nn.Parameter(torch.ones(1) * 0.1)

        # Fusion is only applied from this layer onward
        self.fusion_start_layer = fusion_start_layer

    @staticmethod
    def attention(q, k, v):
        """
        Computes scaled dot-product attention.
        Args:
            q: Query tensor of shape (B, N_q, C)
            k: Key tensor of shape (B, N_k, C)
            v: Value tensor of shape (B, N_k, C)

        Returns:
            Tensor: Attention output of shape (B, N_q, C)
        """
        B, N, C = q.shape
        scale = C ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        return out.reshape(B, N, C)

    def fusion(self, audio_tokens, visual_tokens):
        """
        Performs cross-modal fusion using latent bottleneck tokens in two stages:
        1. Computes temporary bottleneck tokens from each modality via cross-attention.
        2. Averages them to produce shared fusion tokens, then updates each modality.
        Args:
            audio_tokens: Audio tokens of shape (B, N_audio, 768)
            visual_tokens: Visual tokens of shape (B, N_visual, 768)
        Returns:
            Tuple: Updated audio and visual tokens.
        """
        BS = audio_tokens.shape[0]

        # Expand the latent tokens for each sample in the batch
        bottleneck_tokens = self.latents.expand(BS, -1, -1)

        # Stage 1: Generate temporary bottleneck tokens from each modality
        temp_bottleneck_audio = self.attention(q=bottleneck_tokens, k=audio_tokens, v=audio_tokens)
        temp_bottleneck_visual = self.attention(q=bottleneck_tokens, k=visual_tokens, v=visual_tokens)

        # Stage 2: Average both bottlenecks to create the fused bottleneck
        fusion_tokens = (temp_bottleneck_audio + temp_bottleneck_visual) / 2.0

        # Use fusion tokens to update each modality using cross-attention
        audio_tokens = audio_tokens + self.scale_a * self.attention(q=audio_tokens, k=fusion_tokens, v=fusion_tokens)
        visual_tokens = visual_tokens + self.scale_v * self.attention(q=visual_tokens, k=fusion_tokens, v=fusion_tokens)

        return audio_tokens, visual_tokens

    def forward(self, x, y, layer_idx):
        """
        Applies bottleneck fusion and transformer block updates.
        Args:
            x: Audio tokens of shape (B, N_audio, 768)
            y: Visual tokens of shape (B, N_visual, 768)
            layer_idx: Current transformer layer index.
        Returns:
            Tuple: Updated audio and visual token embeddings.
        """
        # Perform fusion only if this layer is after fusion start layer
        # if layer_idx >= self.fusion_start_layer:
        x, y = self.fusion(x, y)

        # Apply self-attention with residual connection
        x = x + self.spec_attn(self.spec_norm1(x))
        y = y + self.rgb_attn(self.rgb_norm1(y))

        # Apply MLP with residual connection
        x = x + self.spec_mlp(self.spec_norm2(x))
        y = y + self.rgb_mlp(self.rgb_norm2(y))

        return x, y
