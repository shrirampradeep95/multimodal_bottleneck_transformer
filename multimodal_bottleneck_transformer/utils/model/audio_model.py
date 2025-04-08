import torch
import torch.nn as nn
from utils.model.ast.src.models.ast_models import ASTModel


class ASTEncoder(nn.Module):
    """
    ASTEncoder using torchaudio's built-in AST backbone.

    Loads a pretrained Audio Spectrogram Transformer (AST) from torchaudio.pipelines
    and removes the classification head to use it as a feature extractor.

    Input:
        x (torch.Tensor): (B, 1, 128, T) log-mel spectrograms

    Output:
        torch.Tensor: (B, N, D) token embeddings
    """
    def __init__(self, model_size='base224'):
        super().__init__()
        self.ast = ASTModel(
            label_dim=527,
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
