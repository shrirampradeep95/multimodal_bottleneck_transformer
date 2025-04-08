# Import relevant libraries
import torch.nn as nn

# Import user defined libraries
from utils.model.audio_model import ASTEncoder
from utils.model.video_model import ViTEncoder
from utils.model.bottleneck_fusion import BottleneckFusion


class MultimodalTransformer(nn.Module):
    """
    Multimodal Transformer for Audio-Visual Classification using Bottleneck Fusion.

    This model processes video and audio inputs independently using modality-specific
    transformer encoders (ViT and AST), then fuses them through a shared bottleneck
    token mechanism. The fused representation is passed through a classifier head
    to predict class labels.

    Args:
        num_classes (int): Number of output classes for classification.
        modality (str): One of "audio", "video", or "av" (default)
    """

    def __init__(self, num_classes=100, modality="av"):
        super().__init__()

        self.modality = modality
        self.vision = ViTEncoder() if modality in ("video", "av") else None
        self.audio = ASTEncoder() if modality in ("audio", "av") else None
        self.fusion = BottleneckFusion() if modality == "av" else None

        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, num_classes)
        )

    def forward(self, video, audio):
        if self.modality == "video":
            V = self.vision(video)
            return self.classifier(V[:, 0])  # Use [CLS] token

        elif self.modality == "audio":
            A = self.audio(audio)
            return self.classifier(A[:, 0])

        elif self.modality == "av":
            V = self.vision(video)
            A = self.audio(audio)
            fused = self.fusion(V, A)
            return self.classifier(fused[:, 0])

        else:
            raise ValueError(f"Unsupported modality: {self.modality}")
