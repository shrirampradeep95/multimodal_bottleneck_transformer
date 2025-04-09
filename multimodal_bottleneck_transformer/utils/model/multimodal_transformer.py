# Import relevant libraries
import torch.nn as nn
from utils.model.video_model import VideoEncoder
from utils.model.audio_model import AudioEncoder
from utils.model.bottleneck_fusion import BottleneckEncoder


class AVModel(nn.Module):
    """
    Audio-Visual classification model using a shared bottleneck transformer architecture.
    This model fuses spectrogram (audio) and video (visual) token sequences through a sequence of
    transformer-based fusion blocks, then performs classification based on the fused representation.
    """

    def __init__(self, num_classes, num_bottlenecks, transformer_layers=12):
        """
        Initializes the AVModel with encoders, fusion layers, and final classifier.
        """
        super(AVModel, self).__init__()

        # Audio encoder
        self.audio_encoder = AudioEncoder(ast_weights_path='audioset_16_16_0.4422.pth')

        # Video encoder
        self.video_encoder = VideoEncoder()

        # Create a list of 12 fusion transformer layers, one per ViT block
        encoder_layers = []
        for i in range(transformer_layers):
            encoder_layers.append(
                BottleneckEncoder(
                    num_bottlenecks=num_bottlenecks,
                    spec_enc=self.audio_encoder.v1.blocks[i],
                    rgb_enc=self.video_encoder.v2.blocks[i]
                )
            )
        self.audio_visual_blocks = nn.Sequential(*encoder_layers)

        # Final normalization layers from ViT encoders
        self.spec_post_norm = self.audio_encoder.v1.norm
        self.rgb_post_norm = self.video_encoder.v2.norm

        # Final classifier: maps the fused 768-dim CLS token to class logits
        self.classifier = nn.Sequential(
            nn.Linear(768, num_classes)
        )

    def forward_spec_features(self, x):
        """
        Extract audio tokens from input spectrogram.
        Args:
            x: Spectrogram input
        Returns:
            Tensor: Audio token
        """
        return self.audio_encoder.forward_features(x)

    def forward_rgb_features(self, x):
        """
        Extract visual tokens from input video frames.
        Args:
            x: Video input
        Returns:
            Tensor: Visual token
        """
        return self.video_encoder.forward_features(x)

    def forward_encoder(self, spec_tokens, rgb_tokens):
        """
        Apply a sequence of bottleneck fusion transformer blocks.
        Args:
            spec_tokens: Audio token
            rgb_tokens: Visual token
        Returns:
            Tuple: Fused audio and visual CLS tokens
        """
        # Pass through transformer_layers bottleneck transformer layers
        for idx, blk in enumerate(self.audio_visual_blocks):
            spec_tokens, rgb_tokens = blk(spec_tokens, rgb_tokens, layer_idx=idx + 1)

        # Apply final normalization
        spec_tokens = self.spec_post_norm(spec_tokens)
        rgb_tokens = self.rgb_post_norm(rgb_tokens)

        # Extract CLS tokens
        spec_cls = spec_tokens[:, 0]
        rgb_cls = rgb_tokens[:, 0]
        return spec_cls, rgb_cls

    def forward(self, x, y):
        """
        Full forward pass for audiovisual classification.
        Args:
            x: Spectrogram input
            y: Video input
        Returns:
            Tensor: Class logits
        """
        # Extract tokens from both encoders
        spec_tokens = self.forward_spec_features(x)
        rgb_tokens = self.forward_rgb_features(y)

        # Fuse audio and visual streams
        spec_cls, rgb_cls = self.forward_encoder(spec_tokens, rgb_tokens)

        # Average fused CLS tokens
        fused = (spec_cls + rgb_cls) * 0.5

        # Predict class logits
        logits = self.classifier(fused)
        return logits


class AudioOnlyModel(nn.Module):
    """
    Audio only classification model using AST-based AudioEncoder.
    """
    def __init__(self, num_classes):
        """
        Initializes the audio only model.
        Args:
            num_classes: Number of output classes.
        """
        super(AudioOnlyModel, self).__init__()

        # Load AudioEncoder with AST weights
        self.audio_encoder = AudioEncoder(ast_weights_path='audioset_16_16_0.4422.pth')

        # Final normalization from audio encoder
        self.spec_post_norm = self.audio_encoder.v1.norm

        # Classifier from CLS token to logits
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        """
        Forward pass for audio-only input.
        Args:
            x: Spectrogram input
        Returns:
            Tensor: Class logits
        """
        spec_tokens = self.audio_encoder.forward_features(x)
        spec_tokens = self.spec_post_norm(spec_tokens)
        spec_cls = spec_tokens[:, 0]
        return self.classifier(spec_cls)


class VideoOnlyModel(nn.Module):
    """
    Video only classification model using ViT-based VideoEncoder.
    """

    def __init__(self, num_classes):
        """
        Initializes the video only model.
        Args:
            num_classes: Number of output classes.
        """
        super(VideoOnlyModel, self).__init__()

        # Load ViT for video frames
        self.video_encoder = VideoEncoder()

        # Final normalization from video encoder
        self.rgb_post_norm = self.video_encoder.v2.norm

        # Classifier from CLS token to logits
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, y):
        """
        Forward pass for video only input.
        Args:
            y: Video input
        Returns:
            Tensor: Class logits
        """
        rgb_tokens = self.video_encoder.forward_features(y)
        rgb_tokens = self.rgb_post_norm(rgb_tokens)
        rgb_cls = rgb_tokens[:, 0]
        return self.classifier(rgb_cls)
