# Import relevant libraries
import torch
import torch.nn as nn
import timm


class VideoEncoder(nn.Module):
    def __init__(self):
        """
        Initializes the video encoder using a Vision Transformer backbone.
        This encoder is designed to process RGB video frames and convert them into token sequences.
        """
        super(VideoEncoder, self).__init__()

        # Create a ViT model pretrained on ImageNet21k
        self.v2 = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)

        # Remove classification layers to retain feature embeddings only
        self.v2.pre_logits = nn.Identity()
        self.v2.head = nn.Identity()

        # Freeze most of the model's parameters to use it as a feature extractor
        self.v2.pos_embed.requires_grad = False
        for p in self.v2.patch_embed.proj.parameters():
            p.requires_grad = False
        for p in self.v2.blocks.parameters():
            p.requires_grad = True

        # Save important components for use in forward pass
        self.rgb_conv = self.v2.patch_embed.proj
        self.rgb_pos_embed = self.v2.pos_embed
        self.rgb_cls_token = self.v2.cls_token

    def forward_features(self, x):
        """
        Processes a batch of RGB video frames and returns token representations.

        Args:
            x: A 5D input tensor of shape (B, num_frames, 3, H, W), where:
                - B is the batch size
                - num_frames is the number of frames per video
                - 3 is the RGB channels
                - H, W are the height and width of the frames

        Returns:
            Tensor: Output token sequence of shape
        """
        B, num_frames, C, H, W = x.shape

        # Merge the batch and frame dimensions to treat each frame as an individual image
        x = x.reshape(B * num_frames, C, H, W)

        # Apply patch embedding via convolution
        x = self.rgb_conv(x)
        _, dim, h, w = x.shape

        # Reshape back to include frame dimension
        x = x.reshape(B, num_frames, dim, h, w)

        # Move channel dimension to the front
        x = x.permute(0, 2, 1, 3, 4)

        # Flatten spatial and temporal dimensions into a token sequence
        x = x.reshape(B, dim, num_frames * h * w)
        x = x.permute(0, 2, 1)

        # Prepend a learned classification (CLS) token
        cls_token = self.rgb_cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Interpolate and add positional embeddings to the tokens
        pos_embed = nn.functional.interpolate(
            self.rgb_pos_embed.permute(0, 2, 1), x.shape[1], mode='linear'
        ).permute(0, 2, 1)
        x = x + pos_embed

        return x
