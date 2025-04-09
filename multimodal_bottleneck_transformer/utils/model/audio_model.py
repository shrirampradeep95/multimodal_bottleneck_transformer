# Import relevant libraries
import torch
import torch.nn as nn
import timm


class AudioEncoder(nn.Module):
    def __init__(self, ast_weights_path='audioset_16_16_0.4422.pth'):
        """
        Initializes the audio encoder using a Vision Transformer backbone adapted for spectrogram input.
        Loads pretrained AST  weights for audio representation learning.

        Args:
            ast_weights_path: Path to the pretrained AST weights
        """
        super(AudioEncoder, self).__init__()

        # Load a base Vision Transformer (ViT) model with ImageNet21k pretrained weights
        self.v1 = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)

        # Remove the classification head and pre-logits layers to extract only feature representations
        self.v1.pre_logits = nn.Identity()
        self.v1.head = nn.Identity()

        # Modify the patch embedding layer to accept single-channel input
        self.v1.patch_embed.proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))

        # Reinitialize the positional embedding to match AST structure
        self.v1.pos_embed = nn.Parameter(torch.randn(1, 514, 768) * 0.02)

        # Load the AST pretrained weights
        ast_pretrained_weight = torch.load(ast_weights_path)
        v = self.v1.state_dict()

        # Manually map the AST checkpoint parameters to the ViT model
        v['cls_token'] = ast_pretrained_weight['module.v.cls_token']
        v['pos_embed'] = ast_pretrained_weight['module.v.pos_embed']
        v['patch_embed.proj.weight'] = ast_pretrained_weight['module.v.patch_embed.proj.weight']
        v['patch_embed.proj.bias'] = ast_pretrained_weight['module.v.patch_embed.proj.bias']

        # Load all transformer block weights
        for i in range(12):
            v[f'blocks.{i}.norm1.weight'] = ast_pretrained_weight[f'module.v.blocks.{i}.norm1.weight']
            v[f'blocks.{i}.norm1.bias'] = ast_pretrained_weight[f'module.v.blocks.{i}.norm1.bias']
            v[f'blocks.{i}.attn.qkv.weight'] = ast_pretrained_weight[f'module.v.blocks.{i}.attn.qkv.weight']
            v[f'blocks.{i}.attn.qkv.bias'] = ast_pretrained_weight[f'module.v.blocks.{i}.attn.qkv.bias']
            v[f'blocks.{i}.attn.proj.weight'] = ast_pretrained_weight[f'module.v.blocks.{i}.attn.proj.weight']
            v[f'blocks.{i}.attn.proj.bias'] = ast_pretrained_weight[f'module.v.blocks.{i}.attn.proj.bias']
            v[f'blocks.{i}.norm2.weight'] = ast_pretrained_weight[f'module.v.blocks.{i}.norm2.weight']
            v[f'blocks.{i}.norm2.bias'] = ast_pretrained_weight[f'module.v.blocks.{i}.norm2.bias']
            v[f'blocks.{i}.mlp.fc1.weight'] = ast_pretrained_weight[f'module.v.blocks.{i}.mlp.fc1.weight']
            v[f'blocks.{i}.mlp.fc1.bias'] = ast_pretrained_weight[f'module.v.blocks.{i}.mlp.fc1.bias']
            v[f'blocks.{i}.mlp.fc2.weight'] = ast_pretrained_weight[f'module.v.blocks.{i}.mlp.fc2.weight']
            v[f'blocks.{i}.mlp.fc2.bias'] = ast_pretrained_weight[f'module.v.blocks.{i}.mlp.fc2.bias']

        # Load the final layer norm parameters
        v['norm.weight'] = ast_pretrained_weight['module.v.norm.weight']
        v['norm.bias'] = ast_pretrained_weight['module.v.norm.bias']

        # Load updated state dict into the model
        self.v1.load_state_dict(v)

        # Freeze positional embedding and patch projection layer parameters to avoid training them
        self.v1.pos_embed.requires_grad = False
        for p in self.v1.patch_embed.proj.parameters():
            p.requires_grad = False
        for p in self.v1.blocks.parameters():
            p.requires_grad = False

        # Save components for later use in forward pass
        self.spec_conv = self.v1.patch_embed.proj
        self.spec_pos_embed = self.v1.pos_embed
        self.spec_cls_token = self.v1.cls_token

    def forward_features(self, x):
        """
        Computes token-level features from a spectrogram input using the adapted ViT model.

        Args:
            x: Input spectrogram tensor
        Returns:
            Tensor: Output token sequence
        """
        # Apply convolution to extract patch embeddings
        x = self.spec_conv(x)
        B, dim, f_dim, t_dim = x.shape

        # Flatten spatial dimensions into token sequence
        x = x.reshape(B, dim, f_dim * t_dim).permute(0, 2, 1)

        # Prepend CLS token to each sample in the batch
        cls_token = self.spec_cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Interpolate positional embeddings to match sequence length
        pos_embed = nn.functional.interpolate(
            self.spec_pos_embed.permute(0, 2, 1), x.shape[1], mode='linear'
        ).permute(0, 2, 1)

        # Add positional embeddings to tokens
        x = x + pos_embed

        return x
