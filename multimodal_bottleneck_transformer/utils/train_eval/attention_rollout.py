import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2


class AttentionRolloutVisualizer:
    def __init__(self, model, dataset, save_dir="attention_maps", device="cuda"):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def compute_rollout_attention(self, attn_matrices):
        """
        Computes attention rollout from a list of attention matrices.
        Each attn matrix shape: (B, H, N, N)
        """
        result = torch.eye(attn_matrices[0].size(-1)).to(self.device)
        for attn in attn_matrices:
            attn_heads_fused = attn.mean(dim=1)  # average over heads
            result = attn_heads_fused @ result
        return result  # (B, N, N)

    def visualize(self, num_samples=3):
        self.model.eval()
        count = 0

        with torch.no_grad():
            for i in range(len(self.dataset)):
                video, audio, labels = self.dataset[i]
                video = video.unsqueeze(0).to(self.device)  # (1, T, 3, 224, 224)
                audio = audio.unsqueeze(0).to(self.device)

                # Forward pass to get attention maps
                vit = self.model.vision
                vit.eval()

                attn_weights = []

                def hook_fn(module, input, output):
                    attn_weights.append(module.attn_weights)  # (B, H, N, N)

                handles = [blk.attn.register_forward_hook(hook_fn) for blk in vit.blocks]
                _ = vit(video)
                for handle in handles:
                    handle.remove()

                rollout = self.compute_rollout_attention(attn_weights)  # (B, N, N)
                cam = rollout[:, 0, 1:]  # attention from CLS to all patches
                num_patches = cam.shape[-1]
                spatial_dim = int(num_patches ** 0.5)

                cam = cam.reshape(1, spatial_dim, spatial_dim)
                cam = F.interpolate(cam.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False)
                cam = cam.squeeze().cpu().numpy()
                cam = (cam - cam.min()) / (cam.max() - cam.min())

                middle_frame = video[0, video.shape[1] // 2].cpu().numpy().transpose(1, 2, 0)
                middle_frame = (middle_frame * 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(middle_frame, 0.5, heatmap, 0.5, 0)

                label_str = ", ".join([str(l) for l in labels.nonzero(as_tuple=True)[0].tolist()])

                fig, axs = plt.subplots(1, 2, figsize=(8, 4))
                axs[0].imshow(middle_frame)
                axs[0].set_title(f"Original\nLabels: {label_str}")
                axs[0].axis('off')
                axs[1].imshow(overlay)
                axs[1].set_title("Attention Rollout")
                axs[1].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(self.save_dir, f"sample_{i}.png"))
                plt.close()

                count += 1
                if count >= num_samples:
                    break
