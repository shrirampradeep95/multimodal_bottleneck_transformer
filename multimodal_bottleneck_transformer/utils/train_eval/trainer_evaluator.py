# Import relevant libraries
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message="User provided device_type of 'cuda', but CUDA is not available. Disabling")

from utils.train_eval.generate_plots import TrainingPlotter


class TrainerEvaluator:
    """
    Trainer class for audio/video/multimodal classification.
    - Uses BCEWithLogitsLoss + mAP for AudioSet (multi-label)
    - Uses CrossEntropyLoss + Top-1/Top-5 for VGGSound (single-label)
    """

    def __init__(self,  parameters, model, train_loader, val_loader, label_to_index, device='cuda', dataset='audioset'):
        self.parameters = parameters.copy()
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.label_to_index = label_to_index
        self.num_classes = len(label_to_index)
        self.device = device
        self.dataset = dataset.lower()

        if self.dataset == "vggsound":
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        self.plotter = TrainingPlotter()

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        loop = tqdm(self.train_loader, desc="Training", leave=False)

        for video, audio, labels in loop:
            video, audio = video.to(self.device), audio.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(video, audio)

            if self.dataset == "vggsound":
                loss = self.criterion(logits, labels.argmax(dim=1))
            else:
                loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(self.train_loader)
        print(f"Train Loss: {avg_loss:.4f}")
        return avg_loss

    def evaluate(self):
        self.model.eval()
        all_targets = []
        all_outputs = []

        loop = tqdm(self.val_loader, desc="Evaluating", leave=False)

        with torch.no_grad():
            for video, audio, labels in loop:
                video, audio = video.to(self.device), audio.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(video, audio)

                if self.dataset == "vggsound":
                    preds = torch.softmax(logits, dim=1)
                    all_targets.append(labels.argmax(dim=1).cpu())
                    all_outputs.append(preds.cpu())
                else:
                    probs = torch.sigmoid(logits)
                    all_targets.append(labels.cpu().numpy())
                    all_outputs.append(probs.cpu().numpy())

        if self.dataset == "vggsound":
            y_true = torch.cat(all_targets)
            y_pred = torch.cat(all_outputs)

            top1 = (y_pred.argmax(dim=1) == y_true).float().mean().item()
            top5 = sum([y_true[i] in y_pred[i].topk(5).indices for i in range(len(y_true))]) / len(y_true)

            print(f"Validation Top-1 Acc: {top1:.4f}, Top-5 Acc: {top5:.4f}")
            return top1, top5
        else:
            y_true = np.vstack(all_targets)
            y_pred = np.vstack(all_outputs)
            mAP = self._mean_average_precision(y_true, y_pred)
            print(f"Validation mAP: {mAP:.4f}")
            return mAP

    def _mean_average_precision(self, y_true, y_score):
        APs = []
        for i in range(self.num_classes):
            if np.sum(y_true[:, i]) > 0:
                ap = average_precision_score(y_true[:, i], y_score[:, i])
                APs.append(ap)
        return np.mean(APs) if APs else 0.0

    def train(self, epochs):
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}")
            train_loss = self.train_one_epoch()
            eval_score = self.evaluate()
            self.plotter.log(train_loss, eval_score)

        # Plot only after all epochs
        self.plotter.plot(
            self.parameters,
            dataset=self.dataset,
            model=self.model,
            test_dataset=self.val_loader.dataset,
            device=self.device
        )


