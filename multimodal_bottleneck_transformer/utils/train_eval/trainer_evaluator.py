# Import relevant libraries
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
import numpy as np
from tqdm import tqdm
import warnings

# Import user defined libraries
from utils.train_eval.generate_plots import TrainingPlotter

warnings.filterwarnings("ignore", message="User provided device_type of 'cuda', but CUDA is not available. Disabling")


class TrainerEvaluator:
    """
    A general trainer and evaluator class for training multimodal models on audio/video datasets.
    - For AudioSet: Uses BCEWithLogitsLoss and evaluates with mean Average Precision (mAP).
    - For VGGSound: Uses CrossEntropyLoss and evaluates with Top-1 and Top-5 accuracy.
    """

    def __init__(self,
                 parameters, model, train_loader, val_loader, label_to_index, device='cuda', dataset='AudioSet',
                 vgg_sound_lr=0.01, audio_set_lr=1e-4
                 ):
        """
        Initializes the trainer/evaluator.
        Args:
            parameters: Training configuration dictionary.
            model: Audio/video/multimodal model to be trained.
            train_loader: DataLoader for training set.
            val_loader: DataLoader for validation set.
            label_to_index: Mapping from label names to indices.
            device: 'cuda' or 'cpu'.
            dataset: Name of the dataset, either 'AudioSet' or 'vggsound'.
            vgg_sound_lr: Learning rate for vggsound
            audio_set_lr: Learning rate for AudioSet
        """
        self.parameters = parameters.copy()
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.label_to_index = label_to_index
        self.num_classes = len(label_to_index)
        self.device = device
        self.dataset = dataset.lower()

        # Set optimizer and loss based on dataset type
        if self.dataset == "vggsound":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=vgg_sound_lr)
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=audio_set_lr)
            self.criterion = nn.BCEWithLogitsLoss()

        self.plotter = TrainingPlotter()

    def train_one_epoch(self):
        """
        Trains the model for one epoch over the training set.
        Returns:
            float: Average loss over the epoch.
        """
        self.model.train()
        running_loss = 0.0
        epoch_loss = []

        loop = tqdm(self.train_loader, desc="Training", leave=False)

        for video, audio, labels in loop:
            video, audio, labels = video.to(self.device), audio.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            if self.parameters['modality'] == 'av':
                logits = self.model(audio, video)
            elif self.parameters['modality'] == 'v':
                logits = self.model(video)
            else:
                logits = self.model(audio)

            # Compute loss based on dataset type
            if self.dataset == "vggsound":
                loss = self.criterion(logits, labels)
            else:
                loss = self.criterion(logits, labels)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            epoch_loss.append(loss.item())
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(self.train_loader)
        print(f"Train Loss: {avg_loss:.4f}")
        return avg_loss

    def evaluate(self):
        """
        Evaluates the model on the validation set.
        Returns:
            float or tuple: mAP (AudioSet) or (Top-1 Acc, Top-5 Acc) for VGGSound.
        """
        self.model.eval()
        all_targets = []
        all_outputs = []

        loop = tqdm(self.val_loader, desc="Evaluating", leave=False)

        with torch.no_grad():
            for video, audio, labels in loop:
                video, audio, labels = video.to(self.device), audio.to(self.device), labels.to(self.device)

                if self.parameters['modality'] == 'av':
                    logits = self.model(audio, video)
                elif self.parameters['modality'] == 'v':
                    logits = self.model(video)
                else:
                    logits = self.model(audio)

                if self.dataset == "vggsound":
                    preds = torch.softmax(logits, dim=1)
                    all_targets.append(labels)
                    all_outputs.append(preds.cpu())
                else:
                    probs = torch.sigmoid(logits)
                    all_targets.append(labels.cpu().numpy())
                    all_outputs.append(probs.cpu().numpy())

        if self.dataset == "vggsound":
            # Compute Top-1 and Top-5 accuracy
            y_true = torch.cat(all_targets)
            y_pred = torch.cat(all_outputs)

            # Make sure y_true is on the same device as y_pred
            y_true = y_true.to(y_pred.device)

            # Top-1 accuracy
            top1 = (y_pred.argmax(dim=1) == y_true).float().mean().item()

            k = min(5, y_pred.shape[1])
            top5 = sum([
                y_true[i].item() in y_pred[i].topk(k).indices.tolist()
                for i in range(len(y_true))
            ]) / len(y_true)

            print(f"Validation Top-1 Acc: {top1:.4f}, Top-5 Acc: {top5:.4f}")
            return top1, top5
        else:
            # Compute mean Average Precision (mAP)
            y_true = np.vstack(all_targets)
            y_pred = np.vstack(all_outputs)
            mAP = self._mean_average_precision(y_true, y_pred)

            print(f"Validation mAP: {mAP:.4f}")
            return mAP

    def _mean_average_precision(self, y_true, y_score):
        """
        Calculates the mean Average Precision (mAP) across all classes.
        Args:
            y_true: Ground-truth binary labels (N x C)
            y_score: Predicted probabilities (N x C)
        Returns:
            float: Mean average precision across all valid classes.
        """
        APs = []
        for i in range(self.num_classes):
            if np.sum(y_true[:, i]) > 0:
                ap = average_precision_score(y_true[:, i], y_score[:, i])
                APs.append(ap)
        return np.mean(APs) if APs else 0.0

    def train(self, epochs):
        """
        Trains the model for a given number of epochs and plots results.
        Args:
            epochs: Number of training epochs.
        """
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}")
            train_loss = self.train_one_epoch()
            eval_score = self.evaluate()

            self.plotter.log(train_loss, eval_score)

        # Final visualization
        self.plotter.plot(
            self.parameters,
            dataset=self.dataset
        )
