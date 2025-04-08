# Import relevant libraries
import numpy as np
import torch


def mixup_data(x1, x2, y, alpha=0.3):
    """
    Applies Mixup augmentation to two modalities (x1 and x2) and their corresponding labels.

    Mixup is a data augmentation technique that creates virtual training examples by linearly
    interpolating between random pairs of samples and their labels.

    Args:
        x1 (Tensor): Input modality 1 (e.g., video tensor) of shape (B, ...)
        x2 (Tensor): Input modality 2 (e.g., audio tensor) of shape (B, ...)
        y (Tensor): Class labels (e.g., integer indices or one-hot vectors)
        alpha (float): Mixup interpolation hyperparameter (Beta distribution)

    Returns:
        mixed_x1 (Tensor): Mixed input for modality 1
        mixed_x2 (Tensor): Mixed input for modality 2
        y_a (Tensor): Original labels
        y_b (Tensor): Shuffled labels
        lam (float): Interpolation factor used
    """
    lam = np.random.beta(alpha, alpha)  # Sample lambda from Beta distribution
    batch_size = x1.size(0)
    index = torch.randperm(batch_size)  # Shuffle the batch

    # Linearly interpolate inputs and labels
    mixed_x1 = lam * x1 + (1 - lam) * x1[index]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index]
    y_a, y_b = y, y[index]

    return mixed_x1, mixed_x2, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Computes the mixup loss by interpolating between two losses.

    Args:
        criterion: A loss function (e.g., nn.CrossEntropyLoss)
        pred (Tensor): Model predictions
        y_a (Tensor): Original labels
        y_b (Tensor): Shuffled labels
        lam (float): Interpolation factor used during mixup

    Returns:
        Tensor: Interpolated loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
