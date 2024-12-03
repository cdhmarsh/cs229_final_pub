"""Use the trained soft_label_predictor model to generate artificial soft labels for the training data."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os
from soft_label_predictor import CIFAR10SoftLabelDataset, ImageHardLabelToSoftLabelModel

class HardToSoftLabelDataset(Dataset):
    def __init__(self, model, dataloader, device):
        """
        :param model: Trained model to predict soft labels.
        :param dataloader: DataLoader of images and hard labels.
        :param device: Device to run the predictions on (CPU or GPU).
        """
        self.images = []
        self.soft_labels = []
        self.device = device
        
        model.eval()  # Ensure the model is in evaluation mode
        
        with torch.no_grad():
            for images, hard_labels in dataloader:
                images = images.to(device)
                hard_labels = F.one_hot(hard_labels, num_classes=10).float().to(device)
                
                # Predict soft labels
                predicted_soft_labels = model(images, hard_labels)
                
                self.images.extend(images.cpu())  # Store images
                self.soft_labels.extend(predicted_soft_labels.cpu())  # Store soft labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.soft_labels[idx]


def create_soft_label_dataloader(model, hard_label_dataloader, batch_size, device):
    """
    Convert a DataLoader of hard labels and images to a DataLoader of soft labels and images.
    :param model: Trained model to predict soft labels.
    :param hard_label_dataloader: DataLoader with images and hard labels.
    :param batch_size: Batch size for the new DataLoader.
    :param device: Device to run predictions on.
    :return: DataLoader with images and predicted soft labels.
    """
    soft_label_dataset = HardToSoftLabelDataset(model, hard_label_dataloader, device)
    return DataLoader(soft_label_dataset, batch_size=batch_size, shuffle=True)

# Example usage:

device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using device: {device}")

# Load the trained model
model = ImageHardLabelToSoftLabelModel().to(device)
model.load_state_dict(torch.load("models/best_model.pt"))
model.eval()

# Load CIFAR-10 dataset and return train, validation, and test DataLoaders
def load_cifar10():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ]
    )
    full_dataset = datasets.CIFAR10(root="../data/cifar-10", train=True, download=True, transform=transform)
    # we use the test dataset for training, similar to the CIFAR-10H experiment
    train_dataset = datasets.CIFAR10(root="../data/cifar-10", train=False, download=True, transform=transform)

    # This dataset will be used for testing and validation.
    #   30% of the data will be used for validation, and 70% for testing.
    test_size = int(0.7 * len(full_dataset))
    val_size = len(full_dataset) - test_size
    test_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [test_size, val_size], generator=torch.Generator().manual_seed(229)
    )

    return train_dataset, test_dataset, val_dataset



cifar10_train_dataset, cifar10_test_dataset, cifar10_val_dataset = load_cifar10() 
cifar10_train_loader = DataLoader(cifar10_train_dataset, batch_size=128, shuffle=True)
cifar10_test_loader = DataLoader(cifar10_test_dataset, batch_size=128, shuffle=False)
cifar10_val_loader = DataLoader(cifar10_val_dataset, batch_size=128, shuffle=False)
print(
    f"CIFAR-10 dataset loaded with {len(cifar10_train_dataset)} training, {len(cifar10_test_dataset)} test, and {len(cifar10_val_dataset)} validation samples"
)

# Convert to DataLoader with predicted soft labels
soft_label_dataloader = create_soft_label_dataloader(model, cifar10_test_loader, batch_size=128, device=device)

# Iterate through the new DataLoader
for images, soft_labels in soft_label_dataloader:
    print(images.shape, soft_labels.shape)
