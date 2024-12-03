"""
CIFAR-10H captures human uncertainty in the form of probability distributions over classes (soft labels).

CIFAR-10 has hard labels (one-hot encoded).

The objective is to train a model or algorithm that, given a CIFAR-10 image and its hard label, predicts a soft label that reflects human uncertainty patterns from CIFAR-10H.

Step 1: Match CIFAR-10 Images to CIFAR-10H Labels
- Ensure you align images from CIFAR-10 that also have annotations in CIFAR-10H.
- Create a dataset where each sample consists of:
    - CIFAR-10 image as input.
    - CIFAR-10 hard label as auxiliary information.
    - CIFAR-10H soft label as the target output.
    
Step 2: Model Architecture
- Train a model to learn the mapping between CIFAR-10 hard labels and CIFAR-10H soft labels. The model could use:
    - Input: CIFAR-10 hard label.
    - Output: Predicted soft label (a probability distribution over 10 classes).
    
Step 3: Training
- Loss Function:
    - Use KL divergence or mean squared error (MSE) to minimize the difference between predicted and actual soft labels from CIFAR-10H.
- Data Augmentation:
    - Train the model on CIFAR-10 images without CIFAR-10H annotations by:
        - Treating hard labels as smoothed soft labels (e.g., label smoothing with [0.9, 0.01, ..., 0.01]).
        - Introducing additional noise or perturbations to simulate ambiguity.
        
Step 4: Generating Soft Labels
- Once trained, use the model to predict soft labels for all CIFAR-10 images.
- These predicted soft labels should mimic the patterns of uncertainty observed in CIFAR-10H, effectively augmenting CIFAR-10 with more realistic annotations.

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import os
from pathlib import Path
from typing import Tuple


class Config:
    """Configuration for training and model parameters."""

    # Data parameters
    data_dir = Path("../data")
    batch_size = 256
    val_split = 0.2

    # Model parameters
    hidden_dims = [128, 64]
    dropout_rate = 0.2

    # Training parameters
    learning_rate = 0.001
    weight_decay = 1e-4
    epochs = 50
    early_stopping_patience = 10

    # Output paths
    model_dir = Path("models")
    model_path = model_dir / "soft_label_model.pt"
    output_dir = Path("outputs")
    soft_labels_path = output_dir / "cifar10_soft_labels.npy"


class CIFAR10SoftLabelDataset(Dataset):
    def __init__(self, cifar10_dataset: Dataset, soft_labels: np.ndarray):
        self.cifar10_dataset = cifar10_dataset
        self.soft_labels = torch.FloatTensor(soft_labels)

    def __len__(self) -> int:
        return len(self.cifar10_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        _, hard_label = self.cifar10_dataset[idx]
        soft_label = self.soft_labels[idx]
        return F.one_hot(torch.tensor(hard_label), num_classes=10).float(), soft_label


class HardToSoftLabelModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        layers = []
        input_dim = 10

        for hidden_dim in config.hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(config.dropout_rate),
                ]
            )
            input_dim = hidden_dim

        layers.extend([nn.Linear(input_dim, 10), nn.Softmax(dim=1)])

        self.mapper = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mapper(x)


def load_cifar10h(config: Config) -> Tuple[Dataset, Dataset]:
    """Load and prepare CIFAR-10H dataset."""
    cifar10h_probs_path = config.data_dir / "cifar-10h/cifar10h-probs.npy"
    if not cifar10h_probs_path.exists():
        raise FileNotFoundError(f"Soft labels not found at {cifar10h_probs_path}")

    cifar10h_probs = np.load(cifar10h_probs_path).astype(np.float32)
    cifar10_test = datasets.CIFAR10(
        root=config.data_dir / "cifar-10", train=False, download=True, transform=transforms.ToTensor()
    )

    full_dataset = CIFAR10SoftLabelDataset(cifar10_test, cifar10h_probs)
    train_size = int((1 - config.val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size

    return random_split(full_dataset, [train_size, val_size])


def train_model(
    model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, config: Config, device: torch.device
) -> nn.Module:
    """Train the model and return the best version."""
    criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.epochs):
        # Training
        model.train()
        train_loss = 0
        for hard_labels, soft_labels in train_loader:
            hard_labels, soft_labels = hard_labels.to(device), soft_labels.to(device)
            optimizer.zero_grad()
            predictions = model(hard_labels)
            loss = criterion(predictions.log(), soft_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for hard_labels, soft_labels in val_loader:
                hard_labels, soft_labels = hard_labels.to(device), soft_labels.to(device)
                predictions = model(hard_labels)
                val_loss += criterion(predictions.log(), soft_labels).item()
        val_loss /= len(val_loader)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1}/{config.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

    # Load best model
    model.load_state_dict(torch.load(config.model_path, weights_only=True))
    return model


def generate_soft_labels(model: nn.Module, config: Config, device: torch.device):
    """Generate soft labels for the entire CIFAR-10 dataset."""
    # Load CIFAR-10 training and test sets
    cifar10_train = datasets.CIFAR10(root=config.data_dir / "cifar-10", train=True, download=True)
    cifar10_test = datasets.CIFAR10(root=config.data_dir / "cifar-10", train=False, download=True)

    model.eval()
    soft_labels = []

    with torch.no_grad():
        # Process training set
        for _, label in cifar10_train:
            hard_label = F.one_hot(torch.tensor(label), num_classes=10).float().to(device)
            soft_label = model(hard_label.unsqueeze(0)).squeeze(0)
            soft_labels.append(soft_label.cpu().numpy())

        # Process test set
        for _, label in cifar10_test:
            hard_label = F.one_hot(torch.tensor(label), num_classes=10).float().to(device)
            soft_label = model(hard_label.unsqueeze(0)).squeeze(0)
            soft_labels.append(soft_label.cpu().numpy())

    # Save soft labels
    config.output_dir.mkdir(exist_ok=True)
    soft_labels = np.array(soft_labels)
    np.save(config.soft_labels_path, soft_labels)
    print(f"Saved soft labels to {config.soft_labels_path}")


def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create necessary directories
    config.model_dir.mkdir(exist_ok=True)

    # Train model
    train_dataset, val_dataset = load_cifar10h(config)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    model = HardToSoftLabelModel(config).to(device)
    model = train_model(model, train_loader, val_loader, config, device)

    # Generate soft labels for the entire dataset
    generate_soft_labels(model, config, device)


if __name__ == "__main__":
    main()
