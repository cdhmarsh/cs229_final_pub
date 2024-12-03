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
    - Input: CIFAR-10 image.
    - Auxiliary Input (Optional): CIFAR-10 hard label (as part of input or a separate embedding layer).
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
    batch_size = 128
    val_split = 0.1

    # Model parameters
    hidden_dims = [128, 64]
    dropout_rate = 0.2

    # Training parameters
    learning_rate = 0.001
    weight_decay = 1e-4
    epochs = 50

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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, hard_label = self.cifar10_dataset[idx]
        soft_label = self.soft_labels[idx]
        return image, F.one_hot(torch.tensor(hard_label), num_classes=10).float(), soft_label


class ImageHardToSoftLabelModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_image = nn.Linear(64 * 8 * 8, 128)
        self.fc_label = nn.Linear(10, 128)

        layers = []
        input_dim = 256

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

        self.fc_output = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor, hard_label: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(image)
        image_features = image_features.view(image_features.size(0), -1)
        image_features = self.fc_image(image_features)
        label_features = self.fc_label(hard_label)
        combined_features = torch.cat([image_features, label_features], dim=1)
        return self.fc_output(combined_features)


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

    for epoch in range(config.epochs):
        # Training
        model.train()
        train_loss = 0
        for images, hard_labels, soft_labels in train_loader:
            images, hard_labels, soft_labels = (
                images.to(device),
                hard_labels.to(device),
                soft_labels.to(device),
            )
            optimizer.zero_grad()
            predictions = model(images, hard_labels)
            loss = criterion(predictions.log(), soft_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, hard_labels, soft_labels in val_loader:
                images, hard_labels, soft_labels = (
                    images.to(device),
                    hard_labels.to(device),
                    soft_labels.to(device),
                )
                predictions = model(images, hard_labels)
                val_loss += criterion(predictions.log(), soft_labels).item()
        val_loss /= len(val_loader)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.model_path)

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
    transform = transforms.ToTensor()
    cifar10_train = datasets.CIFAR10(
        root=config.data_dir / "cifar-10", train=True, download=True, transform=transform
    )
    cifar10_test = datasets.CIFAR10(
        root=config.data_dir / "cifar-10", train=False, download=True, transform=transform
    )

    model.eval()
    soft_labels = []
    hard_labels = []

    with torch.no_grad():
        # Process training set
        for image, label in cifar10_train:
            image = image.unsqueeze(0).to(device)
            hard_label = F.one_hot(torch.tensor(label), num_classes=10).float().unsqueeze(0).to(device)
            soft_label = model(image, hard_label).squeeze(0)
            soft_labels.append(soft_label.cpu().numpy())
            hard_labels.append(label)

        # Process test set
        for image, label in cifar10_test:
            image = image.unsqueeze(0).to(device)
            hard_label = F.one_hot(torch.tensor(label), num_classes=10).float().unsqueeze(0).to(device)
            soft_label = model(image, hard_label).squeeze(0)
            soft_labels.append(soft_label.cpu().numpy())
            hard_labels.append(label)

    # Convert to numpy arrays
    soft_labels = np.array(soft_labels)
    hard_labels = np.array(hard_labels)
    
    # Calculate statistics
    soft_label_argmax = np.argmax(soft_labels, axis=1)
    accuracy = np.mean(soft_label_argmax == hard_labels)
    avg_confidence = np.mean(np.max(soft_labels, axis=1))
    entropy = -np.sum(soft_labels * np.log(soft_labels + 1e-10), axis=1).mean()
    
    print("\nSoft Label Statistics:")
    print(f"Accuracy (argmax matches hard label): {accuracy:.4f}")
    print(f"Average confidence (max probability): {avg_confidence:.4f}")
    print(f"Average entropy (uncertainty): {entropy:.4f}")
    
    # Per-class statistics
    for i in range(10):
        class_mask = hard_labels == i
        class_accuracy = np.mean(soft_label_argmax[class_mask] == hard_labels[class_mask])
        class_confidence = np.mean(np.max(soft_labels[class_mask], axis=1))
        print(f"\nClass {i}:")
        print(f"  Accuracy: {class_accuracy:.4f}")
        print(f"  Average confidence: {class_confidence:.4f}")

    # Save soft labels
    config.output_dir.mkdir(exist_ok=True)
    np.save(config.soft_labels_path, soft_labels)
    print(f"\nSaved soft labels to {config.soft_labels_path}")


def main():
    config = Config()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Create necessary directories
    config.model_dir.mkdir(exist_ok=True)

    # Train model
    train_dataset, val_dataset = load_cifar10h(config)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    model = ImageHardToSoftLabelModel(config).to(device)
    model = train_model(model, train_loader, val_loader, config, device)

    # Generate soft labels for the entire dataset
    generate_soft_labels(model, config, device)


if __name__ == "__main__":
    main()
