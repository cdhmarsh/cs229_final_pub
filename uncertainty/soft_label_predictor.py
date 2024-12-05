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
from typing import Tuple, Optional
import argparse


class Config:
    """Configuration for training and model parameters."""

    # Data parameters
    data_dir = Path("../data")
    batch_size = 128
    val_split = 0.1

    # Training parameters
    learning_rate = 0.001
    weight_decay = 1e-4
    epochs = 100

    # Output paths
    model_dir = Path("models")
    model_path = model_dir / "soft_label_model.pt"
    output_dir = Path("outputs")
    soft_labels_path = output_dir / "cifar10_soft_labels.npy"

    gamma = 0.7


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
    def __init__(self):
        super().__init__()
        # Improved image encoder using ResNet-like blocks
        self.image_encoder = nn.Sequential(
            # Initial conv layer
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # ResNet-like blocks
            self._make_residual_block(64, 128),
            self._make_residual_block(128, 256),
            self._make_residual_block(256, 512),
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Uncertainty-aware label processing
        self.label_encoder = nn.Sequential(
            nn.Linear(10, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 512)
        )

        # Improved decoder with attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)

        self.decoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
            # Temperature-scaled softmax for better uncertainty calibration
            TemperatureSoftmax(temperature=1.5),
        )

    def _make_residual_block(self, in_channels, out_channels):
        return nn.Sequential(ResidualBlock(in_channels, out_channels), nn.MaxPool2d(2))

    def forward(self, image, hard_label):
        # Extract image features
        img_features = self.image_encoder(image)
        img_features = img_features.squeeze(-1).squeeze(-1)

        # Process label features
        label_features = self.label_encoder(hard_label)

        # Apply self-attention to capture feature relationships
        img_features = img_features.unsqueeze(0)
        attn_output, _ = self.attention(img_features, img_features, img_features)
        img_features = attn_output.squeeze(0)

        # Combine features
        combined = torch.cat([img_features, label_features], dim=1)

        return self.decoder(combined)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1), nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(residual)
        out = F.relu(out)

        return out


class TemperatureSoftmax(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, x):
        return F.softmax(x / self.temperature, dim=1)


def load_cifar10h(config: Config, full_training: bool = False) -> Tuple[Dataset, Optional[Dataset]]:
    """Load and prepare CIFAR-10H dataset."""
    cifar10h_probs_path = config.data_dir / "cifar-10h/cifar10h-probs.npy"
    if not cifar10h_probs_path.exists():
        raise FileNotFoundError(f"Soft labels not found at {cifar10h_probs_path}")

    cifar10h_probs = np.load(cifar10h_probs_path).astype(np.float32)
    cifar10_test = datasets.CIFAR10(
        root=config.data_dir / "cifar-10", train=False, download=True, transform=transforms.ToTensor()
    )

    full_dataset = CIFAR10SoftLabelDataset(cifar10_test, cifar10h_probs)

    if full_training:
        return full_dataset, None

    train_size = int((1 - config.val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size

    return random_split(full_dataset, [train_size, val_size])


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: Config,
    device: torch.device,
) -> nn.Module:
    """Train the model and return the best version."""
    kl_criterion = nn.KLDivLoss(reduction="batchmean")
    mse_criterion = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

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
            loss = (
                config.gamma * kl_criterion(predictions.log(), soft_labels)
                + (1 - config.gamma) * mse_criterion(predictions, soft_labels)
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        if val_loader is not None:
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
                    val_loss += loss.item()
            val_loss /= len(val_loader)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), config.model_path)

            if (epoch + 1) % 5 == 0:
                print(
                    f"Epoch {epoch + 1}/{config.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
        else:
            # Save model at regular intervals when training on full dataset
            if (epoch + 1) % 5 == 0:
                torch.save(model.state_dict(), config.model_path)
                print(f"Epoch {epoch + 1}/{config.epochs}, Train Loss: {train_loss:.4f}")

        scheduler.step()

    # Load best model if using validation, otherwise use final model
    if val_loader is not None:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Train on full dataset without validation split")
    args = parser.parse_args()

    config = Config()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Create necessary directories
    config.model_dir.mkdir(exist_ok=True)

    # Train model
    if args.full:
        train_dataset, _ = load_cifar10h(config, full_training=True)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = None
        print(f"Training on full dataset with {len(train_dataset)} samples")
    else:
        train_dataset, val_dataset = load_cifar10h(config)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
        print(f"Training on {len(train_dataset)} samples and validating on {len(val_dataset)} samples")

    model = ImageHardToSoftLabelModel().to(device)
    model = train_model(model, train_loader, val_loader, config, device)

    print("\nGenerating soft labels for the entire dataset")

    # Generate soft labels for the entire dataset
    generate_soft_labels(model, config, device)


if __name__ == "__main__":
    main()
    
    
"""
Training Output:

❯ python soft_label_predictor.py
Using device: mps
Files already downloaded and verified
Training on 9000 samples and validating on 1000 samples
Epoch 5/100, Train Loss: 0.0775, Val Loss: 0.0689
Epoch 10/100, Train Loss: 0.0711, Val Loss: 0.0946
Epoch 15/100, Train Loss: 0.0762, Val Loss: 0.0951
Epoch 20/100, Train Loss: 0.0656, Val Loss: 0.0591
Epoch 25/100, Train Loss: 0.0468, Val Loss: 0.0406
Epoch 30/100, Train Loss: 0.0372, Val Loss: 0.0378
Epoch 35/100, Train Loss: 0.0574, Val Loss: 0.0491
Epoch 40/100, Train Loss: 0.0421, Val Loss: 0.0647
Epoch 45/100, Train Loss: 0.0346, Val Loss: 0.0336
Epoch 50/100, Train Loss: 0.0294, Val Loss: 0.0286
Epoch 55/100, Train Loss: 0.0248, Val Loss: 0.0343
Epoch 60/100, Train Loss: 0.0206, Val Loss: 0.0194
Epoch 65/100, Train Loss: 0.0173, Val Loss: 0.0226
Epoch 70/100, Train Loss: 0.0167, Val Loss: 0.0243
Epoch 75/100, Train Loss: 0.0366, Val Loss: 0.0399
Epoch 80/100, Train Loss: 0.0300, Val Loss: 0.0600
Epoch 85/100, Train Loss: 0.0255, Val Loss: 0.0369
Epoch 90/100, Train Loss: 0.0228, Val Loss: 0.0231
Epoch 95/100, Train Loss: 0.0200, Val Loss: 0.0232
Epoch 100/100, Train Loss: 0.0240, Val Loss: 0.0404

Generating soft labels for the entire dataset
Files already downloaded and verified
Files already downloaded and verified

Soft Label Statistics:
Accuracy (argmax matches hard label): 0.9988
Average confidence (max probability): 0.9765
Average entropy (uncertainty): 0.1195

Class 0:
  Accuracy: 0.9998
  Average confidence: 0.9721

Class 1:
  Accuracy: 0.9993
  Average confidence: 0.9871

Class 2:
  Accuracy: 0.9988
  Average confidence: 0.9774

Class 3:
  Accuracy: 0.9968
  Average confidence: 0.9583

Class 4:
  Accuracy: 0.9955
  Average confidence: 0.9506

Class 5:
  Accuracy: 0.9993
  Average confidence: 0.9771

Class 6:
  Accuracy: 0.9997
  Average confidence: 0.9803

Class 7:
  Accuracy: 0.9997
  Average confidence: 0.9915

Class 8:
  Accuracy: 0.9995
  Average confidence: 0.9864

Class 9:
  Accuracy: 0.9995
  Average confidence: 0.9844

Saved soft labels to outputs/cifar10_soft_labels.npy
"""
