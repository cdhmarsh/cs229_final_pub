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
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os
import argparse


class CIFAR10SoftLabelDataset(Dataset):
    def __init__(self, cifar10_dataset, soft_labels):
        """
        :param cifar10_dataset: CIFAR-10 dataset (test set in this case).
        :param soft_labels: Soft labels from CIFAR-10H.
        """
        self.cifar10_dataset = cifar10_dataset
        self.soft_labels = soft_labels

    def __len__(self):
        return len(self.cifar10_dataset)

    def __getitem__(self, idx):
        image, hard_label = self.cifar10_dataset[idx]
        soft_label = self.soft_labels[idx]  # Access aligned soft label
        return image, F.one_hot(torch.tensor(hard_label), num_classes=10).float(), soft_label


class ImageHardLabelToSoftLabelModel(nn.Module):
    def __init__(self):
        super(ImageHardLabelToSoftLabelModel, self).__init__()
        # Feature extractor for the image

        # Image encoder layers:
        # Input: (batch_size, 3, 32, 32) - CIFAR-10 RGB images
        self.image_encoder = nn.Sequential(
            # Conv1: (3, 32, 32) -> (32, 32, 32)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            # MaxPool1: (32, 32, 32) -> (32, 16, 16)
            nn.MaxPool2d(kernel_size=2),
            # Conv2: (32, 16, 16) -> (64, 16, 16)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            # MaxPool2: (64, 16, 16) -> (64, 8, 8)
            nn.MaxPool2d(kernel_size=2),
        )
        # Fully connected layers
        self.fc_image = nn.Linear(64 * 8 * 8, 128)
        self.fc_label = nn.Linear(10, 128)  # To process hard label
        self.fc_output = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10),
            nn.Softmax(dim=1),  # Predict soft label as probability distribution
        )

    def forward(self, image, hard_label):
        # Encode the image
        image_features = self.image_encoder(image)
        image_features = image_features.view(image_features.size(0), -1)  # Flatten
        image_features = self.fc_image(image_features)

        # Process the hard label
        label_features = self.fc_label(hard_label)

        # Concatenate features and predict soft label
        combined_features = torch.cat([image_features, label_features], dim=1)
        soft_label = self.fc_output(combined_features)
        return soft_label


# Load CIFAR-10H dataset and return a Dataset
def load_cifar10h():
    cifar10h_probs_path = "../data/cifar-10h/cifar10h-probs.npy"
    if not os.path.exists(cifar10h_probs_path):
        raise FileNotFoundError(
            f"Soft labels not found at {cifar10h_probs_path}. Please ensure the CIFAR-10H data is downloaded."
        )

    cifar10h_probs = np.load(cifar10h_probs_path).astype(np.float32)
    cifar10_test = datasets.CIFAR10(
        root="../data/cifar-10", train=False, download=True, transform=transforms.ToTensor()
    )

    # Create full dataset
    full_dataset = CIFAR10SoftLabelDataset(cifar10_test, cifar10h_probs)
    return full_dataset


def train_model(model, train_loader, criterion, optimizer, epochs, device):
    # Track metrics
    best_loss = float("inf")
    train_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0

        for images, hard_labels, soft_labels in train_loader:
            images, hard_labels, soft_labels = (
                images.to(device),
                hard_labels.to(device),
                soft_labels.to(device),
            )

            # Forward pass
            predictions = model(images, hard_labels)

            # Compute loss
            loss = criterion(predictions.log(), soft_labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_epoch_loss = epoch_loss / num_batches
        train_losses.append(avg_epoch_loss)

        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), "models/best_model.pt")

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_epoch_loss:.4f}")


def evaluate_model(model, test_loader, device):
    model.eval()
    total_mse = 0
    total_kl_div = 0
    total_cross_entropy = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, hard_labels, target_soft_labels in test_loader:
            images = images.to(device)
            hard_labels = hard_labels.to(device)
            target_soft_labels = target_soft_labels.to(device)

            # Get model predictions
            predicted_soft_labels = model(images, hard_labels)

            # Calculate various metrics
            mse = F.mse_loss(predicted_soft_labels, target_soft_labels)
            kl_div = F.kl_div(predicted_soft_labels.log(), target_soft_labels, reduction='batchmean')
            cross_entropy = -torch.sum(target_soft_labels * torch.log(predicted_soft_labels + 1e-10)) / images.size(0)

            total_mse += mse.item()
            total_kl_div += kl_div.item()
            total_cross_entropy += cross_entropy.item()
            num_batches += 1

    # Calculate averages
    avg_mse = total_mse / num_batches
    avg_kl_div = total_kl_div / num_batches
    avg_cross_entropy = total_cross_entropy / num_batches

    print(f"Average MSE Loss: {avg_mse:.4f}")
    print(f"Average KL Divergence: {avg_kl_div:.4f}")
    print(f"Average Cross Entropy: {avg_cross_entropy:.4f}")
    
    return {
        'mse': avg_mse,
        'kl_div': avg_kl_div,
        'cross_entropy': avg_cross_entropy
    }


class Config:
    batch_size = 128
    learning_rate = 0.001
    epochs = 50
    model_path = "models/soft_label_model.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Run evaluation only using saved model")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    config = Config()
    full_dataset = load_cifar10h()

    # Initialize model
    model = ImageHardLabelToSoftLabelModel().to(device)

    if not args.eval:
        # Train mode
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.KLDivLoss(reduction="batchmean")

        # Create data loader
        train_loader = DataLoader(full_dataset, batch_size=config.batch_size, shuffle=True)

        # Train the model
        train_model(model, train_loader, criterion, optimizer, config.epochs, device)
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)

        # Save the model
        torch.save(model.state_dict(), config.model_path)
    else:
        # Load saved model
        model.load_state_dict(torch.load(config.model_path, weights_only=True))
        model.eval()

    # Evaluate model
    val_loader = DataLoader(full_dataset, batch_size=config.batch_size, shuffle=False)
    evaluate_model(model, val_loader, device)


if __name__ == "__main__":
    main()
