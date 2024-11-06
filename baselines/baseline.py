"""
This file contains the code to run the baseline experiments.

More specifically, we are investigating the performance of basic models on the CIFAR-10 and CIFAR-10H datasets. The tasks for these datasets are multi-class classification.

The basic models include:
    * ResNet-18
    * Logistic Regression
    * Random Forest
    * XGBoost
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load CIFAR-10H dataset and return a DataLoader
def load_cifar10h():
    cifar10h_probs = np.load("../data/cifar-10h/cifar10h-probs.npy").astype(np.float32)
    cifar10_test = datasets.CIFAR10(
        root="../data/cifar-10", train=False, download=True, transform=transforms.ToTensor()
    )

    class CIFAR10H(Dataset):
        def __init__(self, cifar10_dataset, soft_labels):
            self.cifar10_dataset = cifar10_dataset
            self.soft_labels = soft_labels

        def __len__(self):
            return len(self.cifar10_dataset)

        def __getitem__(self, idx):
            image, _ = self.cifar10_dataset[idx]
            soft_label = self.soft_labels[idx]
            return image.float(), soft_label

    cifar10h_dataset = CIFAR10H(cifar10_test, cifar10h_probs)
    return cifar10h_dataset


# Load CIFAR-10 dataset and return a DataLoader
def load_cifar10():
    transform = transforms.Compose([transforms.ToTensor(), transforms.ConvertImageDtype(torch.float32)])
    full_dataset = datasets.CIFAR10(root="../data/cifar-10", train=True, download=True, transform=transform)

    # This dataset will be used for testing and validation.
    #   30% of the data will be used for validation, and 70% for testing.
    test_size = int(0.7 * len(full_dataset))
    val_size = len(full_dataset) - test_size
    test_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [test_size, val_size], generator=torch.Generator().manual_seed(229)
    )

    return test_dataset, val_dataset


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 25 == 0:
                print(
                    f"  Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/25:.4f}"
                )
                running_loss = 0.0

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)

                if len(labels.shape) > 1:  # For soft labels
                    _, predicted = torch.max(outputs.data, 1)
                    _, labels = torch.max(labels, 1)
                else:  # For hard labels
                    _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_loss += criterion(outputs, labels).item()

        accuracy = 100 * correct / total
        val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return model


def main():
    cifar10h_dataset = load_cifar10h()
    cifar10h_loader = DataLoader(cifar10h_dataset, batch_size=128, shuffle=True)
    print(f"CIFAR-10H dataset loaded with {len(cifar10h_dataset)} samples")

    cifar10_test_dataset, cifar10_val_dataset = load_cifar10()  # Changed variable name to reflect split
    cifar10_test_loader = DataLoader(cifar10_test_dataset, batch_size=128, shuffle=False)
    cifar10_val_loader = DataLoader(cifar10_val_dataset, batch_size=128, shuffle=False)
    print(
        f"CIFAR-10 dataset loaded with {len(cifar10_test_dataset)} test and {len(cifar10_val_dataset)} validation samples"
    )

    # Neural Network models
    nn_models = [
        models.resnet50(weights="DEFAULT"),
        models.vgg16(weights="DEFAULT"),
    ]
    for model in nn_models:
        print(f"\nTraining {model.__class__.__name__} on CIFAR-10H...")

        # Adjust the final layer for CIFAR-10
        if isinstance(model, models.ResNet):
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 10)
        elif isinstance(model, models.VGG):
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, 10)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1)

        train_model(
            model,
            cifar10h_loader,
            cifar10_val_loader,
            criterion,
            optimizer,
            num_epochs=20,
        )
        torch.save(model.state_dict(), f"models/{model.__class__.__name__}_cifar10h.pth")

    # Prepare data for ML models
    X_cifar10h = np.array([img.numpy().flatten() for img, _ in cifar10h_dataset])
    y_cifar10h = np.array([np.argmax(label) for _, label in cifar10h_dataset])

    X_cifar10 = np.array([img.numpy().flatten() for img, _ in cifar10_test_dataset])
    y_cifar10 = np.array([label for _, label in cifar10_test_dataset])

    # Scale the data for ML models
    scaler = StandardScaler()
    X_cifar10h_scaled = scaler.fit_transform(X_cifar10h)
    X_cifar10_scaled = scaler.transform(X_cifar10)

    # Machine Learning models
    ml_models = [
        LogisticRegression(max_iter=3000, n_jobs=-1),  # Increased max_iter
        RandomForestClassifier(n_jobs=-1),
        XGBClassifier(n_jobs=-1),
    ]
    for model in ml_models:
        print(f"\nTraining {model.__class__.__name__} on CIFAR-10H...")
        model.fit(X_cifar10h_scaled, y_cifar10h)  # Use scaled data

    # Evaluate on CIFAR-10 train set
    for model in nn_models:
        model.load_state_dict(torch.load(f"models/{model.__class__.__name__}_cifar10h.pth", weights_only=True))
        model.eval()

        correct = 0
        total = 0
        device = next(model.parameters()).device
        with torch.no_grad():
            for images, labels in cifar10_test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"{model.__class__.__name__} Accuracy on CIFAR-10 test set: {accuracy:.2f}%")

    for model in ml_models:
        y_pred = model.predict(X_cifar10_scaled)  # Use scaled data
        accuracy = accuracy_score(y_cifar10, y_pred)
        accuracy = 100 * accuracy
        print(f"{model.__class__.__name__} Accuracy on CIFAR-10 test set: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
