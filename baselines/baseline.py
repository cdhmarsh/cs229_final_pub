"""
This file contains the baseline experiments for comparing different model architectures
on CIFAR-10 with both hard and soft labels.

Baselines:
1. Training with CIFAR-10 hard labels + CIFAR-10H hard labels
2. Training with CIFAR-10 hard labels + CIFAR-10H soft labels

Models tested:
- ResNet34
- VGG-16
- Logistic Regression
- Random Forest
- XGBoost
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets, transforms, models
import torch.optim as optim
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
)
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import os
import json

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


class ModelFactory:
    @staticmethod
    def create_model(model_name: str, device: str) -> Any:
        if model_name == "resnet34":
            model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 10)
            return model.to(device)

        elif model_name == "vgg16":
            model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, 10)
            return model.to(device)

        elif model_name == "logistic":
            return LogisticRegression(max_iter=2000, n_jobs=-1)

        elif model_name == "randomforest":
            return RandomForestClassifier(n_estimators=100, n_jobs=-1)

        elif model_name == "xgboost":
            return xgb.XGBClassifier(objective="multi:softprob", num_class=10, n_jobs=-1)

        raise ValueError(f"Unknown model: {model_name}")


class CIFAR10LabelDataset(Dataset):
    def __init__(self, dataset, soft_labels=None):
        self.dataset = dataset
        self.soft_labels = soft_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.soft_labels is None:
            # Convert hard labels to one-hot
            label = F.one_hot(torch.tensor(label), num_classes=10).float()
        else:
            label = torch.tensor(self.soft_labels[idx])
        return image, label


def plot_training_curves(metrics: Dict[str, List[float]], title: str = "Training Curves"):
    """Plot training and validation metrics over epochs."""
    plt.figure(figsize=(12, 4))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(metrics["train_loss"], label="Train Loss")
    plt.plot(metrics["val_loss"], label="Val Loss")
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(metrics["train_acc"], label="Train Acc")
    plt.plot(metrics["val_acc"], label="Val Acc")
    plt.title("Accuracy Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f'plots/{title.lower().replace(" ", "_")}.png')
    plt.close()


def plot_comparative_curves(all_metrics: Dict[str, Dict], title: str = "Comparative Training Curves"):
    """Plot training curves for multiple models on the same graph."""
    plt.figure(figsize=(15, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    for model_name, metrics in all_metrics.items():
        if "train_loss" in metrics:  # Only for deep learning models
            plt.plot(metrics["train_loss"], label=f"{model_name} Train")
            plt.plot(metrics["val_loss"], label=f"{model_name} Val", linestyle="--")
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot accuracies
    plt.subplot(1, 2, 2)
    for model_name, metrics in all_metrics.items():
        if "train_acc" in metrics:  # Only for deep learning models
            plt.plot(metrics["train_acc"], label=f"{model_name} Train")
            plt.plot(metrics["val_acc"], label=f"{model_name} Val", linestyle="--")
    plt.title("Accuracy Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f'plots/{title.lower().replace(" ", "_")}.png')
    plt.close()


def save_metrics(metrics: Dict, model_name: str):
    """Save metrics dictionary to JSON file."""
    metrics_path = f"metrics/{model_name}_metrics.json"
    os.makedirs("metrics", exist_ok=True)

    # Convert numpy arrays and tensors to lists for JSON serialization
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (np.ndarray, torch.Tensor)):
            serializable_metrics[key] = value.tolist()
        elif isinstance(value, list):
            serializable_metrics[key] = [
                v.item() if isinstance(v, (np.ndarray, torch.Tensor)) else v for v in value
            ]
        else:
            serializable_metrics[key] = value

    with open(metrics_path, "w") as f:
        json.dump(serializable_metrics, f)


def compute_additional_metrics(all_preds, all_labels, all_probs):
    """Compute additional metrics including AUPRC."""
    metrics = {}

    # Compute precision, recall, and F1 for each class
    for i in range(10):
        binary_labels = (np.array(all_labels) == i).astype(int)
        binary_probs = np.array(all_probs)[:, i]

        precision, recall, _ = precision_recall_curve(binary_labels, binary_probs)
        auprc = average_precision_score(binary_labels, binary_probs)

        metrics[f"class_{CIFAR10_CLASSES[i]}_auprc"] = auprc

    # Compute macro-averaged metrics
    metrics["macro_auprc"] = np.mean([metrics[f"class_{c}_auprc"] for c in CIFAR10_CLASSES])

    return metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    device: str,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    model_name: str = None,
) -> Tuple[nn.Module, Dict]:
    """Train a deep learning model and return the trained model and metrics."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    metrics = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if len(labels.shape) > 1:  # For soft labels
                _, predicted = torch.max(outputs.data, 1)
                _, true_labels = torch.max(labels, 1)
            else:  # For hard labels
                _, predicted = torch.max(outputs.data, 1)
                true_labels = labels

            train_total += labels.size(0)
            train_correct += (predicted == true_labels).sum().item()

        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        # Validation phase
        val_metrics = evaluate_model(model, val_loader, criterion, device)

        # Store metrics
        metrics["train_loss"].append(avg_train_loss)
        metrics["train_acc"].append(train_acc)
        metrics["val_loss"].append(val_metrics["loss"])
        metrics["val_acc"].append(val_metrics["accuracy"])

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
            f"Train Acc: {train_acc:.2f}%, Val Acc: {val_metrics['accuracy']:.2f}%"
        )

        # Save best model
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            if model_name:
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(), f"models/{model_name}.pth")

    return model, metrics


def evaluate_model(model, data_loader, criterion, device):
    """Evaluate model performance."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            if len(labels.shape) > 1:
                _, predicted = torch.max(outputs.data, 1)
                _, true_labels = torch.max(labels, 1)
            else:
                _, predicted = torch.max(outputs.data, 1)
                true_labels = labels

            total += labels.size(0)
            correct += (predicted == true_labels).sum().item()

    return {"loss": total_loss / len(data_loader), "accuracy": 100 * correct / total}


def convert_dataloader_to_numpy(dataloader: DataLoader) -> tuple:
    """Convert PyTorch DataLoader to numpy arrays for traditional ML models."""
    all_features = []
    all_labels = []

    for images, labels in dataloader:
        # Flatten images for traditional ML models
        features = images.view(images.size(0), -1).numpy()
        all_features.append(features)

        if len(labels.shape) > 1:  # Handle soft labels
            labels = torch.argmax(labels, dim=1)
        all_labels.append(labels.numpy())

    return np.concatenate(all_features), np.concatenate(all_labels)


def evaluate_traditional_ml_model(model, X_test, y_test, model_name: str) -> Dict:
    """Evaluate traditional ML models and return metrics."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    metrics = compute_additional_metrics(predictions, y_test, probabilities)

    # Add accuracy metric
    accuracy = (predictions == y_test).mean() * 100
    metrics["accuracy"] = accuracy

    return metrics


def load_cifar10_experiment():
    """Load and prepare CIFAR-10 datasets."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ]
    )

    full_dataset = datasets.CIFAR10(root="../data/cifar-10", train=True, download=True, transform=transform)
    train_dataset = datasets.CIFAR10(root="../data/cifar-10", train=False, download=True, transform=transform)

    augment_size = int(0.7 * len(full_dataset))
    val_size = (len(full_dataset) - augment_size) // 2
    test_size = len(full_dataset) - augment_size - val_size

    augment_dataset, test_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [augment_size, test_size, val_size], generator=torch.Generator().manual_seed(229)
    )

    return augment_dataset, train_dataset, test_dataset, val_dataset


if __name__ == "__main__":
    # Set device
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Load datasets
    augment_dataset, train_dataset, test_dataset, val_dataset = load_cifar10_experiment()

    # Print dataset sizes
    print(f"\nDataset sizes:")
    print(f"- Augment dataset: {len(augment_dataset)} samples")
    print(f"- Train dataset: {len(train_dataset)} samples")
    print(f"- Test dataset: {len(test_dataset)} samples")
    print(f"- Validation dataset: {len(val_dataset)} samples")

    # Create dataloaders
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    augment_loader = DataLoader(augment_dataset, batch_size=batch_size, shuffle=False)

    # Load CIFAR-10H soft labels
    cifar10h_probs = np.load("../data/cifar-10h/cifar10h-probs.npy").astype(np.float32)

    # Create datasets for baseline experiments
    combined_hard_dataset = ConcatDataset([augment_dataset, train_dataset])
    combined_hard_loader = DataLoader(combined_hard_dataset, batch_size=batch_size, shuffle=True)

    hard_label_dataset = CIFAR10LabelDataset(augment_dataset)
    soft_label_dataset = CIFAR10LabelDataset(train_dataset, cifar10h_probs)
    combined_mixed_dataset = ConcatDataset([hard_label_dataset, soft_label_dataset])
    combined_mixed_loader = DataLoader(combined_mixed_dataset, batch_size=batch_size, shuffle=True)

    # Run experiments model by model
    models_to_test = [
        # "resnet34",
        # "vgg16",
        "logistic",
        "randomforest",
        "xgboost",
    ]
    baseline1_metrics = {}
    baseline2_metrics = {}

    for model_name in models_to_test:
        print(f"\n=== Running experiments for {model_name} ===")

        # Baseline 1: Hard Labels
        print(f"\nBaseline 1: Hard Labels ({len(combined_hard_dataset)} hard samples)")
        model = ModelFactory.create_model(model_name, device)

        if model_name in ["resnet34", "vgg16"]:
            model, metrics = train_model(
                model=model,
                train_loader=combined_hard_loader,
                val_loader=val_loader,
                num_epochs=30,
                device=device,
                model_name=f"{model_name}_baseline1",
            )
            # Evaluate last model
            last_metrics = evaluate_model(model, test_loader, nn.CrossEntropyLoss(), device)

            # Load and evaluate best model
            best_model = ModelFactory.create_model(model_name, device)
            best_model.load_state_dict(torch.load(f"models/{model_name}_baseline1.pth", weights_only=True))
            best_metrics = evaluate_model(best_model, test_loader, nn.CrossEntropyLoss(), device)
            metrics.update(best_metrics)

            print(f"\nLast model test accuracy: {last_metrics['accuracy']:.2f}%")
            print(f"Best model test accuracy: {best_metrics['accuracy']:.2f}%")
        else:
            X_train, y_train = convert_dataloader_to_numpy(combined_hard_loader)
            X_test, y_test = convert_dataloader_to_numpy(test_loader)
            model.fit(X_train, y_train)
            metrics = evaluate_traditional_ml_model(model, X_test, y_test, model_name)

        baseline1_metrics[model_name] = metrics
        save_metrics(metrics, f"{model_name}_baseline1")

        # Baseline 2: Mixed Labels
        print(
            f"\nBaseline 2: Mixed Labels ({len(hard_label_dataset)} hard samples + {len(soft_label_dataset)} soft samples)"
        )
        model = ModelFactory.create_model(model_name, device)

        if model_name in ["resnet34", "vgg16"]:
            model, metrics = train_model(
                model=model,
                train_loader=combined_mixed_loader,
                val_loader=val_loader,
                num_epochs=30,
                device=device,
                model_name=f"{model_name}_baseline2",
            )
            # Evaluate last model
            last_metrics = evaluate_model(model, test_loader, nn.CrossEntropyLoss(), device)

            # Load and evaluate best model
            best_model = ModelFactory.create_model(model_name, device)
            best_model.load_state_dict(torch.load(f"models/{model_name}_baseline2.pth", weights_only=True))
            best_metrics = evaluate_model(best_model, test_loader, nn.CrossEntropyLoss(), device)
            metrics.update(best_metrics)

            print(f"\nLast model test accuracy: {last_metrics['accuracy']:.2f}%")
            print(f"Best model test accuracy: {best_metrics['accuracy']:.2f}%")
        else:
            X_train, y_train = convert_dataloader_to_numpy(combined_mixed_loader)
            X_test, y_test = convert_dataloader_to_numpy(test_loader)
            model.fit(X_train, y_train)
            metrics = evaluate_traditional_ml_model(model, X_test, y_test, model_name)

        baseline2_metrics[model_name] = metrics
        save_metrics(metrics, f"{model_name}_baseline2")

        # Print results for this model
        print(f"\nResults for {model_name}:")
        print(f"Baseline 1 (Hard Labels):")
        print(f"- Test Accuracy: {baseline1_metrics[model_name]['accuracy']:.2f}%")
        if "macro_auprc" in baseline1_metrics[model_name]:
            print(f"- Macro AUPRC: {baseline1_metrics[model_name]['macro_auprc']:.4f}")

        print(f"\nBaseline 2 (Mixed Labels):")
        print(f"- Test Accuracy: {baseline2_metrics[model_name]['accuracy']:.2f}%")
        if "macro_auprc" in baseline2_metrics[model_name]:
            print(f"- Macro AUPRC: {baseline2_metrics[model_name]['macro_auprc']:.4f}")

    # Plot comparative curves
    plot_comparative_curves(baseline1_metrics, title="Baseline 1 - Hard Labels")
    plot_comparative_curves(baseline2_metrics, title="Baseline 2 - Mixed Labels")


"""
Dataset sizes:
- Augment dataset: 35000 samples
- Train dataset: 10000 samples
- Test dataset: 7500 samples
- Validation dataset: 7500 samples

=== Running experiments for resnet34 ===

Baseline 1: Hard Labels (45000 hard samples)
Epoch [1/30] Train Loss: 0.9145, Val Loss: 0.7292, Train Acc: 68.64%, Val Acc: 75.44%
Epoch [2/30] Train Loss: 0.5891, Val Loss: 0.8736, Train Acc: 79.81%, Val Acc: 71.80%
Epoch [3/30] Train Loss: 0.4522, Val Loss: 0.8865, Train Acc: 84.72%, Val Acc: 71.93%
Epoch [4/30] Train Loss: 0.3407, Val Loss: 0.6673, Train Acc: 88.28%, Val Acc: 78.23%
Epoch [5/30] Train Loss: 0.2606, Val Loss: 0.6506, Train Acc: 91.01%, Val Acc: 80.00%
Epoch [6/30] Train Loss: 0.2001, Val Loss: 0.6748, Train Acc: 93.13%, Val Acc: 79.72%
Epoch [7/30] Train Loss: 0.1683, Val Loss: 0.7163, Train Acc: 94.15%, Val Acc: 80.57%
Epoch [8/30] Train Loss: 0.1273, Val Loss: 0.7652, Train Acc: 95.61%, Val Acc: 80.31%
Epoch [9/30] Train Loss: 0.1073, Val Loss: 0.8019, Train Acc: 96.27%, Val Acc: 79.84%
Epoch [10/30] Train Loss: 0.0982, Val Loss: 0.7323, Train Acc: 96.69%, Val Acc: 81.32%
Epoch [11/30] Train Loss: 0.0838, Val Loss: 0.8673, Train Acc: 97.12%, Val Acc: 79.72%
Epoch [12/30] Train Loss: 0.0754, Val Loss: 0.8055, Train Acc: 97.42%, Val Acc: 80.93%
Epoch [13/30] Train Loss: 0.0693, Val Loss: 0.8734, Train Acc: 97.60%, Val Acc: 79.76%
Epoch [14/30] Train Loss: 0.0785, Val Loss: 0.8649, Train Acc: 97.30%, Val Acc: 80.47%
Epoch [15/30] Train Loss: 0.0624, Val Loss: 0.8309, Train Acc: 97.86%, Val Acc: 80.92%
Epoch [16/30] Train Loss: 0.0597, Val Loss: 0.8514, Train Acc: 97.95%, Val Acc: 81.11%
Epoch [17/30] Train Loss: 0.0534, Val Loss: 1.0142, Train Acc: 98.22%, Val Acc: 78.23%
Epoch [18/30] Train Loss: 0.0478, Val Loss: 0.9257, Train Acc: 98.40%, Val Acc: 80.21%
Epoch [19/30] Train Loss: 0.1047, Val Loss: 0.8922, Train Acc: 96.63%, Val Acc: 80.48%
Epoch [20/30] Train Loss: 0.1207, Val Loss: 0.8018, Train Acc: 96.21%, Val Acc: 80.19%
Epoch [21/30] Train Loss: 0.0437, Val Loss: 0.9423, Train Acc: 98.55%, Val Acc: 80.29%
Epoch [22/30] Train Loss: 0.0310, Val Loss: 1.0003, Train Acc: 98.95%, Val Acc: 80.64%
Epoch [23/30] Train Loss: 0.0339, Val Loss: 1.0423, Train Acc: 98.88%, Val Acc: 79.92%
Epoch [24/30] Train Loss: 0.0365, Val Loss: 0.9933, Train Acc: 98.80%, Val Acc: 80.12%
Epoch [25/30] Train Loss: 0.0481, Val Loss: 0.9257, Train Acc: 98.38%, Val Acc: 81.32%
Epoch [26/30] Train Loss: 0.0361, Val Loss: 1.1369, Train Acc: 98.72%, Val Acc: 78.48%
Epoch [27/30] Train Loss: 0.0376, Val Loss: 0.9573, Train Acc: 98.71%, Val Acc: 81.08%
Epoch [28/30] Train Loss: 0.0384, Val Loss: 1.0048, Train Acc: 98.72%, Val Acc: 79.33%
Epoch [29/30] Train Loss: 0.0343, Val Loss: 1.0899, Train Acc: 98.85%, Val Acc: 79.09%
Epoch [30/30] Train Loss: 0.0394, Val Loss: 1.0650, Train Acc: 98.65%, Val Acc: 79.81%

Last model test accuracy: 79.57%
Best model test accuracy: 81.44%

Baseline 2: Mixed Labels (35000 hard samples + 10000 soft samples)
Epoch [1/30] Train Loss: 0.9286, Val Loss: 0.8060, Train Acc: 68.76%, Val Acc: 72.88%
Epoch [2/30] Train Loss: 0.6040, Val Loss: 0.8157, Train Acc: 80.33%, Val Acc: 73.11%
Epoch [3/30] Train Loss: 0.4692, Val Loss: 0.6173, Train Acc: 84.98%, Val Acc: 78.65%
Epoch [4/30] Train Loss: 0.3692, Val Loss: 0.6345, Train Acc: 88.24%, Val Acc: 78.95%
Epoch [5/30] Train Loss: 0.2939, Val Loss: 0.6760, Train Acc: 91.23%, Val Acc: 78.92%
Epoch [6/30] Train Loss: 0.2359, Val Loss: 0.6707, Train Acc: 93.41%, Val Acc: 79.56%
Epoch [7/30] Train Loss: 0.2071, Val Loss: 0.6522, Train Acc: 94.35%, Val Acc: 81.35%
Epoch [8/30] Train Loss: 0.1642, Val Loss: 0.6898, Train Acc: 96.00%, Val Acc: 80.43%
Epoch [9/30] Train Loss: 0.1493, Val Loss: 0.6893, Train Acc: 96.49%, Val Acc: 80.75%
Epoch [10/30] Train Loss: 0.1330, Val Loss: 0.7365, Train Acc: 97.17%, Val Acc: 79.49%
Epoch [11/30] Train Loss: 0.1299, Val Loss: 0.6954, Train Acc: 97.37%, Val Acc: 80.87%
Epoch [12/30] Train Loss: 0.1182, Val Loss: 0.8951, Train Acc: 97.81%, Val Acc: 77.37%
Epoch [13/30] Train Loss: 0.1209, Val Loss: 0.6980, Train Acc: 97.64%, Val Acc: 80.71%
Epoch [14/30] Train Loss: 0.1145, Val Loss: 0.8427, Train Acc: 97.89%, Val Acc: 78.41%
Epoch [15/30] Train Loss: 0.1111, Val Loss: 0.9081, Train Acc: 97.99%, Val Acc: 76.56%
Epoch [16/30] Train Loss: 0.1340, Val Loss: 0.7832, Train Acc: 97.13%, Val Acc: 78.91%
Epoch [17/30] Train Loss: 0.0993, Val Loss: 0.7171, Train Acc: 98.41%, Val Acc: 81.39%
Epoch [18/30] Train Loss: 0.0975, Val Loss: 0.7260, Train Acc: 98.49%, Val Acc: 80.85%
Epoch [19/30] Train Loss: 0.0972, Val Loss: 0.7264, Train Acc: 98.52%, Val Acc: 81.20%
Epoch [20/30] Train Loss: 0.2074, Val Loss: 0.6902, Train Acc: 94.64%, Val Acc: 80.85%
Epoch [21/30] Train Loss: 0.0947, Val Loss: 0.7092, Train Acc: 98.60%, Val Acc: 81.59%
Epoch [22/30] Train Loss: 0.0754, Val Loss: 0.6579, Train Acc: 99.29%, Val Acc: 82.47%
Epoch [23/30] Train Loss: 0.0716, Val Loss: 0.6978, Train Acc: 99.31%, Val Acc: 81.64%
Epoch [24/30] Train Loss: 0.0853, Val Loss: 0.7296, Train Acc: 98.77%, Val Acc: 80.93%
Epoch [25/30] Train Loss: 0.1025, Val Loss: 0.8150, Train Acc: 98.20%, Val Acc: 78.48%
Epoch [26/30] Train Loss: 0.1019, Val Loss: 0.7895, Train Acc: 98.29%, Val Acc: 79.23%
Epoch [27/30] Train Loss: 0.0847, Val Loss: 0.8495, Train Acc: 98.89%, Val Acc: 79.32%
Epoch [28/30] Train Loss: 0.0853, Val Loss: 0.8264, Train Acc: 98.85%, Val Acc: 79.11%
Epoch [29/30] Train Loss: 0.0878, Val Loss: 0.8762, Train Acc: 98.67%, Val Acc: 78.20%
Epoch [30/30] Train Loss: 0.0948, Val Loss: 0.8129, Train Acc: 98.45%, Val Acc: 79.19%

Last model test accuracy: 78.95%
Best model test accuracy: 83.31%

Results for resnet34:
Baseline 1 (Hard Labels):
- Test Accuracy: 81.44%

Baseline 2 (Mixed Labels):
- Test Accuracy: 83.31%


=== Running experiments for logistic ===

Baseline 1: Hard Labels (45000 hard samples)
/opt/homebrew/Caskroom/miniconda/base/envs/cs229_project/lib/python3.9/site-packages/sklearn/linear_model/_logistic.py:469: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(

Baseline 2: Mixed Labels (35000 hard samples + 10000 soft samples)
/opt/homebrew/Caskroom/miniconda/base/envs/cs229_project/lib/python3.9/site-packages/sklearn/linear_model/_logistic.py:469: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(

Results for logistic:
Baseline 1 (Hard Labels):
- Test Accuracy: 38.68%
- Macro AUPRC: 0.3735

Baseline 2 (Mixed Labels):
- Test Accuracy: 39.04%
- Macro AUPRC: 0.3753

=== Running experiments for randomforest ===

Baseline 1: Hard Labels (45000 hard samples)

Baseline 2: Mixed Labels (35000 hard samples + 10000 soft samples)

Results for randomforest:
Baseline 1 (Hard Labels):
- Test Accuracy: 46.85%
- Macro AUPRC: 0.4686

Baseline 2 (Mixed Labels):
- Test Accuracy: 45.99%
- Macro AUPRC: 0.4651

=== Running experiments for xgboost ===

Baseline 1: Hard Labels (45000 hard samples)

Baseline 2: Mixed Labels (35000 hard samples + 10000 soft samples)

Results for xgboost:
Baseline 1 (Hard Labels):
- Test Accuracy: 54.28%
- Macro AUPRC: 0.5874

Baseline 2 (Mixed Labels):
- Test Accuracy: 53.67%
- Macro AUPRC: 0.5831


"""
