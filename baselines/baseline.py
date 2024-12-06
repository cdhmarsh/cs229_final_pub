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
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
)
import seaborn as sns
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
    plt.savefig(f'plots/{title.lower().replace(" ", "")}.png')
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


def run_baseline_experiments(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    models_to_test: List[str] = ["resnet34", "vgg16", "logistic", "randomforest", "xgboost"],
    baseline_name: str = "baseline1",  # Add baseline name parameter
) -> Dict[str, Dict]:
    """Run experiments for all specified models and return their metrics."""
    all_metrics = {}
    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    for model_name in models_to_test:
        print(f"\n=== Running {model_name} ===")

        # Create model
        model = ModelFactory.create_model(model_name, device)

        if model_name in ["resnet34", "vgg16"]:
            # Deep learning models
            model, metrics = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=30,
                device=device,
                model_name=f"{model_name}_{baseline_name}",  # Add baseline name to model save
            )
            
            # Evaluate on test set
            test_metrics = evaluate_model(model, test_loader, nn.CrossEntropyLoss(), device)
            metrics.update(test_metrics)

        else:
            # Traditional ML models
            X_train, y_train = convert_dataloader_to_numpy(train_loader)
            X_test, y_test = convert_dataloader_to_numpy(test_loader)

            model.fit(X_train, y_train)
            metrics = evaluate_traditional_ml_model(model, X_test, y_test, model_name)
            
        # Print preview of metrics
        print(f"\nMetrics for {model_name}:")
        print(f"- Test Accuracy: {metrics['accuracy']:.2f}%")
        if "macro_auprc" in metrics:
            print(f"- Macro AUPRC: {metrics['macro_auprc']:.4f}")
        if "train_acc" in metrics:
            print(f"- Final train accuracy: {metrics['train_acc'][-1]:.2f}%")
            print(f"- Final validation accuracy: {metrics['val_acc'][-1]:.2f}%")

        # Store metrics
        all_metrics[model_name] = metrics
        save_metrics(metrics, f"{model_name}_{baseline_name}")  # Add baseline name to metrics save

    return all_metrics


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

    # Run experiments
    models_to_test = ["logistic", "randomforest", "xgboost", "resnet34", "vgg16"]

    print("\n=== Running Baseline 1: Hard Labels ===")
    print(
        f"- Training with {len(combined_hard_loader.dataset)} samples ({len(augment_dataset)} hard + {len(train_dataset)} hard)"
    )
    print(f"- Validation with {len(val_loader.dataset)} samples")
    print(f"- Testing with {len(test_loader.dataset)} samples")

    baseline1_metrics = run_baseline_experiments(
        combined_hard_loader, val_loader, test_loader, device, models_to_test, "baseline1"
    )

    print("\n=== Running Baseline 2: Mixed Labels ===")
    print(
        f"- Training with {len(combined_mixed_loader.dataset)} samples ({len(hard_label_dataset)} hard + {len(soft_label_dataset)} soft)"
    )
    print(f"- Validation with {len(val_loader.dataset)} samples")
    print(f"- Testing with {len(test_loader.dataset)} samples")

    baseline2_metrics = run_baseline_experiments(
        combined_mixed_loader, val_loader, test_loader, device, models_to_test, "baseline2"
    )

    # Plot comparative curves
    plot_comparative_curves(baseline1_metrics, title="Baseline 1 - Hard Labels")
    plot_comparative_curves(baseline2_metrics, title="Baseline 2 - Mixed Labels")

    # Print final results
    print("\n=== Final Results ===")
    print("\nBaseline 1 (Hard Labels):")
    for model_name, metrics in baseline1_metrics.items():
        print(f"\n{model_name}:")
        print(f"Test Accuracy: {metrics['accuracy']}%")
        print(f"Macro AUPRC: {metrics.get('macro_auprc', 'N/A')}")

    print("\nBaseline 2 (Mixed Labels):")
    for model_name, metrics in baseline2_metrics.items():
        print(f"\n{model_name}:")
        print(f"Test Accuracy: {metrics['accuracy']}%")
        print(f"Macro AUPRC: {metrics.get('macro_auprc', 'N/A')}")

"""
Output

"""
