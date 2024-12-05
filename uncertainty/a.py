"""
This file contains the code to run experiments with artificial soft labels.

The experiment is:
    * Train a soft label predictor model on CIFAR-10H
    * Generate artificial soft labels for CIFAR-10
    * Train a model on CIFAR-10 with the artificial soft labels + CIFAR-10H
    * Evaluate the model on CIFAR-10
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import datasets, transforms, models
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import random

from generate_soft_labels import create_soft_label_dataset
from soft_label_predictor import ImageHardToSoftLabelModel

def get_device() -> torch.device:
    """Get the appropriate device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_cifar10_experiment() -> Tuple[Dataset, Dataset, Dataset, Dataset]:
    """Load and split CIFAR-10 dataset into augment, train, test and validation sets."""
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    full_dataset = datasets.CIFAR10(root="../data/cifar-10", train=True, download=True, transform=transform)
    # Use test dataset for training, similar to CIFAR-10H experiment
    train_dataset = datasets.CIFAR10(root="../data/cifar-10", train=False, download=True, transform=transform)

    # Split full dataset for augmenting, testing, and validation
    augment_size = int(0.7 * len(full_dataset))
    val_size = (len(full_dataset) - augment_size) // 2
    test_size = len(full_dataset) - augment_size - val_size

    generator = torch.Generator().manual_seed(229)
    augment_dataset, test_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [augment_size, test_size, val_size], generator=generator
    )

    return augment_dataset, train_dataset, test_dataset, val_dataset

class CIFAR10LabelDataset(Dataset):
    """Dataset wrapper that handles both hard and soft labels consistently."""
    def __init__(self, dataset: Dataset, soft_labels: Optional[np.ndarray] = None):
        self.dataset = dataset
        self.soft_labels = soft_labels

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, label = self.dataset[idx]
        if self.soft_labels is None:
            # Convert hard labels to one-hot
            label = F.one_hot(torch.tensor(label), num_classes=10).float()
        else:
            label = torch.tensor(self.soft_labels[idx])
        return image, label
    
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 20,
    model_path: Optional[str] = None,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Train a neural network model and save the best version based on validation accuracy."""

    # Adjust the final layer for CIFAR-10 and add dropout
    if isinstance(model, models.ResNet):
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.1),  # Add dropout before final layer
            nn.Linear(num_ftrs, 10)
        )
    elif isinstance(model, models.VGG):
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(0.1),  # Add dropout before final layer
            nn.Linear(num_ftrs, 10)
        )
        
    print(f"Training on {device}")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    best_val_acc = 0.0

    print(f"\nTraining {model.__class__.__name__}...")
    
    # Initialize history dictionary
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            
            # Handle soft labels for training accuracy
            train_labels = labels
            if len(labels.shape) > 1:
                _, train_labels = torch.max(labels, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_train += train_labels.size(0)
            correct_train += (predicted == train_labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train

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

        # Store metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(accuracy)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Train Acc: {train_acc:.2f}%, "
            f"Val Acc: {accuracy:.2f}%"
        )

        # Save model if validation accuracy improves
        if model_path is not None and accuracy > best_val_acc:
            best_val_acc = accuracy
            torch.save(model.state_dict(), model_path)
            print(f"Saved model with improved validation accuracy: {accuracy:.2f}%")

    return model, history


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict:
    """Evaluate model performance with multiple metrics."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    total = correct = total_loss = total_cross_entropy = total_kl_div = 0
    all_preds, all_labels = [], []
    all_pred_probs, all_true_probs = [], []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            original_labels = labels.clone()
            
            if len(labels.shape) > 1:
                _, labels = torch.max(labels, 1)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            pred_probs = F.softmax(outputs, dim=1)
            log_probs = F.log_softmax(outputs, dim=1)
            
            # Handle soft vs hard labels
            if len(original_labels.shape) > 1:
                original_labels = original_labels.to(device)
                cross_entropy = -(original_labels * log_probs).sum()
                total_cross_entropy += cross_entropy.item()
                kl_div = F.kl_div(log_probs, original_labels, reduction='sum')
                all_true_probs.extend(original_labels.cpu().numpy())
            else:
                total_cross_entropy += loss.item()
                true_probs = F.one_hot(labels, num_classes=outputs.size(1)).float()
                kl_div = F.kl_div(log_probs, true_probs, reduction='sum')
                all_true_probs.extend(true_probs.cpu().numpy())
            
            total_kl_div += kl_div.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_pred_probs.extend(pred_probs.cpu().numpy())

    # Calculate metrics
    metrics = {
        'accuracy': correct / total,
        'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'loss': total_loss / total,  # Per-sample loss
        'cross_entropy': total_cross_entropy / total,  # Per-sample cross entropy
        'kl_divergence': total_kl_div / total,  # Per-sample KL divergence
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
        'true_labels': np.array(all_labels),
        'predictions': np.array(all_preds),
        'pred_probabilities': np.array(all_pred_probs),
        'true_probabilities': np.array(all_true_probs)
    }
    
    return metrics

# Load CIFAR-10H soft labels
cifar10h_probs_path = "../data/cifar-10h/cifar10h-probs.npy"
cifar10h_probs = np.load(cifar10h_probs_path).astype(np.float32)

def run_proportion_experiment_with_cifar10h(
    cifar10_hard_train_dataset: Dataset,
    cifar10_hard_augment_dataset: Dataset,
    cifar10h_probs: np.ndarray,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    soft_proportions: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    num_epochs: int = 20,
) -> Tuple[Dict, Dict]:
    """Run experiments with different proportions of CIFAR-10H soft labels."""
    results = {}
    histories = {}

    total_samples = 10000  # Always use 10,000 samples

    for prop in soft_proportions:
        print(f"\nRunning experiment with {int(prop*100)}% CIFAR-10H soft labels")
        model_path = f"models/ResNet_cifar10h_{int(prop*100)}percent.pth"

        # Determine the number of samples to use from CIFAR-10H
        soft_size = int(total_samples * prop)
        hard_size = total_samples - soft_size

        # Create datasets with soft labels from CIFAR-10H
        cifar10_soft_label_dataset = CIFAR10LabelDataset(
            Subset(cifar10_hard_train_dataset, range(soft_size)), 
            cifar10h_probs[:soft_size]
        )
        
        # Create dataset with hard labels from CIFAR-10 augment dataset
        cifar10_hard_label_dataset = CIFAR10LabelDataset(
            Subset(cifar10_hard_augment_dataset, range(hard_size))
        )

        # Combine datasets
        combined_train_dataset = ConcatDataset([cifar10_hard_label_dataset, cifar10_soft_label_dataset])
        train_loader = DataLoader(combined_train_dataset, batch_size=128, shuffle=True)
        
        print(f"Training on {len(train_loader.dataset)} samples: {len(cifar10_hard_label_dataset)} hard and {len(cifar10_soft_label_dataset)} soft labels")

        # Train and evaluate model
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        model, history = train_model(
            model, train_loader, val_loader, num_epochs=num_epochs, model_path=model_path, device=device
        )
        histories[prop] = history

        model.load_state_dict(torch.load(model_path, weights_only=True))
        model = model.to(device)
        model.eval()

        # Collect metrics
        results[prop] = {
            **{f'train_{k}': v for k, v in evaluate_model(model, train_loader, device).items()},
            **{f'val_{k}': v for k, v in evaluate_model(model, val_loader, device).items()},
            **{f'test_{k}': v for k, v in evaluate_model(model, test_loader, device).items()}
        }

    return results, histories


def plot_basic_metrics(results: Dict):
    """Plot basic metrics like accuracy, loss etc."""
    metrics = [
        ('cross_entropy', 'Cross Entropy Loss'),
        ('kl_divergence', 'KL Divergence'),
        ('accuracy', 'Accuracy'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('f1', 'F1 Score')
    ]

    proportions = list(results.keys())
    x_axis = [p*100 for p in proportions]

    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    axes = axes.flatten()

    for i, (metric, title) in enumerate(metrics):
        train_metric = [results[p][f'train_{metric}'] * 100 for p in proportions]
        val_metric = [results[p][f'val_{metric}'] * 100 for p in proportions]
        test_metric = [results[p][f'test_{metric}'] * 100 for p in proportions]
        
        axes[i].plot(x_axis, train_metric, 'b-o', label='Training')
        axes[i].plot(x_axis, val_metric, 'r-o', label='Validation')
        axes[i].plot(x_axis, test_metric, 'g-o', label='Test')
        axes[i].set_xlabel('Percentage of Soft Labels')
        axes[i].set_ylabel(f'{title} (%)')
        axes[i].set_title(f'{title} vs Proportion of Soft Labels')
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    plt.savefig('results/metrics.png')
    plt.show()

def plot_confusion_matrices(results: Dict):
    """Plot confusion matrices for different proportions of soft labels."""
    proportions = list(results.keys())
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for i, prop in enumerate(proportions):
        cm = results[prop]['test_confusion_matrix']  # Use test set confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', ax=axes[i], cmap='Blues')
        axes[i].set_title(f'Normalized Confusion Matrix ({int(prop*100)}% Soft Labels)')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')

    plt.tight_layout()
    plt.savefig('results/confusion_matrices.png')
    plt.show()

def plot_roc_pr_curves(results: Dict):
    """Plot ROC and Precision-Recall curves."""
    proportions = list(results.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for prop in proportions:
        test_true = results[prop]['test_true_labels']
        test_probs = results[prop]['test_pred_probabilities']
        
        # One-vs-Rest ROC curves
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        # Convert to one-hot format
        test_true_bin = np.eye(10)[test_true]
        
        for class_idx in range(10):
            fpr[class_idx], tpr[class_idx], _ = roc_curve(
                test_true_bin[:, class_idx], test_probs[:, class_idx])
            roc_auc[class_idx] = auc(fpr[class_idx], tpr[class_idx])
        
        # Compute micro-average ROC curve and ROC area
        fpr_micro, tpr_micro, _ = roc_curve(test_true_bin.ravel(), test_probs.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        
        ax1.plot(fpr_micro, tpr_micro, 
                label=f'{int(prop*100)}% Soft Labels (AUC = {roc_auc_micro:.2f})')
        
        # PR curve
        precision = dict()
        recall = dict()
        pr_auc = dict()
        
        for class_idx in range(10):
            precision[class_idx], recall[class_idx], _ = precision_recall_curve(
                test_true_bin[:, class_idx], test_probs[:, class_idx])
            pr_auc[class_idx] = auc(recall[class_idx], precision[class_idx])
        
        # Compute micro-average PR curve
        precision_micro, recall_micro, _ = precision_recall_curve(
            test_true_bin.ravel(), test_probs.ravel())
        pr_auc_micro = auc(recall_micro, precision_micro)
        
        ax2.plot(recall_micro, precision_micro,
                label=f'{int(prop*100)}% Soft Labels (AUC = {pr_auc_micro:.2f})')

    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves (Micro-averaged)')
    ax1.legend()
    ax1.grid(True)

    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curves (Micro-averaged)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('results/roc_pr_curves.png')
    plt.show()
    
def plot_training_curves(histories: Dict[float, Dict[str, List[float]]]):
    """Plot training curves for different soft label proportions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    for prop, history in histories.items():
        label = f"{int(prop*100)}% Soft Labels"
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        ax1.plot(epochs, history['train_loss'], '--', label=f'Train ({label})')
        ax1.plot(epochs, history['val_loss'], '-', label=f'Val ({label})')
        
        # Accuracy curves
        ax2.plot(epochs, history['train_acc'], '--', label=f'Train ({label})')
        ax2.plot(epochs, history['val_acc'], '-', label=f'Val ({label})')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/training_curves.png')
    plt.show()
    
# Load datasets
cifar10_datasets = load_cifar10_experiment()
cifar10_hard_augment_dataset, cifar10_hard_train_dataset, cifar10_hard_test_dataset, cifar10_hard_val_dataset = cifar10_datasets

# Create data loaders
cifar10_hard_test_loader = DataLoader(cifar10_hard_test_dataset, batch_size=128, shuffle=False)
cifar10_hard_val_loader = DataLoader(cifar10_hard_val_dataset, batch_size=128, shuffle=False)

print(
    f"CIFAR-10 dataset loaded with "
    f"{len(cifar10_hard_augment_dataset)} augment, "
    f"{len(cifar10_hard_test_dataset)} test, and "
    f"{len(cifar10_hard_val_dataset)} validation samples"
)

# Load and prepare soft label model
device = get_device()
soft_label_model = ImageHardToSoftLabelModel().to(device)
soft_label_model.load_state_dict(torch.load("models/soft_label_model.pt", weights_only=True))
soft_label_model.eval()



# Run experiments with CIFAR-10H soft labels
results, histories = run_proportion_experiment_with_cifar10h(
    cifar10_hard_train_dataset=cifar10_hard_train_dataset,
    cifar10_hard_augment_dataset=cifar10_hard_augment_dataset,
    cifar10h_probs=cifar10h_probs,
    val_loader=cifar10_hard_val_loader,
    test_loader=cifar10_hard_test_loader,
    soft_proportions=[0.0, 0.25, 0.5, 0.75, 1.0],
    num_epochs=50,
    device=device
)

# Display results
df = pd.DataFrame(results).T * 100
df.index = [f"{idx:.0f}%" for idx in df.index * 100]
print("\nResults by Soft Label Percentage:")
print("================================")
df[['train_accuracy', 'val_accuracy', 'test_accuracy', 'train_loss', 'val_loss', 'train_cross_entropy', 'val_cross_entropy', 'train_kl_divergence', 'val_kl_divergence']]

plot_training_curves(histories)
plot_basic_metrics(results)
plot_confusion_matrices(results)
plot_roc_pr_curves(results)
plot_roc_pr_curves(results)
plot_roc_pr_curves(results)
plot_roc_pr_curves(results)

"""
(cs229_project) ubuntu@ip-172-31-41-43:~/cs229-final/uncertainty$ python a.py
Files already downloaded and verified
Files already downloaded and verified
CIFAR-10 dataset loaded with 35000 augment, 7500 test, and 7500 validation samples

Running experiment with 0% CIFAR-10H soft labels
Training on 10000 samples: 10000 hard and 0 soft labels
Training on cuda

Training ResNet...
Epoch [1/50] Train Loss: 1.4471, Val Loss: 1.3280, Train Acc: 48.82%, Val Acc: 54.76%
Saved model with improved validation accuracy: 54.76%
Epoch [2/50] Train Loss: 1.0414, Val Loss: 1.1046, Train Acc: 64.23%, Val Acc: 62.60%
Saved model with improved validation accuracy: 62.60%
Epoch [3/50] Train Loss: 0.9103, Val Loss: 1.6680, Train Acc: 68.95%, Val Acc: 54.04%
Epoch [4/50] Train Loss: 0.8323, Val Loss: 0.8709, Train Acc: 70.99%, Val Acc: 69.71%
Saved model with improved validation accuracy: 69.71%
Epoch [5/50] Train Loss: 0.7581, Val Loss: 0.9941, Train Acc: 74.05%, Val Acc: 67.68%
Epoch [6/50] Train Loss: 0.7113, Val Loss: 1.1835, Train Acc: 75.62%, Val Acc: 60.75%
Epoch [7/50] Train Loss: 0.6809, Val Loss: 0.8261, Train Acc: 76.57%, Val Acc: 72.35%
Saved model with improved validation accuracy: 72.35%
Epoch [8/50] Train Loss: 0.6413, Val Loss: 1.4510, Train Acc: 78.15%, Val Acc: 58.59%
Epoch [9/50] Train Loss: 0.5940, Val Loss: 0.8930, Train Acc: 79.28%, Val Acc: 71.21%
Epoch [10/50] Train Loss: 0.5648, Val Loss: 0.7717, Train Acc: 80.74%, Val Acc: 74.41%
Saved model with improved validation accuracy: 74.41%
Epoch [11/50] Train Loss: 0.5477, Val Loss: 0.9721, Train Acc: 81.01%, Val Acc: 69.09%
Epoch [12/50] Train Loss: 0.5092, Val Loss: 0.8582, Train Acc: 82.47%, Val Acc: 72.44%
Epoch [13/50] Train Loss: 0.4733, Val Loss: 0.7896, Train Acc: 83.65%, Val Acc: 74.68%
Saved model with improved validation accuracy: 74.68%
Epoch [14/50] Train Loss: 0.4256, Val Loss: 0.9190, Train Acc: 85.98%, Val Acc: 71.36%
Epoch [15/50] Train Loss: 0.5011, Val Loss: 0.9194, Train Acc: 83.24%, Val Acc: 71.35%
Epoch [16/50] Train Loss: 0.5617, Val Loss: 1.1286, Train Acc: 80.94%, Val Acc: 66.48%
Epoch [17/50] Train Loss: 0.4656, Val Loss: 0.7712, Train Acc: 84.03%, Val Acc: 75.23%
Saved model with improved validation accuracy: 75.23%
Epoch [18/50] Train Loss: 0.3947, Val Loss: 0.8321, Train Acc: 86.35%, Val Acc: 74.23%
Epoch [19/50] Train Loss: 0.3713, Val Loss: 0.8475, Train Acc: 86.64%, Val Acc: 74.92%
Epoch [20/50] Train Loss: 0.3310, Val Loss: 0.8245, Train Acc: 88.39%, Val Acc: 74.72%
Epoch [21/50] Train Loss: 0.3185, Val Loss: 0.8957, Train Acc: 88.98%, Val Acc: 73.76%
Epoch [22/50] Train Loss: 0.3491, Val Loss: 0.7829, Train Acc: 87.86%, Val Acc: 76.56%
Saved model with improved validation accuracy: 76.56%
Epoch [23/50] Train Loss: 0.2872, Val Loss: 0.8463, Train Acc: 89.83%, Val Acc: 75.60%
Epoch [24/50] Train Loss: 0.2751, Val Loss: 0.8747, Train Acc: 90.22%, Val Acc: 73.83%
Epoch [25/50] Train Loss: 0.2800, Val Loss: 0.7781, Train Acc: 90.34%, Val Acc: 77.44%
Saved model with improved validation accuracy: 77.44%
Epoch [26/50] Train Loss: 0.2714, Val Loss: 0.8050, Train Acc: 90.66%, Val Acc: 76.17%
Epoch [27/50] Train Loss: 0.2779, Val Loss: 0.8099, Train Acc: 90.34%, Val Acc: 76.19%
Epoch [28/50] Train Loss: 0.2490, Val Loss: 0.9665, Train Acc: 91.17%, Val Acc: 73.60%
Epoch [29/50] Train Loss: 0.2620, Val Loss: 1.2188, Train Acc: 91.41%, Val Acc: 67.32%
Epoch [30/50] Train Loss: 0.3642, Val Loss: 0.8135, Train Acc: 87.99%, Val Acc: 76.28%
Epoch [31/50] Train Loss: 0.4394, Val Loss: 2.0063, Train Acc: 85.62%, Val Acc: 52.61%
Epoch [32/50] Train Loss: 0.4076, Val Loss: 0.8520, Train Acc: 86.46%, Val Acc: 74.68%
Epoch [33/50] Train Loss: 0.3021, Val Loss: 0.7631, Train Acc: 90.03%, Val Acc: 77.41%
Epoch [34/50] Train Loss: 0.2439, Val Loss: 0.9043, Train Acc: 91.94%, Val Acc: 75.11%
Epoch [35/50] Train Loss: 0.2248, Val Loss: 0.9192, Train Acc: 92.71%, Val Acc: 75.75%
Epoch [36/50] Train Loss: 0.2219, Val Loss: 0.9427, Train Acc: 92.73%, Val Acc: 75.25%
Epoch [37/50] Train Loss: 0.2369, Val Loss: 1.0251, Train Acc: 92.34%, Val Acc: 72.51%
Epoch [38/50] Train Loss: 0.2219, Val Loss: 0.8580, Train Acc: 93.24%, Val Acc: 77.19%
Epoch [39/50] Train Loss: 0.2230, Val Loss: 0.8728, Train Acc: 92.63%, Val Acc: 76.24%
Epoch [40/50] Train Loss: 0.1738, Val Loss: 0.9214, Train Acc: 94.24%, Val Acc: 75.39%
Epoch [41/50] Train Loss: 0.1784, Val Loss: 0.8545, Train Acc: 94.02%, Val Acc: 77.08%
Epoch [42/50] Train Loss: 0.2203, Val Loss: 0.8154, Train Acc: 92.42%, Val Acc: 77.05%
Epoch [43/50] Train Loss: 0.1546, Val Loss: 0.9902, Train Acc: 95.00%, Val Acc: 75.21%
Epoch [44/50] Train Loss: 0.1882, Val Loss: 0.9431, Train Acc: 93.71%, Val Acc: 76.11%
Epoch [45/50] Train Loss: 0.1342, Val Loss: 0.9167, Train Acc: 95.49%, Val Acc: 76.79%
Epoch [46/50] Train Loss: 0.1351, Val Loss: 0.9233, Train Acc: 95.49%, Val Acc: 76.80%
Epoch [47/50] Train Loss: 0.1427, Val Loss: 1.0067, Train Acc: 95.14%, Val Acc: 75.04%
Epoch [48/50] Train Loss: 0.1548, Val Loss: 0.9605, Train Acc: 94.85%, Val Acc: 76.08%
Epoch [49/50] Train Loss: 0.1579, Val Loss: 0.9735, Train Acc: 95.20%, Val Acc: 76.29%
Epoch [50/50] Train Loss: 0.1437, Val Loss: 0.9410, Train Acc: 95.04%, Val Acc: 77.59%
Saved model with improved validation accuracy: 77.59%

Running experiment with 25% CIFAR-10H soft labels
Training on 10000 samples: 7500 hard and 2500 soft labels
Training on cuda

Training ResNet...
Epoch [1/50] Train Loss: 1.4713, Val Loss: 1.5804, Train Acc: 48.31%, Val Acc: 50.36%
Saved model with improved validation accuracy: 50.36%
Epoch [2/50] Train Loss: 1.0481, Val Loss: 1.2343, Train Acc: 63.86%, Val Acc: 58.37%
Saved model with improved validation accuracy: 58.37%
Epoch [3/50] Train Loss: 0.9108, Val Loss: 1.0393, Train Acc: 69.18%, Val Acc: 64.80%
Saved model with improved validation accuracy: 64.80%
Epoch [4/50] Train Loss: 0.8783, Val Loss: 0.9893, Train Acc: 70.43%, Val Acc: 66.28%
Saved model with improved validation accuracy: 66.28%
Epoch [5/50] Train Loss: 0.8056, Val Loss: 1.0790, Train Acc: 73.13%, Val Acc: 63.36%
Epoch [6/50] Train Loss: 0.7455, Val Loss: 0.8690, Train Acc: 74.66%, Val Acc: 70.16%
Saved model with improved validation accuracy: 70.16%
Epoch [7/50] Train Loss: 0.6787, Val Loss: 0.9215, Train Acc: 77.03%, Val Acc: 70.01%
Epoch [8/50] Train Loss: 0.6409, Val Loss: 0.9839, Train Acc: 79.12%, Val Acc: 68.27%
Epoch [9/50] Train Loss: 0.6635, Val Loss: 1.0396, Train Acc: 78.51%, Val Acc: 67.41%
Epoch [10/50] Train Loss: 0.6561, Val Loss: 1.1786, Train Acc: 78.31%, Val Acc: 63.89%
Epoch [11/50] Train Loss: 0.6444, Val Loss: 0.8628, Train Acc: 79.28%, Val Acc: 71.25%
Saved model with improved validation accuracy: 71.25%
Epoch [12/50] Train Loss: 0.5484, Val Loss: 0.8450, Train Acc: 81.91%, Val Acc: 72.27%
Saved model with improved validation accuracy: 72.27%
Epoch [13/50] Train Loss: 0.5046, Val Loss: 0.8192, Train Acc: 83.62%, Val Acc: 72.79%
Saved model with improved validation accuracy: 72.79%
Epoch [14/50] Train Loss: 0.4995, Val Loss: 0.7928, Train Acc: 83.76%, Val Acc: 73.29%
Saved model with improved validation accuracy: 73.29%
Epoch [15/50] Train Loss: 0.4877, Val Loss: 0.8330, Train Acc: 84.53%, Val Acc: 73.63%
Saved model with improved validation accuracy: 73.63%
Epoch [16/50] Train Loss: 0.5124, Val Loss: 0.8500, Train Acc: 83.46%, Val Acc: 72.84%
Epoch [17/50] Train Loss: 0.4546, Val Loss: 0.8185, Train Acc: 85.59%, Val Acc: 73.87%
Saved model with improved validation accuracy: 73.87%
Epoch [18/50] Train Loss: 0.4579, Val Loss: 0.7874, Train Acc: 85.15%, Val Acc: 74.28%
Saved model with improved validation accuracy: 74.28%
Epoch [19/50] Train Loss: 0.4045, Val Loss: 0.7898, Train Acc: 87.40%, Val Acc: 74.89%
Saved model with improved validation accuracy: 74.89%
Epoch [20/50] Train Loss: 0.4372, Val Loss: 1.0894, Train Acc: 86.13%, Val Acc: 68.25%
Epoch [21/50] Train Loss: 0.4001, Val Loss: 0.8594, Train Acc: 87.56%, Val Acc: 74.20%
Epoch [22/50] Train Loss: 0.3572, Val Loss: 0.8149, Train Acc: 88.93%, Val Acc: 74.67%
Epoch [23/50] Train Loss: 0.3447, Val Loss: 0.8089, Train Acc: 89.59%, Val Acc: 74.80%
Epoch [24/50] Train Loss: 0.3360, Val Loss: 0.7783, Train Acc: 89.75%, Val Acc: 75.71%
Saved model with improved validation accuracy: 75.71%
Epoch [25/50] Train Loss: 0.3485, Val Loss: 0.9719, Train Acc: 89.12%, Val Acc: 71.55%
Epoch [26/50] Train Loss: 0.3587, Val Loss: 0.9750, Train Acc: 88.73%, Val Acc: 71.60%
Epoch [27/50] Train Loss: 0.3300, Val Loss: 1.0387, Train Acc: 90.50%, Val Acc: 71.20%
Epoch [28/50] Train Loss: 0.5343, Val Loss: 0.8311, Train Acc: 82.64%, Val Acc: 73.24%
Epoch [29/50] Train Loss: 0.3479, Val Loss: 0.8332, Train Acc: 89.32%, Val Acc: 74.64%
Epoch [30/50] Train Loss: 0.2994, Val Loss: 0.8356, Train Acc: 91.10%, Val Acc: 75.04%
Epoch [31/50] Train Loss: 0.2909, Val Loss: 0.8736, Train Acc: 91.56%, Val Acc: 74.56%
Epoch [32/50] Train Loss: 0.2983, Val Loss: 0.8068, Train Acc: 91.91%, Val Acc: 76.29%
Saved model with improved validation accuracy: 76.29%
Epoch [33/50] Train Loss: 0.3494, Val Loss: 4.3226, Train Acc: 89.78%, Val Acc: 47.03%
Epoch [34/50] Train Loss: 0.3454, Val Loss: 0.8447, Train Acc: 89.40%, Val Acc: 75.49%
Epoch [35/50] Train Loss: 0.2929, Val Loss: 0.9670, Train Acc: 91.41%, Val Acc: 72.57%
Epoch [36/50] Train Loss: 0.3253, Val Loss: 1.0658, Train Acc: 90.49%, Val Acc: 69.91%
Epoch [37/50] Train Loss: 0.3221, Val Loss: 0.8484, Train Acc: 90.65%, Val Acc: 75.36%
Epoch [38/50] Train Loss: 0.3213, Val Loss: 0.9078, Train Acc: 90.62%, Val Acc: 73.91%
Epoch [39/50] Train Loss: 0.3007, Val Loss: 0.7805, Train Acc: 91.23%, Val Acc: 76.97%
Saved model with improved validation accuracy: 76.97%
Epoch [40/50] Train Loss: 0.2681, Val Loss: 0.8568, Train Acc: 92.95%, Val Acc: 76.19%
Epoch [41/50] Train Loss: 0.2484, Val Loss: 0.7681, Train Acc: 93.07%, Val Acc: 77.75%
Saved model with improved validation accuracy: 77.75%
Epoch [42/50] Train Loss: 0.2427, Val Loss: 0.7807, Train Acc: 93.31%, Val Acc: 77.81%
Saved model with improved validation accuracy: 77.81%
Epoch [43/50] Train Loss: 0.2195, Val Loss: 0.8168, Train Acc: 94.21%, Val Acc: 77.09%
Epoch [44/50] Train Loss: 0.2476, Val Loss: 0.8350, Train Acc: 93.30%, Val Acc: 76.45%
Epoch [45/50] Train Loss: 0.2384, Val Loss: 0.8481, Train Acc: 93.59%, Val Acc: 76.01%
Epoch [46/50] Train Loss: 0.2213, Val Loss: 0.7776, Train Acc: 94.15%, Val Acc: 78.05%
Saved model with improved validation accuracy: 78.05%
Epoch [47/50] Train Loss: 0.1922, Val Loss: 0.8090, Train Acc: 95.09%, Val Acc: 76.64%
Epoch [48/50] Train Loss: 0.1789, Val Loss: 0.7789, Train Acc: 95.54%, Val Acc: 77.97%
Epoch [49/50] Train Loss: 0.1730, Val Loss: 0.8432, Train Acc: 96.07%, Val Acc: 76.72%
Epoch [50/50] Train Loss: 0.1938, Val Loss: 0.7804, Train Acc: 94.96%, Val Acc: 77.73%

Running experiment with 50% CIFAR-10H soft labels
Training on 10000 samples: 5000 hard and 5000 soft labels
Training on cuda

Training ResNet...
Epoch [1/50] Train Loss: 1.4610, Val Loss: 1.3696, Train Acc: 48.22%, Val Acc: 53.92%
Saved model with improved validation accuracy: 53.92%
Epoch [2/50] Train Loss: 1.0707, Val Loss: 1.8058, Train Acc: 64.11%, Val Acc: 42.65%
Epoch [3/50] Train Loss: 1.0323, Val Loss: 1.1512, Train Acc: 65.20%, Val Acc: 61.45%
Saved model with improved validation accuracy: 61.45%
Epoch [4/50] Train Loss: 0.9130, Val Loss: 1.1280, Train Acc: 69.72%, Val Acc: 61.15%
Epoch [5/50] Train Loss: 0.8228, Val Loss: 0.9922, Train Acc: 73.55%, Val Acc: 67.31%
Saved model with improved validation accuracy: 67.31%
Epoch [6/50] Train Loss: 0.7542, Val Loss: 1.3307, Train Acc: 75.67%, Val Acc: 59.99%
Epoch [7/50] Train Loss: 0.7381, Val Loss: 1.0262, Train Acc: 76.80%, Val Acc: 66.60%
Epoch [8/50] Train Loss: 0.7140, Val Loss: 1.0469, Train Acc: 77.22%, Val Acc: 65.25%
Epoch [9/50] Train Loss: 0.7266, Val Loss: 0.9888, Train Acc: 76.89%, Val Acc: 67.93%
Saved model with improved validation accuracy: 67.93%
Epoch [10/50] Train Loss: 0.6856, Val Loss: 1.0755, Train Acc: 78.35%, Val Acc: 64.68%
Epoch [11/50] Train Loss: 0.6401, Val Loss: 0.7910, Train Acc: 79.94%, Val Acc: 73.04%
Saved model with improved validation accuracy: 73.04%
Epoch [12/50] Train Loss: 0.5906, Val Loss: 0.8306, Train Acc: 81.62%, Val Acc: 72.55%
Epoch [13/50] Train Loss: 0.6157, Val Loss: 0.8415, Train Acc: 80.67%, Val Acc: 71.87%
Epoch [14/50] Train Loss: 0.5607, Val Loss: 0.9673, Train Acc: 82.68%, Val Acc: 69.60%
Epoch [15/50] Train Loss: 0.5627, Val Loss: 0.7875, Train Acc: 82.87%, Val Acc: 74.52%
Saved model with improved validation accuracy: 74.52%
Epoch [16/50] Train Loss: 0.5240, Val Loss: 0.8154, Train Acc: 84.53%, Val Acc: 73.65%
Epoch [17/50] Train Loss: 0.5115, Val Loss: 0.7691, Train Acc: 84.81%, Val Acc: 74.48%
Epoch [18/50] Train Loss: 0.4769, Val Loss: 0.7832, Train Acc: 85.73%, Val Acc: 75.16%
Saved model with improved validation accuracy: 75.16%
Epoch [19/50] Train Loss: 0.4468, Val Loss: 0.7465, Train Acc: 86.82%, Val Acc: 75.71%
Saved model with improved validation accuracy: 75.71%
Epoch [20/50] Train Loss: 0.4449, Val Loss: 0.8772, Train Acc: 86.93%, Val Acc: 71.73%
Epoch [21/50] Train Loss: 0.4512, Val Loss: 0.7912, Train Acc: 86.97%, Val Acc: 74.91%
Epoch [22/50] Train Loss: 0.4152, Val Loss: 1.0651, Train Acc: 88.22%, Val Acc: 67.48%
Epoch [23/50] Train Loss: 0.4390, Val Loss: 0.7737, Train Acc: 87.45%, Val Acc: 74.97%
Epoch [24/50] Train Loss: 0.3896, Val Loss: 0.7790, Train Acc: 89.20%, Val Acc: 74.51%
Epoch [25/50] Train Loss: 0.3782, Val Loss: 0.7646, Train Acc: 89.87%, Val Acc: 75.20%
Epoch [26/50] Train Loss: 0.3987, Val Loss: 0.7345, Train Acc: 88.97%, Val Acc: 77.45%
Saved model with improved validation accuracy: 77.45%
Epoch [27/50] Train Loss: 0.3615, Val Loss: 0.8259, Train Acc: 90.57%, Val Acc: 74.80%
Epoch [28/50] Train Loss: 0.3707, Val Loss: 0.7327, Train Acc: 89.90%, Val Acc: 76.43%
Epoch [29/50] Train Loss: 0.3508, Val Loss: 0.7718, Train Acc: 90.87%, Val Acc: 76.32%
Epoch [30/50] Train Loss: 0.3481, Val Loss: 0.9289, Train Acc: 90.77%, Val Acc: 72.04%
Epoch [31/50] Train Loss: 0.3633, Val Loss: 0.7289, Train Acc: 90.30%, Val Acc: 76.81%
Epoch [32/50] Train Loss: 0.3241, Val Loss: 0.7584, Train Acc: 91.94%, Val Acc: 77.40%
Epoch [33/50] Train Loss: 0.3374, Val Loss: 0.7760, Train Acc: 91.04%, Val Acc: 76.81%
Epoch [34/50] Train Loss: 0.3061, Val Loss: 0.8112, Train Acc: 92.17%, Val Acc: 75.75%
Epoch [35/50] Train Loss: 0.2834, Val Loss: 0.7384, Train Acc: 93.32%, Val Acc: 77.44%
Epoch [36/50] Train Loss: 0.3150, Val Loss: 0.8119, Train Acc: 92.36%, Val Acc: 76.15%
Epoch [37/50] Train Loss: 0.3175, Val Loss: 0.8045, Train Acc: 91.90%, Val Acc: 75.79%
Epoch [38/50] Train Loss: 0.3105, Val Loss: 0.8673, Train Acc: 92.07%, Val Acc: 75.57%
Epoch [39/50] Train Loss: 0.3217, Val Loss: 0.9252, Train Acc: 91.81%, Val Acc: 72.31%
Epoch [40/50] Train Loss: 0.4519, Val Loss: 0.7662, Train Acc: 86.96%, Val Acc: 75.40%
Epoch [41/50] Train Loss: 0.3339, Val Loss: 0.7766, Train Acc: 91.59%, Val Acc: 76.20%
Epoch [42/50] Train Loss: 0.3035, Val Loss: 0.7766, Train Acc: 92.40%, Val Acc: 76.07%
Epoch [43/50] Train Loss: 0.3224, Val Loss: 0.7645, Train Acc: 91.76%, Val Acc: 77.24%
Epoch [44/50] Train Loss: 0.2571, Val Loss: 0.8138, Train Acc: 94.14%, Val Acc: 76.36%
Epoch [45/50] Train Loss: 0.2745, Val Loss: 0.7340, Train Acc: 93.90%, Val Acc: 77.71%
Saved model with improved validation accuracy: 77.71%
Epoch [46/50] Train Loss: 0.2614, Val Loss: 0.7594, Train Acc: 93.89%, Val Acc: 76.93%
Epoch [47/50] Train Loss: 0.2369, Val Loss: 0.7557, Train Acc: 95.18%, Val Acc: 77.52%
Epoch [48/50] Train Loss: 0.2617, Val Loss: 0.7946, Train Acc: 94.01%, Val Acc: 77.24%
Epoch [49/50] Train Loss: 0.2546, Val Loss: 0.7699, Train Acc: 94.51%, Val Acc: 77.32%
Epoch [50/50] Train Loss: 0.2705, Val Loss: 0.7839, Train Acc: 93.81%, Val Acc: 76.55%

Running experiment with 75% CIFAR-10H soft labels
Training on 10000 samples: 2500 hard and 7500 soft labels
Training on cuda

Training ResNet...
Epoch [1/50] Train Loss: 1.5352, Val Loss: 1.5017, Train Acc: 46.49%, Val Acc: 50.29%
Saved model with improved validation accuracy: 50.29%
Epoch [2/50] Train Loss: 1.1885, Val Loss: 1.1253, Train Acc: 59.78%, Val Acc: 61.15%
Saved model with improved validation accuracy: 61.15%
Epoch [3/50] Train Loss: 0.9734, Val Loss: 0.9699, Train Acc: 67.76%, Val Acc: 66.97%
Saved model with improved validation accuracy: 66.97%
Epoch [4/50] Train Loss: 0.8851, Val Loss: 0.9773, Train Acc: 71.55%, Val Acc: 65.63%
Epoch [5/50] Train Loss: 0.8429, Val Loss: 1.0435, Train Acc: 72.59%, Val Acc: 65.01%
Epoch [6/50] Train Loss: 0.7776, Val Loss: 0.8717, Train Acc: 75.91%, Val Acc: 70.33%
Saved model with improved validation accuracy: 70.33%
Epoch [7/50] Train Loss: 0.7737, Val Loss: 0.8499, Train Acc: 75.38%, Val Acc: 71.85%
Saved model with improved validation accuracy: 71.85%
Epoch [8/50] Train Loss: 0.7256, Val Loss: 0.8001, Train Acc: 76.99%, Val Acc: 73.19%
Saved model with improved validation accuracy: 73.19%
Epoch [9/50] Train Loss: 0.6798, Val Loss: 0.9302, Train Acc: 79.48%, Val Acc: 69.44%
Epoch [10/50] Train Loss: 0.6907, Val Loss: 0.7712, Train Acc: 78.46%, Val Acc: 74.04%
Saved model with improved validation accuracy: 74.04%
Epoch [11/50] Train Loss: 0.6148, Val Loss: 0.8107, Train Acc: 81.77%, Val Acc: 72.67%
Epoch [12/50] Train Loss: 0.5930, Val Loss: 0.8402, Train Acc: 82.41%, Val Acc: 71.87%
Epoch [13/50] Train Loss: 0.6175, Val Loss: 0.8360, Train Acc: 81.57%, Val Acc: 71.41%
Epoch [14/50] Train Loss: 0.7519, Val Loss: 0.8962, Train Acc: 76.85%, Val Acc: 70.12%
Epoch [15/50] Train Loss: 0.6151, Val Loss: 0.7548, Train Acc: 81.40%, Val Acc: 74.36%
Saved model with improved validation accuracy: 74.36%
Epoch [16/50] Train Loss: 0.5631, Val Loss: 1.0727, Train Acc: 83.26%, Val Acc: 67.37%
Epoch [17/50] Train Loss: 0.5699, Val Loss: 0.7448, Train Acc: 83.49%, Val Acc: 75.12%
Saved model with improved validation accuracy: 75.12%
Epoch [18/50] Train Loss: 0.5269, Val Loss: 0.8099, Train Acc: 85.21%, Val Acc: 72.83%
Epoch [19/50] Train Loss: 0.4938, Val Loss: 0.8137, Train Acc: 86.52%, Val Acc: 73.03%
Epoch [20/50] Train Loss: 0.4597, Val Loss: 0.7824, Train Acc: 87.49%, Val Acc: 74.76%
Epoch [21/50] Train Loss: 0.4610, Val Loss: 0.7488, Train Acc: 87.49%, Val Acc: 75.73%
Saved model with improved validation accuracy: 75.73%
Epoch [22/50] Train Loss: 0.4457, Val Loss: 0.7535, Train Acc: 88.28%, Val Acc: 75.43%
Epoch [23/50] Train Loss: 0.4756, Val Loss: 0.7554, Train Acc: 87.64%, Val Acc: 74.89%
Epoch [24/50] Train Loss: 0.5091, Val Loss: 0.9878, Train Acc: 85.87%, Val Acc: 69.97%
Epoch [25/50] Train Loss: 1.0562, Val Loss: 1.1387, Train Acc: 65.02%, Val Acc: 63.21%
Epoch [26/50] Train Loss: 0.7312, Val Loss: 0.8459, Train Acc: 77.64%, Val Acc: 71.16%
Epoch [27/50] Train Loss: 0.6497, Val Loss: 0.7635, Train Acc: 80.66%, Val Acc: 74.69%
Epoch [28/50] Train Loss: 0.5450, Val Loss: 0.7941, Train Acc: 84.18%, Val Acc: 74.29%
Epoch [29/50] Train Loss: 0.5262, Val Loss: 1.0320, Train Acc: 85.40%, Val Acc: 68.92%
Epoch [30/50] Train Loss: 0.5049, Val Loss: 0.7814, Train Acc: 85.62%, Val Acc: 74.37%
Epoch [31/50] Train Loss: 0.4332, Val Loss: 0.7574, Train Acc: 88.46%, Val Acc: 74.92%
Epoch [32/50] Train Loss: 0.3996, Val Loss: 0.6917, Train Acc: 89.76%, Val Acc: 77.47%
Saved model with improved validation accuracy: 77.47%
Epoch [33/50] Train Loss: 0.4029, Val Loss: 0.7103, Train Acc: 90.33%, Val Acc: 77.81%
Saved model with improved validation accuracy: 77.81%
Epoch [34/50] Train Loss: 0.4113, Val Loss: 0.7313, Train Acc: 89.55%, Val Acc: 77.19%
Epoch [35/50] Train Loss: 0.3935, Val Loss: 0.7172, Train Acc: 90.71%, Val Acc: 77.61%
Epoch [36/50] Train Loss: 0.3765, Val Loss: 0.7072, Train Acc: 90.93%, Val Acc: 77.45%
Epoch [37/50] Train Loss: 0.3556, Val Loss: 0.7373, Train Acc: 91.95%, Val Acc: 76.89%
Epoch [38/50] Train Loss: 0.3442, Val Loss: 0.7849, Train Acc: 92.24%, Val Acc: 75.89%
Epoch [39/50] Train Loss: 0.3841, Val Loss: 0.8081, Train Acc: 91.03%, Val Acc: 75.35%
Epoch [40/50] Train Loss: 0.3325, Val Loss: 0.7592, Train Acc: 93.02%, Val Acc: 76.03%
Epoch [41/50] Train Loss: 0.3216, Val Loss: 0.7656, Train Acc: 93.39%, Val Acc: 76.01%
Epoch [42/50] Train Loss: 0.3226, Val Loss: 0.7631, Train Acc: 93.21%, Val Acc: 76.25%
Epoch [43/50] Train Loss: 0.3194, Val Loss: 0.7708, Train Acc: 93.37%, Val Acc: 76.49%
Epoch [44/50] Train Loss: 0.3580, Val Loss: 0.8579, Train Acc: 91.86%, Val Acc: 74.73%
Epoch [45/50] Train Loss: 0.3831, Val Loss: 0.8064, Train Acc: 90.80%, Val Acc: 75.00%
Epoch [46/50] Train Loss: 0.3396, Val Loss: 0.7220, Train Acc: 92.49%, Val Acc: 77.73%
Epoch [47/50] Train Loss: 0.3289, Val Loss: 0.7343, Train Acc: 93.50%, Val Acc: 77.33%
Epoch [48/50] Train Loss: 0.3430, Val Loss: 0.7615, Train Acc: 92.78%, Val Acc: 77.13%
Epoch [49/50] Train Loss: 0.5056, Val Loss: 0.8932, Train Acc: 86.01%, Val Acc: 71.83%
Epoch [50/50] Train Loss: 0.3716, Val Loss: 0.8504, Train Acc: 91.16%, Val Acc: 74.15%

Running experiment with 100% CIFAR-10H soft labels
Training on 10000 samples: 0 hard and 10000 soft labels
Training on cuda

Training ResNet...
Epoch [1/50] Train Loss: 1.4888, Val Loss: 1.1693, Train Acc: 48.81%, Val Acc: 58.95%
Saved model with improved validation accuracy: 58.95%
Epoch [2/50] Train Loss: 1.1129, Val Loss: 1.0952, Train Acc: 63.40%, Val Acc: 62.75%
Saved model with improved validation accuracy: 62.75%
Epoch [3/50] Train Loss: 0.9717, Val Loss: 0.9783, Train Acc: 69.16%, Val Acc: 65.51%
Saved model with improved validation accuracy: 65.51%
Epoch [4/50] Train Loss: 0.8664, Val Loss: 0.9019, Train Acc: 72.69%, Val Acc: 68.57%
Saved model with improved validation accuracy: 68.57%
Epoch [5/50] Train Loss: 0.8053, Val Loss: 1.0155, Train Acc: 75.39%, Val Acc: 67.29%
Epoch [6/50] Train Loss: 0.7966, Val Loss: 0.8359, Train Acc: 75.64%, Val Acc: 72.32%
Saved model with improved validation accuracy: 72.32%
Epoch [7/50] Train Loss: 0.7133, Val Loss: 0.8382, Train Acc: 78.93%, Val Acc: 70.76%
Epoch [8/50] Train Loss: 0.7151, Val Loss: 0.8848, Train Acc: 78.63%, Val Acc: 70.08%
Epoch [9/50] Train Loss: 0.6749, Val Loss: 0.8212, Train Acc: 80.34%, Val Acc: 73.01%
Saved model with improved validation accuracy: 73.01%
Epoch [10/50] Train Loss: 0.7033, Val Loss: 0.7952, Train Acc: 79.08%, Val Acc: 74.76%
Saved model with improved validation accuracy: 74.76%
Epoch [11/50] Train Loss: 0.6495, Val Loss: 0.7822, Train Acc: 81.68%, Val Acc: 73.60%
Epoch [12/50] Train Loss: 0.6095, Val Loss: 0.8174, Train Acc: 83.18%, Val Acc: 72.89%
Epoch [13/50] Train Loss: 0.5997, Val Loss: 0.8204, Train Acc: 83.69%, Val Acc: 72.95%
Epoch [14/50] Train Loss: 0.5987, Val Loss: 0.8203, Train Acc: 83.42%, Val Acc: 73.03%
Epoch [15/50] Train Loss: 0.5547, Val Loss: 0.7036, Train Acc: 84.31%, Val Acc: 76.61%
Saved model with improved validation accuracy: 76.61%
Epoch [16/50] Train Loss: 0.5305, Val Loss: 0.7644, Train Acc: 86.12%, Val Acc: 75.51%
Epoch [17/50] Train Loss: 0.5397, Val Loss: 0.7305, Train Acc: 86.28%, Val Acc: 75.68%
Epoch [18/50] Train Loss: 0.5424, Val Loss: 0.9853, Train Acc: 86.00%, Val Acc: 68.63%
Epoch [19/50] Train Loss: 0.5475, Val Loss: 0.7242, Train Acc: 85.32%, Val Acc: 75.61%
Epoch [20/50] Train Loss: 0.4949, Val Loss: 0.7607, Train Acc: 87.28%, Val Acc: 75.73%
Epoch [21/50] Train Loss: 0.4546, Val Loss: 0.7939, Train Acc: 88.98%, Val Acc: 74.20%
Epoch [22/50] Train Loss: 0.4741, Val Loss: 0.7560, Train Acc: 88.03%, Val Acc: 76.16%
Epoch [23/50] Train Loss: 0.4643, Val Loss: 0.6819, Train Acc: 88.37%, Val Acc: 77.47%
Saved model with improved validation accuracy: 77.47%
Epoch [24/50] Train Loss: 0.4384, Val Loss: 0.7106, Train Acc: 89.68%, Val Acc: 77.32%
Epoch [25/50] Train Loss: 0.5008, Val Loss: 0.7432, Train Acc: 87.06%, Val Acc: 75.59%
Epoch [26/50] Train Loss: 0.4303, Val Loss: 0.7207, Train Acc: 89.81%, Val Acc: 76.43%
Epoch [27/50] Train Loss: 0.4356, Val Loss: 0.7134, Train Acc: 89.67%, Val Acc: 76.91%
Epoch [28/50] Train Loss: 0.4309, Val Loss: 0.7332, Train Acc: 89.92%, Val Acc: 76.29%
Epoch [29/50] Train Loss: 0.4038, Val Loss: 0.7037, Train Acc: 91.43%, Val Acc: 77.45%
Epoch [30/50] Train Loss: 0.3937, Val Loss: 0.7845, Train Acc: 91.44%, Val Acc: 75.19%
Epoch [31/50] Train Loss: 0.3991, Val Loss: 0.7085, Train Acc: 91.64%, Val Acc: 77.72%
Saved model with improved validation accuracy: 77.72%
Epoch [32/50] Train Loss: 0.3846, Val Loss: 0.7238, Train Acc: 91.87%, Val Acc: 76.88%
Epoch [33/50] Train Loss: 0.3769, Val Loss: 0.7664, Train Acc: 92.73%, Val Acc: 77.12%
Epoch [34/50] Train Loss: 0.3776, Val Loss: 0.7191, Train Acc: 92.56%, Val Acc: 77.45%
Epoch [35/50] Train Loss: 0.3815, Val Loss: 0.7033, Train Acc: 92.12%, Val Acc: 77.96%
Saved model with improved validation accuracy: 77.96%
Epoch [36/50] Train Loss: 0.3616, Val Loss: 0.7501, Train Acc: 93.31%, Val Acc: 77.12%
Epoch [37/50] Train Loss: 0.3826, Val Loss: 0.8101, Train Acc: 92.10%, Val Acc: 75.67%
Epoch [38/50] Train Loss: 0.3679, Val Loss: 0.7722, Train Acc: 92.84%, Val Acc: 75.56%
Epoch [39/50] Train Loss: 0.3428, Val Loss: 0.6916, Train Acc: 93.56%, Val Acc: 78.36%
Saved model with improved validation accuracy: 78.36%
Epoch [40/50] Train Loss: 0.3584, Val Loss: 0.8020, Train Acc: 93.77%, Val Acc: 75.05%
Epoch [41/50] Train Loss: 0.4170, Val Loss: 0.7487, Train Acc: 90.50%, Val Acc: 76.07%
Epoch [42/50] Train Loss: 0.3705, Val Loss: 0.7253, Train Acc: 93.12%, Val Acc: 77.57%
Epoch [43/50] Train Loss: 0.3729, Val Loss: 0.7701, Train Acc: 92.47%, Val Acc: 76.19%
Epoch [44/50] Train Loss: 0.3435, Val Loss: 0.7185, Train Acc: 93.75%, Val Acc: 77.47%
Epoch [45/50] Train Loss: 0.3173, Val Loss: 0.7256, Train Acc: 95.12%, Val Acc: 77.69%
Epoch [46/50] Train Loss: 0.3205, Val Loss: 0.7930, Train Acc: 94.87%, Val Acc: 76.29%
Epoch [47/50] Train Loss: 0.3572, Val Loss: 0.7164, Train Acc: 92.94%, Val Acc: 77.29%
Epoch [48/50] Train Loss: 0.3404, Val Loss: 0.8134, Train Acc: 94.33%, Val Acc: 75.29%
Epoch [49/50] Train Loss: 0.3459, Val Loss: 0.7158, Train Acc: 93.72%, Val Acc: 78.15%
Epoch [50/50] Train Loss: 0.3159, Val Loss: 0.7230, Train Acc: 95.24%, Val Acc: 77.16%
"""