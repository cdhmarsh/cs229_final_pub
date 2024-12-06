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
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, List
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from generate_soft_labels import create_soft_label_dataset
from soft_label_predictor import ImageHardToSoftLabelModel
import os
import json
from sklearn.metrics import precision_recall_curve, average_precision_score

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

def plot_training_curves(metrics: Dict[str, List[float]], title: str = "Training Curves"):
    """Plot training and validation metrics over epochs."""
    plt.figure(figsize=(12, 4))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['val_loss'], label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(metrics['train_acc'], label='Train Acc')
    plt.plot(metrics['val_acc'], label='Val Acc')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
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
            serializable_metrics[key] = [v.item() if isinstance(v, (np.ndarray, torch.Tensor)) else v for v in value]
        else:
            serializable_metrics[key] = value
    
    with open(metrics_path, 'w') as f:
        json.dump(serializable_metrics, f)

def plot_comparative_curves(all_metrics: Dict[str, Dict], title: str = "Comparative Training Curves"):
    """Plot training curves for multiple models on the same graph."""
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    for model_name, metrics in all_metrics.items():
        plt.plot(metrics['train_loss'], label=f'{model_name} Train')
        plt.plot(metrics['val_loss'], label=f'{model_name} Val', linestyle='--')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    for model_name, metrics in all_metrics.items():
        plt.plot(metrics['train_acc'], label=f'{model_name} Train')
        plt.plot(metrics['val_acc'], label=f'{model_name} Val', linestyle='--')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('comparative_curves.png')
    plt.show()

def compute_additional_metrics(all_preds, all_labels, all_probs):
    """Compute additional metrics including AUPRC."""
    metrics = {}
    
    # Compute precision, recall, and F1 for each class
    for i in range(10):
        binary_labels = (np.array(all_labels) == i).astype(int)
        binary_probs = np.array(all_probs)[:, i]
        
        precision, recall, _ = precision_recall_curve(binary_labels, binary_probs)
        auprc = average_precision_score(binary_labels, binary_probs)
        
        metrics[f'class_{CIFAR10_CLASSES[i]}_auprc'] = auprc
    
    # Compute macro-averaged metrics
    metrics['macro_auprc'] = np.mean([metrics[f'class_{c}_auprc'] for c in CIFAR10_CLASSES])
    
    return metrics

def evaluate_metrics(model, test_loader, device, classes=CIFAR10_CLASSES):
    """Evaluate model performance with detailed metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            if len(labels.shape) > 1:  # For soft labels
                _, predicted = torch.max(outputs.data, 1)
                _, true_labels = torch.max(labels, 1)
            else:  # For hard labels
                _, predicted = torch.max(outputs.data, 1)
                true_labels = labels
                
            total += labels.size(0)
            correct += (predicted == true_labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(true_labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate basic metrics
    accuracy = 100 * correct / total
    avg_loss = test_loss / len(test_loader)
    
    # Calculate additional metrics
    additional_metrics = compute_additional_metrics(all_preds, all_labels, all_probs)
    
    report = classification_report(all_labels, all_preds, 
                                                    target_names=classes if classes else None,
                                                    output_dict=True)
    
    # Print summary of metrics
    print(f"\nTest Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%") 
    print(f"Macro AUPRC: {additional_metrics['macro_auprc']:.4f}")
    print(report)

    
    # Combine all metrics
    metrics = {
        'test_loss': avg_loss,
        'test_accuracy': accuracy,
        **additional_metrics,
        'classification_report': report
    }
    
    return metrics

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    device: str = None,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    model_name: str = None,
) -> Tuple[nn.Module, Dict]:
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    print(f"Training on {device}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    metrics = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
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

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                if len(labels.shape) > 1:  # For soft labels
                    _, predicted = torch.max(outputs.data, 1)
                    _, true_labels = torch.max(labels, 1)
                else:  # For hard labels
                    _, predicted = torch.max(outputs.data, 1)
                    true_labels = labels
                    
                val_total += labels.size(0)
                val_correct += (predicted == true_labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Store metrics
        metrics['train_loss'].append(avg_train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(avg_val_loss)
        metrics['val_acc'].append(val_acc)
        
        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
            f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%"
        )

        # Save model if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if model_name:
                torch.save(model.state_dict(), f"models/{model_name}.pth")
                print(f"Saved model with improved validation accuracy: {val_acc:.2f}%")

    # Plot training curves
    plot_training_curves(metrics, title=f"Training Curves - {model_name if model_name else 'Model'}")
    
    return model, metrics


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
    
# Load CIFAR-10 dataset and return train, validation, and test DataLoaders
def load_cifar10_experiment():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float32),
    ])
    
    full_dataset = datasets.CIFAR10(root="../data/cifar-10", train=True, download=True, transform=transform)
    train_dataset = datasets.CIFAR10(root="../data/cifar-10", train=False, download=True, transform=transform)

    # Split full dataset into augment, test, and validation sets
    augment_size = int(0.7 * len(full_dataset))
    val_size = (len(full_dataset) - augment_size) // 2
    test_size = len(full_dataset) - augment_size - val_size
    
    augment_dataset, test_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [augment_size, test_size, val_size], 
        generator=torch.Generator().manual_seed(229)
    )

    return augment_dataset, train_dataset, test_dataset, val_dataset


device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)
print(f"Using device: {device}")

# Load datasets
augment_dataset, train_dataset, test_dataset, val_dataset = load_cifar10_experiment()

# Create dataloaders
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
augment_loader = DataLoader(augment_dataset, batch_size=batch_size, shuffle=False)

# Load CIFAR-10H soft labels
cifar10h_probs = np.load("../data/cifar-10h/cifar10h-probs.npy").astype(np.float32)

print(f"CIFAR-10 dataset loaded with {len(augment_dataset)} augment samples, {len(train_dataset)} training samples, {len(test_dataset)} testing samples, and {len(val_dataset)} validation samples")
print(f"CIFAR-10H soft labels loaded with shape {cifar10h_probs.shape}")

all_metrics = {}


print("\n=== Running Main Experiment ===")
print("Training with artificial soft labels + CIFAR-10H")

# Load soft label predictor model
soft_label_model = ImageHardToSoftLabelModel().to(device)
soft_label_model.load_state_dict(torch.load("models/soft_label_model.pt", weights_only=True))
soft_label_model.eval()

# Generate artificial soft labels for augment dataset
augmented_soft_dataset = create_soft_label_dataset(soft_label_model, augment_loader, device)
cifar10h_soft_dataset = CIFAR10LabelDataset(train_dataset, cifar10h_probs)

# Combine datasets and create loader
combined_soft_dataset = ConcatDataset([augmented_soft_dataset, cifar10h_soft_dataset])
combined_soft_loader = DataLoader(combined_soft_dataset, batch_size=batch_size, shuffle=True)

print(f"Combined soft dataset size: {len(combined_soft_dataset)} samples")
print(f"- Augmented soft labels: {len(augmented_soft_dataset)} samples")
print(f"- CIFAR-10H soft labels: {len(cifar10h_soft_dataset)} samples")

# Train model
model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

model, metrics = train_model(
    model=model,
    train_loader=combined_soft_loader,
    val_loader=val_loader,
    num_epochs=30,
    device=device,
    model_name="resnet34_artificial_soft"
)

# Evaluate and save metrics for main experiment
model_metrics = evaluate_metrics(model, test_loader, device)
all_metrics['artificial_soft'] = {
    **metrics,  # Training metrics
    **model_metrics  # Test metrics
}
save_metrics(all_metrics['artificial_soft'], 'artificial_soft')


print("\n=== Running Baseline 1 ===")
print("Training with CIFAR-10 hard labels + CIFAR-10H hard labels")

# Create combined hard label dataset
combined_hard_dataset = ConcatDataset([augment_dataset, train_dataset])
combined_hard_loader = DataLoader(combined_hard_dataset, batch_size=batch_size, shuffle=True)

print(f"Combined hard dataset size: {len(combined_hard_dataset)} samples")
print(f"- Augment hard labels: {len(augment_dataset)} samples")
print(f"- CIFAR-10H hard labels: {len(train_dataset)} samples")

model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

model, metrics = train_model(
    model=model,
    train_loader=combined_hard_loader,
    val_loader=val_loader,
    num_epochs=30,
    device=device,
    model_name="resnet34_hard_baseline"
)

# Evaluate and save metrics for baseline 1
model_metrics = evaluate_metrics(model, test_loader, device)
all_metrics['hard_baseline'] = {
    **metrics,  # Training metrics
    **model_metrics  # Test metrics
}
save_metrics(all_metrics['hard_baseline'], 'hard_baseline')


print("\n=== Running Baseline 2 ===")
print("Training with CIFAR-10 hard labels + CIFAR-10H soft labels")

# Create mixed dataset (hard + soft labels)
hard_label_dataset = CIFAR10LabelDataset(augment_dataset)
soft_label_dataset = CIFAR10LabelDataset(train_dataset, cifar10h_probs)
combined_mixed_dataset = ConcatDataset([hard_label_dataset, soft_label_dataset])
combined_mixed_loader = DataLoader(combined_mixed_dataset, batch_size=batch_size, shuffle=True)

print(f"Combined mixed dataset size: {len(combined_mixed_dataset)} samples")
print(f"- Augment hard labels: {len(hard_label_dataset)} samples")
print(f"- CIFAR-10H soft labels: {len(soft_label_dataset)} samples")

model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

model, metrics = train_model(
    model=model,
    train_loader=combined_mixed_loader,
    val_loader=val_loader,
    num_epochs=30,
    device=device,
    model_name="resnet34_mixed_baseline"
)

# Evaluate and save metrics for baseline 2
model_metrics = evaluate_metrics(model, test_loader, device)
all_metrics['mixed_baseline'] = {
    **metrics,  # Training metrics
    **model_metrics  # Test metrics
}
save_metrics(all_metrics['mixed_baseline'], 'mixed_baseline')