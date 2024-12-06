import numpy as np
import timm
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
import os
from typing import Tuple

class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        for model in self.models:
            model.eval() 

    def forward(self, x):
        # Aggregate predictions from all models
        predictions = []
        for model in self.models:
            outputs = model(x)
            probs = nn.functional.softmax(outputs, dim=1)
            predictions.append(probs)
        ensemble_probs = torch.mean(torch.stack(predictions), dim=0)
        return ensemble_probs

# Load CIFAR-10 dataset and return train, validation, and test DataLoaders
def load_cifar10() -> Tuple[Dataset, Dataset, Dataset]:
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




cifar10_train_dataset, cifar10_test_dataset, cifar10_val_dataset = load_cifar10()  # Changed variable name to reflect split
cifar10_train_loader = DataLoader(cifar10_train_dataset, batch_size=128, shuffle=True)
cifar10_test_loader = DataLoader(cifar10_test_dataset, batch_size=128, shuffle=False)
cifar10_val_loader = DataLoader(cifar10_val_dataset, batch_size=128, shuffle=False)
print(
    f"CIFAR-10 dataset loaded with {len(cifar10_train_dataset)} training, {len(cifar10_test_dataset)} test, and {len(cifar10_val_dataset)} validation samples"
)

def evaluate_nn_model(model, cifar10_test_loader):
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

model1 = models.densenet121(weights=None)
model2 = models.regnet_x_400mf(weights=None)

# Update the classifiers for CIFAR-10 (10 classes)
num_classes = 10
model1.classifier = nn.Linear(model1.classifier.in_features, num_classes)
model2.fc = nn.Linear(model2.fc.in_features, num_classes)

# Load saved weights
print('Loading Weights')
model1.load_state_dict(torch.load("models/DenseNet_cifar10h.pth"))
model2.load_state_dict(torch.load("models/RegNet_cifar10h.pth"))

# Move models to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = model1.to(device)
model2 = model2.to(device)

# Create the ensemble model
ensemble_model = EnsembleModel(models=[model1, model2])
ensemble_model = ensemble_model.to(device)

print('EVALUATING ENSEMBLE')
evaluate_nn_model(ensemble_model, cifar10_test_loader)