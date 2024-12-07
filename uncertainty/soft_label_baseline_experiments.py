"""
This file contains the code to run the baseline experiments.

More specifically, we are investigating the performance of basic models on the CIFAR-10 and CIFAR-10H datasets. The tasks for these datasets are multi-class classification.

The basic models include:
    * ResNet-50
    * WideResNet: IF BEING USED: Install Timm using conda install -c conda-forge timm
    * DenseNet
    * RegNet
"""

import numpy as np
import timm
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler
import os
from typing import Tuple


# # Loading Data

def load_cifar10h() -> Dataset:
    cifar10h_probs_path = "../data/cifar-10h/cifar10h-probs.npy"
    if not os.path.exists(cifar10h_probs_path):
        raise FileNotFoundError(
            f"Soft labels not found at {cifar10h_probs_path}. Please ensure the CIFAR-10H data is downloaded."
        )

    cifar10h_probs = np.load(cifar10h_probs_path).astype(np.float32)
    cifar10_test = datasets.CIFAR10(
        root="../data/cifar-10", train=False, download=True, transform=transforms.ToTensor()
    )

    class CIFAR10H(Dataset):
        def __init__(self, cifar10_dataset: Dataset, soft_labels: np.ndarray):
            self.cifar10_dataset = cifar10_dataset
            self.soft_labels = soft_labels

        def __len__(self) -> int:
            return len(self.cifar10_dataset)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            image, _ = self.cifar10_dataset[idx]
            soft_label = torch.from_numpy(self.soft_labels[idx])
            return image.float(), soft_label

    cifar10h_dataset = CIFAR10H(cifar10_test, cifar10h_probs)
    return cifar10h_dataset


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




# Load CIFAR-10 dataset and return augment, train, validation, and test DataLoaders
def load_cifar10_experiment() -> Tuple[Dataset, Dataset, Dataset]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ]
    )
    full_dataset = datasets.CIFAR10(root="../data/cifar-10", train=True, download=True, transform=transform)
    # we use the test dataset for training, similar to the CIFAR-10H experiment
    train_dataset = datasets.CIFAR10(root="../data/cifar-10", train=False, download=True, transform=transform)

    # This dataset will be used for augmenting, testing, and validation.
    augment_size = int(0.7 * len(full_dataset))
    val_size = (len(full_dataset) - augment_size) // 2
    test_size = len(full_dataset) - augment_size - val_size
    augment_dataset, test_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [augment_size, test_size, val_size], generator=torch.Generator().manual_seed(229)
    )

    return augment_dataset, train_dataset, test_dataset, val_dataset




cifar10_exp_augment_dataset, cifar10_exp_train_dataset, cifar10_exp_test_dataset, cifar10_exp_val_dataset = load_cifar10_experiment()  # Changed variable name to reflect split
combined_train_dataset = ConcatDataset([cifar10_exp_augment_dataset, cifar10_exp_train_dataset])
cifar10_exp_train_loader = DataLoader(combined_train_dataset, batch_size=128, shuffle=True)
cifar10_exp_test_loader = DataLoader(cifar10_exp_test_dataset, batch_size=128, shuffle=False)
cifar10_exp_val_loader = DataLoader(cifar10_exp_val_dataset, batch_size=128, shuffle=False)
print(
    f"CIFAR-10 dataset loaded with {len(cifar10_exp_augment_dataset)} augment, {len(cifar10_exp_train_dataset)} training, {len(cifar10_exp_test_dataset)} test, and {len(cifar10_exp_val_dataset)} validation samples"
)


# # Training
# Training is done on the CIFAR-10H dataset. Evaluation is done on the CIFAR-10 train set, which we use as a test set.

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
) -> nn.Module:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    model = model.to(device)

    best_val_acc = 0.0

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
        print(
            f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {running_loss/len(train_loader):.4f}, Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )

        # Save model if validation accuracy improves
        if accuracy > best_val_acc:
            best_val_acc = accuracy
            torch.save(model.state_dict(), f"models/{model.__class__.__name__}_cifar10h.pth")
            print(f"Saved model with improved validation accuracy: {accuracy:.2f}%")

    return model


# ## Training Neural Networks



def train_nn_model(
    model, cifar10h_loader: DataLoader, cifar10_val_loader: DataLoader, num_epochs: int = 20, lr: float = 0.001
) -> list:
    print(f"\nTraining {model.__class__.__name__} on CIFAR-10H...")

    # Adjust the final layer for CIFAR-10
    if isinstance(model, models.ResNet):
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
    elif isinstance(model, models.VGG):
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, 10)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = train_model(
        model=model,
        train_loader=cifar10h_loader,
        val_loader=cifar10_val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
    )

def evaluate_nn_model(model, cifar10_test_loader):
    model.load_state_dict(
        torch.load(f"models/{model.__class__.__name__}_cifar10h.pth", weights_only=True)
    )
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

def main():
    # Load CIFAR-10 dataset
    print("Loading CIFAR-10 dataset...")
    cifar10_train_dataset, cifar10_test_dataset, cifar10_val_dataset = load_cifar10()
    cifar10_train_loader = DataLoader(cifar10_train_dataset, batch_size=128, shuffle=True)
    cifar10_val_loader = DataLoader(cifar10_val_dataset, batch_size=128, shuffle=False)
    cifar10_test_loader = DataLoader(cifar10_test_dataset, batch_size=128, shuffle=False)

    # # Step 1: Train ResNet Model on CIFAR-10 hard labels
    # print("\nStep 1: Training Resnet on CIFAR-10 (hard labels)...")
    # config = Config()
    # device = torch.device(
    #     "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    # )
    # # Using RESNET34
    # print("Using ResNet")
    # model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)


    # # Train on CIFAR-10
    # train_nn_model(
    #     model=model,
    #     cifar10h_loader=cifar10_train_loader,
    #     cifar10_val_loader=cifar10_val_loader,
    #     num_epochs=20,
    # )

    # # Evaluate on CIFAR-10 test set
    # print("\nEvaluating on CIFAR-10 test set...")
    # evaluate_nn_model(model, cifar10_test_loader)

    # Step 1: Training on CIFAR-10H soft labels ResNet
    # print("\nStep 1: Training on CIFAR-10H (soft labels)... ResNet Model")
    # model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    # config = Config()
    # device = torch.device(
    #     "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    # )
    # cifar10h_loader = load_cifar10()

    # # Define loss function and optimizer for fine-tuning with soft labels

    # # Fine-tune on CIFAR-10H
    # train_nn_model(
    #     model=model,
    #     cifar10h_loader=cifar10h_loader,
    #     cifar10_val_loader=cifar10_val_loader,
    #     num_epochs=20,  # Fewer epochs for fine-tuning
    # )

    # # Evaluate on CIFAR-10 test set again after fine-tuning
    # print("\nEvaluating on CIFAR-10 test set after training...")
    # evaluate_nn_model(model, cifar10_test_loader)

    # Step 2: Training on CIFAR-10H soft labels WideResNet  / Timm Pretrained weights
    print("\nStep 2.1: Training on CIFAR-10H (soft labels)... WideResNet Model")
    model = timm.create_model('wide_resnet50_2', pretrained=True)
    num_classes = 10
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    #Load Cifar10H
    cifar10h_dataset = load_cifar10h()
    cifar10h_loader = DataLoader(cifar10h_dataset, batch_size=128, shuffle=True)


    # Define loss function and optimizer for fine-tuning with soft labels

    # Fine-tune on CIFAR-10H
    train_nn_model(
        model=model,
        cifar10h_loader=cifar10h_loader,
        cifar10_val_loader=cifar10_val_loader,
        num_epochs=20,  # Fewer epochs for fine-tuning
    )

    # Evaluate on CIFAR-10 test set again after fine-tuning
    print("\nEvaluating on CIFAR-10 test set after training...")
    evaluate_nn_model(model, cifar10_test_loader)

    # Step 2.2: Training on CIFAR-10H soft labels DenseNet
    print("\nStep 2.2: Training on CIFAR-10H (soft labels)... DenseNet Model")
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    num_classes = 10
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    cifar10h_dataset = load_cifar10h()
    cifar10h_loader = DataLoader(cifar10h_dataset, batch_size=128, shuffle=True)

    # Define loss function and optimizer for fine-tuning with soft labels

    # Fine-tune on CIFAR-10H
    train_nn_model(
        model=model,
        cifar10h_loader=cifar10h_loader,
        cifar10_val_loader=cifar10_val_loader,
        num_epochs=20,  # Fewer epochs for fine-tuning
    )

    # Evaluate on CIFAR-10 test set again after fine-tuning
    print("\nEvaluating on CIFAR-10 test set after training...")
    evaluate_nn_model(model, cifar10_test_loader)

    # Step 2.3: Training on CIFAR-10H soft labels DenseNet
    print("\nStep 2.3: Training on CIFAR-10H (soft labels)... RegNet Model")
    model = models.regnet_x_400mf(weights=models.RegNet_X_400MF_Weights.DEFAULT)

    num_classes = 10
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    cifar10h_dataset = load_cifar10h()
    cifar10h_loader = DataLoader(cifar10h_dataset, batch_size=128, shuffle=True)

    # Define loss function and optimizer for fine-tuning with soft labels

    # Fine-tune on CIFAR-10H
    train_nn_model(
        model=model,
        cifar10h_loader=cifar10h_loader,
        cifar10_val_loader=cifar10_val_loader,
        num_epochs=20,  # Fewer epochs for fine-tuning
    )

    # Evaluate on CIFAR-10 test set again after fine-tuning
    print("\nEvaluating on CIFAR-10 test set after training...")
    evaluate_nn_model(model, cifar10_test_loader)


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)  # Create directory for saving models
    main()