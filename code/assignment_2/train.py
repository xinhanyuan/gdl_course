from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


OUTPUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUTPUT_DIR / "data"
MODEL_PATH = OUTPUT_DIR / "mnist_mlp.pth"
CURVE_PATH = OUTPUT_DIR / "training_curves.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MLP on MNIST.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MNISTMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)

    worker_count = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=worker_count,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=worker_count,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device, criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * targets.size(0)
            total_correct += (outputs.argmax(dim=1) == targets).sum().item()
            total_samples += targets.size(0)

    return total_loss / total_samples, total_correct / total_samples


def plot_curves(train_losses: list[float], test_accuracies: list[float]) -> None:
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_losses, marker="o")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(epochs, test_accuracies, marker="o", color="tab:orange")
    axes[1].set_title("Test Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.9, 1.0)
    axes[1].grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(CURVE_PATH, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = build_dataloaders(args.batch_size)

    model = MNISTMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses: list[float] = []
    test_accuracies: list[float] = []
    best_accuracy = 0.0

    print(f"Using device: {device}")
    print(f"Training data: {len(train_loader.dataset)} samples")
    print(f"Test data: {len(test_loader.dataset)} samples")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)

        train_loss = running_loss / total_samples
        test_loss, test_accuracy = evaluate(model, test_loader, device, criterion)
        train_losses.append(train_loss)
        test_accuracies.append(test_accuracy)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "test_accuracy": test_accuracy,
                    "epochs": args.epochs,
                },
                MODEL_PATH,
            )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"test_loss={test_loss:.4f} | "
            f"test_acc={test_accuracy * 100:.2f}%"
        )

    plot_curves(train_losses, test_accuracies)
    print(f"Best test accuracy: {best_accuracy * 100:.2f}%")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved curves to: {CURVE_PATH}")


if __name__ == "__main__":
    main()
