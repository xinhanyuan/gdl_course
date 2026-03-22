from __future__ import annotations

import argparse
import os
import random
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


matplotlib.use("Agg")


OUTPUT_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = OUTPUT_DIR / "best_cnn.pth"
CURVE_PATH = OUTPUT_DIR / "cnn_scratch_curves.png"
DEFAULT_DATA_ROOT = ROOT_DIR / "data"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_TO_IDX = {"cat": 0, "dog": 1}
DEFAULT_NUM_WORKERS = 0 if os.name == "nt" else min(4, os.cpu_count() or 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a scratch CNN for cats vs dogs classification.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Dataset directory. If omitted, the script will auto-detect it.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate for Adam.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for Adam.")
    parser.add_argument("--image-size", type=int, default=128, help="Resize images to IMAGE_SIZE x IMAGE_SIZE.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="DataLoader worker count.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on total samples for quick smoke tests.")
    parser.add_argument("--skip-integrity-check", action="store_true", help="Skip image verification during dataset discovery.")
    parser.add_argument("--disable-amp", action="store_true", help="Disable mixed precision on CUDA.")
    parser.add_argument("--log-interval", type=int, default=20, help="Print batch-level progress every N steps.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def looks_like_dataset_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False

    if (path / "Cat").is_dir() and (path / "Dog").is_dir():
        return True

    for pattern in ("cat.*", "dog.*"):
        if any(path.glob(pattern)):
            return True
    return False


def resolve_data_dir(user_path: Path | None) -> Path:
    if user_path is not None:
        candidates = [
            user_path,
            user_path / "PetImages",
            user_path / "train",
            user_path / "kagglecatsanddogs_5340" / "PetImages",
        ]
    else:
        candidates = [
            DEFAULT_DATA_ROOT / "kagglecatsanddogs_5340" / "PetImages",
            DEFAULT_DATA_ROOT / "train",
            DEFAULT_DATA_ROOT,
        ]

    for candidate in candidates:
        if looks_like_dataset_dir(candidate):
            return candidate

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find a cats-vs-dogs dataset directory. Checked:\n"
        f"{searched}"
    )


def infer_label(path: Path) -> int | None:
    parent_name = path.parent.name.lower()
    if parent_name in CLASS_TO_IDX:
        return CLASS_TO_IDX[parent_name]

    stem = path.name.lower()
    for name, idx in CLASS_TO_IDX.items():
        if stem.startswith(f"{name}."):
            return idx
    return None


def verify_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, UnidentifiedImageError):
        return False


def discover_samples(data_dir: Path, seed: int, max_samples: int | None, check_integrity: bool) -> list[tuple[Path, int]]:
    label_to_candidates: dict[int, list[Path]] = {0: [], 1: []}

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = infer_label(path)
        if label is None:
            continue
        label_to_candidates[label].append(path)

    if not label_to_candidates[0] or not label_to_candidates[1]:
        raise RuntimeError(f"Expected both cat and dog images under {data_dir}.")

    rng = random.Random(seed)
    invalid_files: list[Path] = []
    samples: list[tuple[Path, int]] = []

    if max_samples is None:
        for label, paths in label_to_candidates.items():
            rng.shuffle(paths)
            for path in paths:
                if check_integrity and not verify_image(path):
                    invalid_files.append(path)
                    continue
                samples.append((path, label))
    else:
        labels = sorted(label_to_candidates)
        base_take = max_samples // len(labels)
        remainder = max_samples % len(labels)

        for index, label in enumerate(labels):
            paths = label_to_candidates[label][:]
            rng.shuffle(paths)
            take = base_take + (1 if index < remainder else 0)
            taken = 0

            for path in paths:
                if check_integrity and not verify_image(path):
                    invalid_files.append(path)
                    continue
                samples.append((path, label))
                taken += 1
                if taken >= take:
                    break

    if not samples:
        raise RuntimeError(f"No valid samples found under {data_dir}.")

    if invalid_files:
        print(f"Skipped {len(invalid_files)} invalid images during discovery.")

    return samples


def split_samples(samples: list[tuple[Path, int]], train_ratio: float, seed: int) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    grouped: dict[int, list[tuple[Path, int]]] = {0: [], 1: []}
    for sample in samples:
        grouped[sample[1]].append(sample)

    rng = random.Random(seed)
    train_samples: list[tuple[Path, int]] = []
    val_samples: list[tuple[Path, int]] = []

    for label, label_samples in grouped.items():
        rng.shuffle(label_samples)
        split_index = int(len(label_samples) * train_ratio)
        split_index = max(1, min(split_index, len(label_samples) - 1))
        train_samples.extend(label_samples[:split_index])
        val_samples.extend(label_samples[split_index:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


class CatsDogsDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform: transforms.Compose) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), label


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.01)],
                p=0.5,
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, val_transform


def make_dataloaders(
    train_samples: list[tuple[Path, int]],
    val_samples: list[tuple[Path, int]],
    batch_size: int,
    image_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    train_transform, val_transform = build_transforms(image_size)
    train_dataset = CatsDogsDataset(train_samples, train_transform)
    val_dataset = CatsDogsDataset(val_samples, val_transform)

    common_loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_loader_args)
    return train_loader, val_loader


class ConvBlock(nn.Module):
    """Basic convolution block: Conv -> BN -> ReLU -> MaxPool."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = (32, 64, 128, 256)
        self.features = nn.Sequential(
            ConvBlock(3, channels[0]),
            ConvBlock(channels[0], channels[1]),
            ConvBlock(channels[1], channels[2], dropout=0.05),
            ConvBlock(channels[2], channels[3], dropout=0.10),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(channels[-1], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 2),
        )
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    amp_enabled: bool,
    epoch: int,
    total_epochs: int,
    log_interval: int,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    num_batches = len(data_loader)

    for step, (inputs, targets) in enumerate(data_loader, start=1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * targets.size(0)
        total_correct += (outputs.argmax(dim=1) == targets).sum().item()
        total_samples += targets.size(0)

        should_log = log_interval > 0 and (step == 1 or step % log_interval == 0 or step == num_batches)
        if should_log:
            running_loss = total_loss / total_samples
            running_acc = total_correct / total_samples
            print(
                f"[Train] epoch {epoch:02d}/{total_epochs} | "
                f"step {step:04d}/{num_batches:04d} | "
                f"loss={running_loss:.4f} | acc={running_acc * 100:.2f}%"
            )

    return total_loss / total_samples, total_correct / total_samples


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    amp_enabled: bool,
    epoch: int,
    total_epochs: int,
    log_interval: int,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    num_batches = len(data_loader)

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(data_loader, start=1):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            total_loss += loss.item() * targets.size(0)
            total_correct += (outputs.argmax(dim=1) == targets).sum().item()
            total_samples += targets.size(0)

            should_log = log_interval > 0 and (step == 1 or step % log_interval == 0 or step == num_batches)
            if should_log:
                running_loss = total_loss / total_samples
                running_acc = total_correct / total_samples
                print(
                    f"[Val]   epoch {epoch:02d}/{total_epochs} | "
                    f"step {step:04d}/{num_batches:04d} | "
                    f"loss={running_loss:.4f} | acc={running_acc * 100:.2f}%"
                )

    return total_loss / total_samples, total_correct / total_samples


def plot_curves(history: dict[str, list[float]]) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("CrossEntropy")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].plot(epochs, [acc * 100 for acc in history["train_acc"]], marker="o", label="Train")
    axes[1].plot(epochs, [acc * 100 for acc in history["val_acc"]], marker="o", label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(CURVE_PATH, dpi=200)
    plt.close(fig)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_acc: float,
    data_dir: Path,
    image_size: int,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "class_to_idx": CLASS_TO_IDX,
            "data_dir": str(data_dir),
            "image_size": image_size,
        },
        MODEL_PATH,
    )


def main() -> None:
    args = parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1.")

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = torch.cuda.is_available()

    data_dir = resolve_data_dir(args.data_dir)
    samples = discover_samples(
        data_dir=data_dir,
        seed=args.seed,
        max_samples=args.max_samples,
        check_integrity=not args.skip_integrity_check,
    )
    train_samples, val_samples = split_samples(samples, args.train_ratio, args.seed)

    train_loader, val_loader = make_dataloaders(
        train_samples=train_samples,
        val_samples=val_samples,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.disable_amp

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    label_counts = Counter(label for _, label in samples)
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    best_val_acc = 0.0

    print(f"Using device: {device}")
    print(f"Dataset directory: {data_dir}")
    print(f"Total valid samples: {len(samples)}")
    print(f"Class distribution: cat={label_counts[0]}, dog={label_counts[1]}")
    print(f"Train / Val split: {len(train_samples)} / {len(val_samples)}")
    print(f"AMP enabled: {amp_enabled}")
    print(f"Batch log interval: {args.log_interval}")

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            device,
            criterion,
            optimizer,
            scaler,
            amp_enabled,
            epoch,
            args.epochs,
            args.log_interval,
        )
        val_loss, val_acc = evaluate(
            model,
            val_loader,
            device,
            criterion,
            amp_enabled,
            epoch,
            args.epochs,
            args.log_interval,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_acc=best_val_acc,
                data_dir=data_dir,
                image_size=args.image_size,
            )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc * 100:.2f}% | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc * 100:.2f}%"
        )
        scheduler.step()

    plot_curves(history)
    print(f"Best validation accuracy: {best_val_acc * 100:.2f}%")
    print(f"Saved checkpoint to: {MODEL_PATH}")
    print(f"Saved curves to: {CURVE_PATH}")


if __name__ == "__main__":
    main()
