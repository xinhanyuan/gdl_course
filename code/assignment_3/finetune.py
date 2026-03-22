from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models, transforms

from train_scratch import (
    CLASS_TO_IDX,
    CatsDogsDataset,
    DEFAULT_NUM_WORKERS,
    discover_samples,
    resolve_data_dir,
    set_seed,
    split_samples,
)


OUTPUT_DIR = Path(__file__).resolve().parent
MODEL_PATH = OUTPUT_DIR / "best_phase2.pth"
CURVE_PATH = OUTPUT_DIR / "finetune_curves.png"
DEFAULT_TORCH_HOME = OUTPUT_DIR / ".torch"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained ResNet18 on cats vs dogs.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Dataset directory. If omitted, the script will auto-detect it.")
    parser.add_argument("--phase1-epochs", type=int, default=5, help="Warm-up epochs with a frozen backbone.")
    parser.add_argument("--phase2-epochs", type=int, default=10, help="Fine-tuning epochs with the full network unfrozen.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--fc-lr", type=float, default=1e-3, help="Learning rate for the classifier head.")
    parser.add_argument("--backbone-lr", type=float, default=1e-4, help="Learning rate for the ResNet backbone in phase 2.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for Adam.")
    parser.add_argument("--image-size", type=int, default=224, help="Resize images to IMAGE_SIZE x IMAGE_SIZE.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="DataLoader worker count.")
    parser.add_argument("--torch-home", type=Path, default=DEFAULT_TORCH_HOME, help="Directory used to cache pretrained weights.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on total samples for quick smoke tests.")
    parser.add_argument("--skip-integrity-check", action="store_true", help="Skip image verification during dataset discovery.")
    parser.add_argument("--disable-amp", action="store_true", help="Disable mixed precision on CUDA.")
    parser.add_argument(
        "--weights",
        choices=("imagenet", "none"),
        default="imagenet",
        help="Use ImageNet pretrained weights by default. Choose 'none' only for offline smoke tests.",
    )
    return parser.parse_args()


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)],
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


def build_model(weights_name: str) -> models.ResNet:
    weights = None
    if weights_name == "imagenet":
        weights = models.ResNet18_Weights.IMAGENET1K_V1

    try:
        model = models.resnet18(weights=weights)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load ResNet18 weights. If the pretrained weights are not cached locally, "
            "run once with network access or use --weights none only for a smoke test."
        ) from exc

    model.fc = nn.Linear(model.fc.in_features, len(CLASS_TO_IDX))
    return model


def freeze_backbone(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def get_backbone_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [param for name, param in model.named_parameters() if not name.startswith("fc.")]


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    amp_enabled: bool,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in data_loader:
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

    return total_loss / total_samples, total_correct / total_samples


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    amp_enabled: bool,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            total_loss += loss.item() * targets.size(0)
            total_correct += (outputs.argmax(dim=1) == targets).sum().item()
            total_samples += targets.size(0)

    return total_loss / total_samples, total_correct / total_samples


def format_lrs(optimizer: optim.Optimizer) -> str:
    if len(optimizer.param_groups) == 1:
        return f"fc={optimizer.param_groups[0]['lr']:.6f}"
    return f"fc={optimizer.param_groups[0]['lr']:.6f}, backbone={optimizer.param_groups[1]['lr']:.6f}"


def plot_curves(history: dict[str, list[float]], phase1_epochs: int) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("CrossEntropy")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(epochs, [acc * 100 for acc in history["train_acc"]], marker="o", label="Train")
    axes[1].plot(epochs, [acc * 100 for acc in history["val_acc"]], marker="o", label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    for axis in axes:
        axis.axvline(phase1_epochs + 0.5, color="gray", linestyle="--", linewidth=1.5, label="Phase boundary")
        axis.legend()

    fig.tight_layout()
    fig.savefig(CURVE_PATH, dpi=200)
    plt.close(fig)


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    best_val_acc: float,
    data_dir: Path,
    image_size: int,
    weights_name: str,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "phase": 2,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "class_to_idx": CLASS_TO_IDX,
            "data_dir": str(data_dir),
            "image_size": image_size,
            "weights": weights_name,
        },
        MODEL_PATH,
    )


def run_stage(
    stage_name: str,
    model: nn.Module,
    data_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    amp_enabled: bool,
    num_epochs: int,
    start_epoch: int,
    total_epochs: int,
    history: dict[str, list[float]],
    data_dir: Path,
    image_size: int,
    weights_name: str,
    best_val_acc: float,
    save_best: bool,
) -> float:
    for offset in range(num_epochs):
        epoch = start_epoch + offset + 1
        stage_epoch = offset + 1

        train_loss, train_acc = train_one_epoch(model, data_loader, device, criterion, optimizer, scaler, amp_enabled)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion, amp_enabled)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if save_best and val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_acc=best_val_acc,
                data_dir=data_dir,
                image_size=image_size,
                weights_name=weights_name,
            )

        print(
            f"{stage_name} | epoch {stage_epoch:02d}/{num_epochs} | "
            f"overall {epoch:02d}/{total_epochs} | "
            f"lr({format_lrs(optimizer)}) | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc * 100:.2f}% | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc * 100:.2f}%"
        )

        scheduler.step()

    return best_val_acc


def main() -> None:
    args = parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1.")
    if args.phase1_epochs <= 0 or args.phase2_epochs <= 0:
        raise ValueError("--phase1-epochs and --phase2-epochs must be positive.")

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    args.torch_home.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(args.torch_home)
    torch.hub.set_dir(str(args.torch_home))

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
    model = build_model(args.weights).to(device)

    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    total_epochs = args.phase1_epochs + args.phase2_epochs

    label_counts = Counter(label for _, label in samples)
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    print(f"Using device: {device}")
    print(f"Dataset directory: {data_dir}")
    print(f"Total valid samples: {len(samples)}")
    print(f"Class distribution: cat={label_counts[0]}, dog={label_counts[1]}")
    print(f"Train / Val split: {len(train_samples)} / {len(val_samples)}")
    print(f"AMP enabled: {amp_enabled}")
    print(f"Pretrained weights: {args.weights}")
    print(f"Torch cache directory: {args.torch_home}")

    freeze_backbone(model)
    print(f"Phase 1 trainable parameters: {count_trainable_parameters(model):,}")
    phase1_optimizer = optim.Adam(model.fc.parameters(), lr=args.fc_lr, weight_decay=args.weight_decay)
    phase1_scheduler = optim.lr_scheduler.CosineAnnealingLR(phase1_optimizer, T_max=max(1, args.phase1_epochs))

    run_stage(
        stage_name="Phase 1",
        model=model,
        data_loader=train_loader,
        val_loader=val_loader,
        device=device,
        criterion=criterion,
        optimizer=phase1_optimizer,
        scheduler=phase1_scheduler,
        scaler=scaler,
        amp_enabled=amp_enabled,
        num_epochs=args.phase1_epochs,
        start_epoch=0,
        total_epochs=total_epochs,
        history=history,
        data_dir=data_dir,
        image_size=args.image_size,
        weights_name=args.weights,
        best_val_acc=0.0,
        save_best=False,
    )

    unfreeze_all(model)
    print(f"Phase 2 trainable parameters: {count_trainable_parameters(model):,}")
    phase2_optimizer = optim.Adam(
        [
            {"params": model.fc.parameters(), "lr": args.fc_lr},
            {"params": get_backbone_parameters(model), "lr": args.backbone_lr},
        ],
        weight_decay=args.weight_decay,
    )
    phase2_scheduler = optim.lr_scheduler.CosineAnnealingLR(phase2_optimizer, T_max=max(1, args.phase2_epochs))

    best_val_acc = run_stage(
        stage_name="Phase 2",
        model=model,
        data_loader=train_loader,
        val_loader=val_loader,
        device=device,
        criterion=criterion,
        optimizer=phase2_optimizer,
        scheduler=phase2_scheduler,
        scaler=scaler,
        amp_enabled=amp_enabled,
        num_epochs=args.phase2_epochs,
        start_epoch=args.phase1_epochs,
        total_epochs=total_epochs,
        history=history,
        data_dir=data_dir,
        image_size=args.image_size,
        weights_name=args.weights,
        best_val_acc=0.0,
        save_best=True,
    )

    plot_curves(history, args.phase1_epochs)
    print(f"Best phase 2 validation accuracy: {best_val_acc * 100:.2f}%")
    print(f"Saved checkpoint to: {MODEL_PATH}")
    print(f"Saved curves to: {CURVE_PATH}")


if __name__ == "__main__":
    main()
