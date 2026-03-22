from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from PIL import Image

from train_scratch import SimpleCNN, build_transforms


OUTPUT_DIR = Path(__file__).resolve().parent
MODEL_PATH = OUTPUT_DIR / "best_cnn.pth"
OUTPUT_PATH = OUTPUT_DIR / "cnn_feature_maps.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize intermediate feature maps from the trained scratch CNN.")
    parser.add_argument("--checkpoint", type=Path, default=MODEL_PATH, help="Path to best_cnn.pth.")
    parser.add_argument("--image-path", type=Path, default=None, help="Optional image path. If omitted, a sample cat image is used.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Output image path.")
    parser.add_argument("--top-k", type=int, default=8, help="Number of channels to visualize for each block.")
    return parser.parse_args()


def default_image_path(data_dir: Path) -> Path:
    candidates = [
        data_dir / "Cat" / "100.jpg",
        data_dir / "Cat" / "0.jpg",
        data_dir / "Dog" / "100.jpg",
        data_dir / "Dog" / "0.jpg",
    ]
    for path in candidates:
        if path.exists():
            return path

    for path in sorted(data_dir.rglob("*.jpg")):
        return path

    raise FileNotFoundError(f"No image file found under {data_dir}.")


def load_checkpoint(checkpoint_path: Path) -> tuple[SimpleCNN, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = SimpleCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def normalize_map(feature_map: torch.Tensor) -> torch.Tensor:
    feature_map = feature_map.detach().cpu()
    min_value = feature_map.min()
    max_value = feature_map.max()
    if float(max_value - min_value) < 1e-8:
        return torch.zeros_like(feature_map)
    return (feature_map - min_value) / (max_value - min_value)


def top_channel_indices(activation: torch.Tensor, top_k: int) -> list[int]:
    channel_scores = activation.abs().mean(dim=(1, 2))
    top_k = min(top_k, activation.size(0))
    return torch.topk(channel_scores, k=top_k).indices.tolist()


def main() -> None:
    args = parse_args()
    model, checkpoint = load_checkpoint(args.checkpoint)

    data_dir = Path(checkpoint["data_dir"])
    image_size = int(checkpoint.get("image_size", 128))
    image_path = args.image_path if args.image_path is not None else default_image_path(data_dir)

    _, val_transform = build_transforms(image_size)

    activation: dict[str, torch.Tensor] = {}

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            activation[name] = output.detach().cpu()

        return hook

    hooks = [
        model.features[0].register_forward_hook(make_hook("block1")),
        model.features[1].register_forward_hook(make_hook("block2")),
        model.features[3].register_forward_hook(make_hook("block4")),
    ]

    class_names = {0: "cat", 1: "dog"}
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        original_image = image.copy()
        input_tensor = val_transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)
        prediction = logits.argmax(dim=1).item()

    for hook in hooks:
        hook.remove()

    block_names = ["block1", "block2", "block4"]
    fig, axes = plt.subplots(4, args.top_k, figsize=(2.2 * args.top_k, 9))

    for col in range(args.top_k):
        axes[0, col].axis("off")
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title(f"Input\npred={class_names[prediction]}")

    for row, block_name in enumerate(block_names, start=1):
        block_activation = activation[block_name].squeeze(0)
        channel_indices = top_channel_indices(block_activation, args.top_k)

        for col in range(args.top_k):
            ax = axes[row, col]
            ax.axis("off")
            if col >= len(channel_indices):
                continue

            channel_index = channel_indices[col]
            feature_map = normalize_map(block_activation[channel_index])
            ax.imshow(feature_map, cmap="viridis")
            ax.set_title(f"{block_name}\nch {channel_index}")

    fig.suptitle(f"Feature Maps for {image_path.name}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.output, dpi=200)
    plt.close(fig)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Image: {image_path}")
    print(f"Predicted class: {class_names[prediction]}")
    print(f"Saved feature visualization to: {args.output}")


if __name__ == "__main__":
    main()
