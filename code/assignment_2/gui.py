from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import torch
from PIL import Image, ImageDraw, ImageOps
from torch import nn


MODEL_PATH = Path(__file__).resolve().parent / "mnist_mlp.pth"
CANVAS_SIZE = 280
IMAGE_SIZE = 280
MNIST_SIZE = 28
LINE_WIDTH = 18
BACKGROUND = "white"
FOREGROUND = "black"


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


class DigitRecognizerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("MNIST Handwritten Digit Recognizer")
        self.root.resizable(False, False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.last_x: int | None = None
        self.last_y: int | None = None

        self.image = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=255)
        self.draw = ImageDraw.Draw(self.image)

        self.prediction_var = tk.StringVar(value="Prediction: -")
        self.confidence_var = tk.StringVar(value=f"Device: {self.device}")

        self._build_layout()
        self._reset_bars()

    def _load_model(self) -> MNISTMLP:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        checkpoint = torch.load(MODEL_PATH, map_location=self.device)
        model = MNISTMLP().to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=16)
        container.grid(row=0, column=0, sticky="nsew")

        left_panel = ttk.Frame(container)
        left_panel.grid(row=0, column=0, padx=(0, 16), sticky="n")

        right_panel = ttk.Frame(container)
        right_panel.grid(row=0, column=1, sticky="n")

        ttk.Label(left_panel, text="Draw a digit (0-9)", font=("Segoe UI", 14, "bold")).grid(
            row=0, column=0, pady=(0, 8)
        )

        self.canvas = tk.Canvas(
            left_panel,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg=BACKGROUND,
            cursor="cross",
            highlightthickness=1,
            highlightbackground="#999999",
        )
        self.canvas.grid(row=1, column=0)
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw_digit)
        self.canvas.bind("<ButtonRelease-1>", self.finish_drawing)

        controls = ttk.Frame(left_panel)
        controls.grid(row=2, column=0, pady=(12, 0), sticky="ew")
        ttk.Button(controls, text="Predict", command=self.predict_digit).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(controls, text="Clear", command=self.clear_canvas).grid(row=0, column=1)

        ttk.Label(right_panel, textvariable=self.prediction_var, font=("Segoe UI", 16, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(right_panel, textvariable=self.confidence_var).grid(row=1, column=0, sticky="w", pady=(4, 12))

        self.bar_canvas = tk.Canvas(right_panel, width=280, height=260, bg="white", highlightthickness=0)
        self.bar_canvas.grid(row=2, column=0)

        self.bar_items: list[tuple[int, int]] = []
        for digit in range(10):
            y = 12 + digit * 24
            self.bar_canvas.create_text(18, y + 8, text=str(digit), font=("Consolas", 11), anchor="w")
            rect = self.bar_canvas.create_rectangle(40, y, 40, y + 16, fill="#2d6a4f", width=0)
            label = self.bar_canvas.create_text(250, y + 8, text="0.0%", font=("Consolas", 10), anchor="e")
            self.bar_items.append((rect, label))

    def start_drawing(self, event: tk.Event) -> None:
        self.last_x = event.x
        self.last_y = event.y
        self._draw_point(event.x, event.y)

    def draw_digit(self, event: tk.Event) -> None:
        if self.last_x is None or self.last_y is None:
            self.start_drawing(event)
            return

        self.canvas.create_line(
            self.last_x,
            self.last_y,
            event.x,
            event.y,
            fill=FOREGROUND,
            width=LINE_WIDTH,
            capstyle=tk.ROUND,
            smooth=True,
        )
        self.draw.line((self.last_x, self.last_y, event.x, event.y), fill=0, width=LINE_WIDTH)
        self.last_x = event.x
        self.last_y = event.y

    def finish_drawing(self, _event: tk.Event) -> None:
        self.last_x = None
        self.last_y = None
        self.predict_digit()

    def _draw_point(self, x: int, y: int) -> None:
        radius = LINE_WIDTH // 2
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=FOREGROUND, outline=FOREGROUND)
        self.draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=0)

    def clear_canvas(self) -> None:
        self.canvas.delete("all")
        self.image = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_var.set("Prediction: -")
        self.confidence_var.set(f"Device: {self.device}")
        self._reset_bars()

    def _prepare_image(self) -> torch.Tensor:
        inverted = ImageOps.invert(self.image)
        bbox = inverted.getbbox()
        if bbox is None:
            raise ValueError("Please draw a digit before predicting.")

        digit = inverted.crop(bbox)
        digit.thumbnail((20, 20), Image.Resampling.LANCZOS)

        canvas = Image.new("L", (MNIST_SIZE, MNIST_SIZE), color=0)
        left = (MNIST_SIZE - digit.width) // 2
        top = (MNIST_SIZE - digit.height) // 2
        canvas.paste(digit, (left, top))

        tensor = torch.tensor(list(canvas.getdata()), dtype=torch.float32).view(1, 1, MNIST_SIZE, MNIST_SIZE) / 255.0
        tensor = (tensor - 0.1307) / 0.3081
        return tensor.to(self.device)

    def predict_digit(self) -> None:
        try:
            input_tensor = self._prepare_image()
        except ValueError as exc:
            self.prediction_var.set("Prediction: -")
            self.confidence_var.set(str(exc))
            self._reset_bars()
            return

        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

        predicted_digit = int(probabilities.argmax().item())
        confidence = float(probabilities[predicted_digit].item() * 100)

        self.prediction_var.set(f"Prediction: {predicted_digit}")
        self.confidence_var.set(f"Confidence: {confidence:.2f}%")
        self._update_bars(probabilities.tolist())

    def _reset_bars(self) -> None:
        for rect, label in self.bar_items:
            self.bar_canvas.coords(rect, 40, 0, 40, 0)
            self.bar_canvas.itemconfigure(label, text="0.0%")
        for digit, (rect, _label) in enumerate(self.bar_items):
            y = 12 + digit * 24
            self.bar_canvas.coords(rect, 40, y, 40, y + 16)

    def _update_bars(self, probabilities: list[float]) -> None:
        max_width = 180
        for digit, probability in enumerate(probabilities):
            y = 12 + digit * 24
            width = 40 + probability * max_width
            rect, label = self.bar_items[digit]
            self.bar_canvas.coords(rect, 40, y, width, y + 16)
            self.bar_canvas.itemconfigure(label, text=f"{probability * 100:5.1f}%")


def main() -> None:
    try:
        root = tk.Tk()
        app = DigitRecognizerApp(root)
    except FileNotFoundError as exc:
        messagebox.showerror("Missing Model", str(exc))
        return

    root.mainloop()


if __name__ == "__main__":
    main()
