"""
实验作业 1：深度学习基础与 PyTorch 入门
任务 A：自动微分实验
任务 B：MLP 拟合 sin(x)
任务 C：激活函数对比（可选）
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

print(f"PyTorch 版本: {torch.__version__}")
print(f"GPU 是否可用: {torch.cuda.is_available()}")
print()

# ─────────────────────────────────────────────
# 任务 A：自动微分实验
# ─────────────────────────────────────────────
print("=" * 50)
print("任务 A：自动微分实验")
print("=" * 50)

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

y = a ** 2 + b ** 2
y.backward()

print(f"a = {a.item()}, b = {b.item()}")
print(f"y = a^2 + b^2 = {y.item()}")
print(f"dy/da = 2a = {a.grad.item()}  (expected: {2 * a.item():.1f})")
print(f"dy/db = 2b = {b.grad.item()}  (expected: {2 * b.item():.1f})")
print()

# ─────────────────────────────────────────────
# 任务 B & C：MLP 拟合 sin(x) + 激活函数对比
# ─────────────────────────────────────────────
print("=" * 50)
print("任务 B & C：MLP 拟合 sin(x) + 激活函数对比")
print("=" * 50)

# 数据准备（跟随模型设备）
def make_data(n, device):
    x = torch.linspace(-2 * np.pi, 2 * np.pi, n).unsqueeze(1).to(device)
    return x, torch.sin(x)

_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_train, y_train = make_data(1000, _dev)
x_test, _        = make_data(300,  _dev)


class SimpleNet(nn.Module):
    def __init__(self, activation=nn.ReLU()):
        super(SimpleNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            activation,
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)


EPOCHS = 2000
LR = 0.01

activations = {
    "ReLU":    nn.ReLU(),
    "Tanh":    nn.Tanh(),
    "Sigmoid": nn.Sigmoid(),
    "GELU":    nn.GELU(),
}

history = {}   # {name: [loss_per_epoch]}
predictions = {}

for name, act in activations.items():
    model = SimpleNet(act).cuda() if torch.cuda.is_available() else SimpleNet(act)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    losses = []

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred = model(x_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    history[name] = losses

    model.eval()
    with torch.no_grad():
        predictions[name] = model(x_test).cpu().numpy()


    final_loss = losses[-1]
    print(f"{name:8s} | 最终 Loss: {final_loss:.6f}")

print()

# ─────────────────────────────────────────────
# 绘图
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 图1：训练 Loss 曲线
ax = axes[0]
for name, losses in history.items():
    ax.plot(losses, label=name)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.set_title("训练 Loss 下降曲线")
ax.legend()
ax.set_yscale("log")
ax.grid(True, alpha=0.3)

# 图2：拟合效果对比
ax = axes[1]
x_np = x_test.cpu().numpy().flatten()
ax.plot(x_np, np.sin(x_np), "k--", linewidth=2, label="sin(x) 真实值")
for name, pred in predictions.items():
    ax.plot(x_np, pred.flatten(), label=name, alpha=0.8)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("sin(x) 拟合效果对比")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("assignment1_result.png", dpi=150, bbox_inches="tight")
print("结果图已保存至 assignment1_result.png")
plt.show()
