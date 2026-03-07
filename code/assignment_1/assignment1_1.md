# 几何深度学习 (Geometric Deep Learning)

## 实验作业 1：深度学习基础与 PyTorch 入门

**发布日期**：2026年3月1日

**截止日期**：2026年3月8日 23:59 (CST)

## 1. 实验目标

本周实验旨在帮助大家跨越从“数学公式”到“工程实现”的鸿沟。你将完成：

* **环境搭建**：配置支持 GPU 加速的深度学习开发环境。
* **理论内化**：通过代码理解自动微分（Autograd）与计算图。
* **手写实践**：构建并训练一个简单的多层感知机（MLP）。

## 2. 环境配置指南

对于数学学院的学生，我们推荐使用 **Miniconda** 方案，它是最轻量且逻辑清晰的隔离环境管理工具。

### 第一步：安装驱动与基础环境

1. **CUDA Driver**: 请确保你的机器已安装 NVIDIA 驱动。
2. **Miniconda**: 下载并安装 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)。

### 第二步：创建虚拟环境

打开终端（Terminal 或 PowerShell），执行以下命令：

```bash
# 创建名为 gdl_env 的环境，使用 Python 3.10
conda create -n gdl_env python=3.10 -y
conda activate gdl_env

# 安装 PyTorch 生态（以 CUDA 12.1 为例，请根据官网指令调整）
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

```

### 第三步：验证安装

创建一个 `check_gpu.py` 文件并运行：

```python
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"GPU 是否可用: {torch.cuda.is_available()}")

```

## 3. 核心概念回顾

在编写代码前，请建立以下数学对象与 PyTorch 类的映射关系：

| 数学对象                                                                      | PyTorch 实现                         | 备注                         |
| ----------------------------------------------------------------------------- | ------------------------------------ | ---------------------------- |
| **张量 (Tensor) $x \in \mathbb{R}^{n \times m}$**                             | `torch.Tensor`                       | 支持自动求导的数据容器       |
| **映射 (Mapping) $f(x; \theta)$**                                             | `nn.Module`                          | 神经网络层或模型             |
| **目标函数 (Objective) $\mathcal{L}$**                                        | `nn.MSELoss` / `nn.CrossEntropyLoss` | 计算预测值与真实值的标量差异 |
| **梯度下降 (GD) $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$** | `torch.optim.Optimizer`              | 负责根据梯度更新参数         |

## 4. 代码作业任务

请在提供的 `assignment1_demo.ipynb`（或自行创建脚本）中完成以下内容：

### 任务 A：自动微分实验

定义函数 $y = a^2 + b^2$，其中 $a=2, b=3$。利用 PyTorch 的 `autograd` 计算 $\frac{\partial y}{\partial a}$ 和 $\frac{\partial y}{\partial b}$。

* **要点**：设置 `requires_grad=True`。

### 任务 B：跑通 MLP 训练 Demo

利用 PyTorch 构建一个包含一个隐藏层的神经网络，拟合正弦函数 $f(x) = \sin(x)$。

**代码框架示例：**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 定义网络结构
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)

# 2. 准备数据、损失函数与优化器
model = SimpleNet().cuda() if torch.cuda.is_available() else SimpleNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 3. 训练循环 (请补充完整)
# Hint: zero_grad() -> forward -> loss -> backward -> step()

```

### 任务 C: 尝试不同的激活函数（可选）

了解深度学习常用的激活函数，总结它们的发展史和改进点。并在任务 B 的基础上尝试它们，分析不同激活函数的表现和原因。

## 5. 提交要求

请将以下文件打包为 `学号_姓名_HW1.zip`：

1. **环境配置截图**：包含 `check_gpu.py` 的运行结果。
2. **代码脚本**：`.py` 或 `.ipynb` 文件。
3. **结果图表**：训练 Loss 下降的曲线图（使用 `matplotlib` 绘制）。
4. **激活函数分析报告**（可选）：形式、内容自由。

通过课程 QQ 群的作业系统上传。
