# 作业三：卷积神经网络（CNN）与猫狗图像分类

**发布日期**：2026 年 3 月 15 日

**截止日期**：2026 年 3 月 22 日 23:59（CST）

---

## 一、作业背景

卷积神经网络（CNN, Convolutional Neural Network）是计算机视觉领域最核心的模型架构，也是几何深度学习的重要基础。上周我们用 MLP 在 MNIST 上取得了极高的准确率，但如果任务变成彩色图像分类，MLP 会面临参数爆炸与感知结构缺失的双重困境。CNN 通过**权重共享**与**局部连接**两种机制，将图像的空间结构先验直接编码进网络，使其在图像任务上远优于 MLP。

本次作业以 Kaggle 经典竞赛 **Dogs vs. Cats** 为实验场景，在 25,000 张猫狗图像上完成两个核心任务：

- **任务一**：从零搭建一个 CNN，理解卷积、BatchNorm、池化等基础组件
- **任务二**：基于预训练 ResNet18 进行迁移学习，体验"站在巨人肩膀上"的力量

---

## 二、数据集准备

### 方法一：Kaggle CLI（推荐）

```bash
pip install kaggle
# 将 kaggle.json 凭证放入 ~/.kaggle/ 后执行：
kaggle competitions download -c dogs-vs-cats
unzip dogs-vs-cats.zip
unzip train.zip -d data/
```

如何获取 `kaggle.json`：登录 [Kaggle](https://www.kaggle.com) → Account → API → Create New Token。

### 方法二：浏览器手动下载

1. 访问 [Dogs vs. Cats 数据页面](https://www.kaggle.com/c/dogs-vs-cats/data)（需注册 Kaggle 账号）
2. 下载 `train.zip`，解压到项目根目录下的 `data/train/` 文件夹

### 目录结构

解压完成后，`data/train/` 目录中应含如下文件：

```
data/
  train/
    cat.0.jpg
    cat.1.jpg
    ...         (共 12,500 张猫图)
    dog.0.jpg
    dog.1.jpg
    ...         (共 12,500 张狗图)
```

代码将按 **80 / 20** 比例自动划分训练集（20,000 张）和验证集（5,000 张）。

---

## 三、作业内容

### 任务一：从零搭建 CNN（`train_scratch.py`）

编写 Python 脚本，完成以下功能：

1. **自定义数据集**：编写继承 `torch.utils.data.Dataset` 的类，从目录读取图像，根据文件名前缀（`cat` / `dog`）自动解析标签（0 / 1）。

2. **数据增强**：训练时使用 `RandomHorizontalFlip`、`RandomRotation`、`ColorJitter`；验证时仅做 Resize + Normalize（不使用随机增强）。

3. **模型搭建**：搭建至少包含 **4 个卷积块** 的 CNN，每块包含：

   ```
   Conv2d(kernel=3, padding=1) → BatchNorm2d → ReLU → MaxPool2d(2×2)
   ```

   通道数建议沿网络加深逐步翻倍（如 32 → 64 → 128 → 256）。最后用 `AdaptiveAvgPool2d(1)` 压缩空间维度，再接全连接分类头。

4. **训练循环**：Adam 优化器 + `CosineAnnealingLR` 学习率调度，每轮在验证集上评估，保存最佳模型权重。

5. **可视化**：绘制训练/验证 Loss 和 Accuracy 曲线，保存为 `cnn_scratch_curves.png`。

**代码框架（供参考）**：

```python
class ConvBlock(nn.Module):
    """基础卷积块：Conv → BN → ReLU → MaxPool"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.block(x)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            # TODO: 继续添加卷积块，使总深度不少于 4 块
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # 将任意空间尺寸压至 1×1
        self.classifier = nn.Sequential(
            # TODO: 添加 Dropout + 全连接层，输出 2 类
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # 展平
        return self.classifier(x)
```

---

### 任务二：迁移学习（`finetune.py`）

编写 Python 脚本，基于预训练 ResNet18 完成迁移学习，**分两阶段**进行训练：

| 阶段 | 操作 | 学习率 | Epoch 数 |
|------|------|--------|---------|
| **阶段一（热身）** | 冻结全部骨干网络，只训练新的分类头 | `1e-3` | 5 |
| **阶段二（微调）** | 解冻全部参数，端到端微调 | 分类头 `1e-3`，骨干 `1e-4` | 10 |

**关键代码提示**：

```python
from torchvision import models

# 加载带 ImageNet 预训练权重的 ResNet18
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# 阶段一：冻结骨干，只留 fc 可训
for param in resnet.parameters():
    param.requires_grad = False

# 替换分类头（ResNet18 的 fc.in_features = 512）
resnet.fc = nn.Linear(resnet.fc.in_features, 2)

# 阶段二：解冻所有参数
for param in resnet.parameters():
    param.requires_grad = True

# Hint：使用参数组（param groups）为不同层设置不同学习率
optimizer = optim.Adam([
    {'params': resnet.fc.parameters(),      'lr': 1e-3},
    {'params': ...,                          'lr': 1e-4},  # 骨干参数
])
```

生成并保存训练曲线 `finetune_curves.png`，图中需用竖线标注两阶段的分界点。

---

### 任务三（可选）：特征图可视化

选择任务一中训练好的 CNN，对**同一张图像**提取不同卷积层（第 1 块、第 2 块、第 4 块）的特征图，展示并分析浅层与深层特征的视觉差异。

**Hint**：使用 `register_forward_hook` 钩子在前向传播时捕获中间层输出：

```python
activation = {}

def make_hook(name):
    def hook(module, input, output):
        activation[name] = output.detach()
    return hook

# 注册钩子
model.features[0].register_forward_hook(make_hook('block1'))

# 前向传播后，activation['block1'] 即为第 1 卷积块的输出
```

---

## 四、具体要求

| 项目 | 要求 |
|------|------|
| 编程语言 | Python 3.8+ |
| 深度学习框架 | PyTorch |
| 任务一模型 | 至少 4 个卷积块（Conv2d + BN + ReLU + MaxPool） |
| 任务一精度 | 验证集准确率 ≥ **85%** |
| 任务二模型 | 基于预训练 ResNet18，实现两阶段微调 |
| 任务二精度 | 验证集准确率 ≥ **95%** |
| GPU 支持 | 代码自动检测 CUDA，兼容纯 CPU 运行 |
| 代码规范 | 结构清晰，关键步骤有注释 |

---

## 五、运行方式

```bash
# 任务一：从零训练 CNN
python train_scratch.py

# 任务二：迁移学习微调
python finetune.py
```

> **提示**：任务一在 CPU 上训练 20 个 epoch 约需 60–90 分钟；建议使用 GPU 或将 `IMAGE_SIZE` 调小至 64 先验证流程。任务二使用预训练权重，即使只训练 15 个 epoch 也能快速收敛。

---

## 六、检查方法

1. **训练流程**：两个脚本均能正常运行至收敛，控制台打印每 epoch 的 Loss 和 Accuracy。
2. **精度达标**：任务一验证集 ≥ 85%，任务二验证集 ≥ 95%。
3. **曲线图**：生成 `cnn_scratch_curves.png` 和 `finetune_curves.png`，观察是否有过拟合现象，以及迁移学习相比从零训练的收敛速度差异。
4. **模型文件**：运行后生成 `best_cnn.pth`（任务一）和 `best_phase2.pth`（任务二）。

---

## 七、思考问题（可选）

> 以下思考题按照**网络结构从输入到输出、历史发展从早到晚**的顺序排列：先理解为什么需要 CNN（Q1），再理解 CNN 的完整结构（Q2），深入各训练组件（Q3、Q4），跟随历史演进到 ResNet（Q5）和迁移学习（Q6），最后以宏观视角对比不同架构（Q7）。

---

### Q1｜CNN vs MLP 

对于一张 $224 \times 224 \times 3$ 的 RGB 图像，若用全连接层处理，第一层（输出 512 个神经元）需要多少参数？而本次作业第一个 ConvBlock 中等效的 $3 \times 3$ 卷积层（32 个滤波器）只需要多少参数？

计算完这两个数字后，回答：参数量的巨大差异背后，CNN 利用了图像数据的哪两种结构性先验？若去掉其中任意一个先验，网络会退化成什么形式？

> **Hint**：猫耳朵出现在图像左上角和右下角时，检测它们是否需要两套独立的权重？把这个直觉翻译成数学术语：卷积核在所有空间位置**共享**同一组参数，对应了图像的什么物理性质？而每个神经元只连接 $3 \times 3$ 邻域而非全图，对应了什么假设？

---

### Q2｜CNN 结构：特征提取器与分类头

我们的 `SimpleCNN` 最后使用了与作业 2 相同结构的分类头（两层线性 + 一层 ReLU），整体结构如下：

```
ConvBlock × 4  →  AdaptiveAvgPool  →  Linear(256→128) → ReLU → Linear(128→2)
     ↑                  ↑                          ↑
  空间特征提取        全局聚合               MLP 分类头（与作业 2 完全相同）
```

(a) CNN 卷积层和最后的 MLP 分类头各自承担什么职责？能否互换——用 MLP 提取空间特征、用 CNN 做最终分类决策？

(b) 如果 `AdaptiveAvgPool` 输出的 256 维向量已经是好的语义特征，直接用 `Linear(256→2)` 分类可以吗？加入中间隐藏层（`256→128→ReLU→2`）的意义是什么？

(c) 在任务二中，阶段一只训练 `fc` 层（冻结 CNN 骨干），就能达到约 95% 的准确率。如果把任务从"猫 vs 狗"换成"猫的情绪分类（高兴 / 生气）"，你需要重新训练哪个部分？

> **Hint**：阶段一的做法叫做 **Linear Probing（线性探针）**——如果线性探针效果已经很好，说明 CNN 骨干输出的特征在 256 维空间中对目标类别是**线性可分**的。这对"CNN 在学什么"意味着什么？迁移到情绪分类时，猫的视觉外观特征（毛色、耳型）和情绪特征（嘴角弧度、眼睛开合）分别由网络的哪个部分负责？

---

### Q3｜BatchNorm

每个 `ConvBlock` 都在 `Conv2d` 之后紧跟一个 `BatchNorm2d`。

(a) Conv2d 是一个线性变换，它会保持输入向量的范数（长度）吗？若不保持，深层网络中逐层叠加会带来什么后果？

(b) BatchNorm 具体执行了哪些操作？它的可学习参数 $\gamma$ 和 $\beta$ 各自扮演什么角色？为什么不能去掉这两个参数，直接固定为"零均值单位方差"？

(c) 在**推理阶段**，BatchNorm 应该使用当前 batch 的统计量，还是训练时积累的全局统计量？为什么必须如此？

> **Hint**：
> - 对于 (a)：一个随机初始化的矩阵 $W$，$\|Wx\|$ 与 $\|x\|$ 的比值（即算子范数）是多少？叠加 $L$ 层后这个比值如何增长？
> - 对于 (b)：如果强制所有层输出都是"零均值单位方差"，网络还能学到"某层应该输出全正数激活"这类有意义的模式吗？$\gamma$ 和 $\beta$ 提供了什么自由度？
> - 对于 (c)：把 batch size 改为 1，batch 统计量 $\mu = \bar{x}$ 等于什么？此时 $\hat{x} = \frac{x - \mu}{\sigma}$ 会发生什么？

---

### Q4｜数据增强

我们使用了 `RandomHorizontalFlip`、`RandomRotation(15°)`、`ColorJitter` 三种增强。

(a) 每种增强背后各编码了什么**不变性假设**（即"这种变换不改变类别标签"）？

(b) 如果把任务改为**识别车牌号码**，这三种增强哪些应该保留、哪些必须去掉？请说明理由。

(c) 数据增强（在训练数据上施加变换）与几何深度学习中"将对称性直接编码进网络结构"（如卷积的权重共享）有何异同？各自的代价和收益是什么？

> **Hint**：对于 (b)：水平翻转后，"6"的镜像在车牌中是合法字符吗？对于 (c)：数据增强是在"告诉模型这些变换不重要"，而权重共享是"结构上保证这些变换不重要"——哪种方式需要更多数据才能学会不变性？哪种方式可能误伤（即错误地声明某些变换不重要）？

---

### Q5｜从 CNN 到 ResNet

本次作业的 `SimpleCNN` 使用 BatchNorm 稳定训练；任务二微调的 ResNet18 额外引入了**残差连接** $x_{l+1} = F(x_l, W_l) + x_l$。

(a) 对于 $L$ 层深网络，反向传播梯度为 $\dfrac{\partial \mathcal{L}}{\partial x_0} = \dfrac{\partial \mathcal{L}}{\partial x_L} \cdot \prod_{l=1}^{L} J_l$。BatchNorm 稳定了前向激活分布，但能阻止这个连乘积指数级衰减吗？为什么？

(b) 对 $x_L = x_l + \sum_{i=l}^{L-1} F(x_i)$ 求 $\dfrac{\partial x_L}{\partial x_l}$，结果中出现了哪一项使得梯度不再消失？

(c) 若某层最优变换 $H(x) \approx x$：没有残差时需要让 $F(x) \approx x$，有残差时只需让 $F(x) \approx 0$。从权重初始化的角度解释，为什么后者在训练初期更容易实现？

(d) ResNet 同时使用了 BN 和残差连接，填写它们各自解决的问题：

| 技术 | 解决的问题 | 作用方向 | 单独使用能训练 1000 层吗？ |
|------|-----------|---------|--------------------------|
| BatchNorm | ？ | 前向传播 | ？ |
| 残差连接 | ？ | ？ | ？ |
| 两者结合 | ？ | 双向 | ？ |

> **Hint**：对 $x_L = x_l + \sum F(x_i)$ 关于 $x_l$ 求偏导，注意 $x_l$ 本身作为加法项出现——它的偏导是什么常数矩阵？这个常数矩阵如何保证梯度"至少不消失"？再想想：以小值初始化权重时，$F(x) = W_2 \sigma(W_1 x)$ 的初始输出接近多少？

---

### Q6｜迁移学习

ResNet18 预训练于 ImageNet 的 1,000 类自然图像。

(a) 如果将其迁移到**医学 CT 切片分类**或**卫星遥感图像分析**，迁移学习仍然有效吗？浅层（Block1–2）和深层（Block4+）的特征分别应该如何处理（冻结 / 微调 / 随机初始化重训）？

(b) 根据任务二两阶段训练的典型结果（阶段一 ≈ 95%，阶段二 ≈ 98%），解释：为什么阶段一不训练骨干就能接近 95%？阶段二解冻后为何还能提升到 98%？

(c) 如果目标域数据极少（仅 500 张图），全量微调会带来什么风险？这个风险叫做什么？如何通过学习率设置缓解它？

> **Hint**：对于 (a)：可视化 ResNet 第一层卷积核，会看到边缘检测器、颜色块检测器——这些模式在猫、狗、汽车、CT 切片中都存在吗？越浅的层学到的特征越"通用"还是越"特殊"？对于 (c)：用 500 张图、较大学习率微调一个拥有 1100 万参数的模型，训练集 loss 能降到很低但验证集 loss 会怎样？这和作业 2 中 Dropout 解决的是同一类问题吗？

---

### Q7｜架构比较

> 本题现在只要求完成 MLP 和 CNN 两行；等课程介绍 GNN 和 Self-Attention 后，请回来补全整张表格。

**（一）填表**

| 机制 | 连接方式 | 权重共享 | 编码的对称性 | 适用数据结构 |
|------|---------|---------|------------|------------|
| **MLP** | ？ | ？ | 无（平凡群） | ？ |
| **CNN** | ？ | ？ | ？（平移等变） | ？ |
| **GNN** | ？ | ？ | ？ | ？（图结构数据） |
| **Self-Attention** | ？ | ？ | ？ | ？ |

> **Hint**：CNN 的每个神经元只连接 $K \times K$ 邻域，且所有位置共用同一套卷积核——如果把图像像素看作图节点，CNN 对应的是什么样的图（稀疏还是稠密？固定还是动态？）。Self-Attention 中，每个 token 与所有其他 token 相连，边权重由 Query-Key 内积动态计算——这对应什么图？两者都是 GNN 的特例，差别仅在于**图的结构**和**边权重的来源**。

**（二）小计算题：CNN 需要多少层才能看到全局？**

设输入 $H \times H$，使用 $3\times3$ Conv（stride=1，same padding）。

(a) 用归纳法证明：$L$ 层后每个神经元的**感受野**（Receptive Field）为 $(2L+1)\times(2L+1)$。

(b) 对 $128\times128$ 图像，纯 $3\times3$ Conv（无任何池化）至少需要多少层才能让感受野覆盖整张图？

(c) 若每层卷积后接 MaxPool(2×2)，感受野增长速度如何变化？计算本次作业 `SimpleCNN`（4 个 ConvBlock）结束时的感受野大小，并说明 `AdaptiveAvgPool` 如何弥补感受野尚未覆盖全图的不足。

(d) Self-Attention 经过几层就能让每个 token 看到所有其他 token？与 (b) 的数字对比，说明 CNN 和 Self-Attention 在**信息传播路径长度**上的本质差异，以及这个差异带来的计算代价权衡。

> **Hint**：感受野递推公式为 $\text{RF}_l = \text{RF}_{l-1} + (k_l - 1)\cdot\prod_{i < l}s_i$，其中 $s_i$ 是第 $i$ 层的步幅。MaxPool(2×2) 的步幅 $s=2$ 如何影响后续所有层的感受野增长速度？对 $128\times128$ 图像，令 $2L+1 \geq 128$，解出 $L$。

**（三）进阶思考**

几何深度学习（GDL）认为，不同架构的本质区别是对输入数据所假设的**对称群**不同。**CNN 是 GNN 在规则网格图上施加权重共享约束后的特例**——去掉"规则网格"约束就得到 GNN，去掉"权重共享"约束就得到 Locally Connected Network，两个约束都去掉再加全局注意力就得到 Transformer。你目前能写出 MLP 和 CNN 分别对应的对称群吗？

---

## 八、提交要求

将以下文件打包为 `学号_姓名_HW3.zip`：

- `train_scratch.py` — 从零搭建 CNN 的训练脚本
- `finetune.py` — 迁移学习微调脚本
- `cnn_scratch_curves.png` — 任务一训练曲线截图
- `finetune_curves.png` — 任务二两阶段训练曲线截图
- （可选）特征图可视化图片

通过课程 QQ 群的作业系统上传。

---

## 九、参考依赖

```
torch
torchvision
matplotlib
Pillow
```
