# GDL Course — Geometric Deep Learning

本仓库用作中国科大2026春季《几何深度学习》课程作业代码存放。

## 目录结构

```
gdl_course/
├── code/       # 课程作业代码
├── data/       # 数据集
└── slides/     # 课件
```

## 任务概要

### assignment_1

- 完成基础环境检查与课程首次编程作业提交。

### assignment_2

- 基于 PyTorch 在 MNIST 数据集上实现两层隐藏层的 MLP 手写数字分类模型。
- 训练脚本支持自动下载数据、CPU/GPU 自适应训练、保存最佳模型，并输出损失与准确率曲线。
- 提供手写数字识别 GUI，支持画板输入、实时预测以及 0-9 类别置信度可视化。
- 当前实验报告记录的最佳测试集准确率为 `98.01%`。

### assignment_3

- 基于 Dogs vs. Cats 数据集完成卷积神经网络图像分类实验。
- `train_scratch.py` 从零搭建 4 个卷积块的 CNN，并保存 `best_cnn.pth` 与 `cnn_scratch_curves.png`。
- `finetune.py` 基于 ImageNet 预训练 `ResNet18` 实现两阶段迁移学习，并保存 `best_phase2.pth` 与 `finetune_curves.png`。
- `visualize_features.py` 对任务一训练好的 CNN 做中间层特征图可视化，并生成 `cnn_feature_maps.png`。
- 当前实验报告记录的最佳验证集准确率为：任务一 `91.58%`，任务二 `99.22%`。

## 参考资料

- [Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges](https://geometricdeeplearning.com/)
- [PyTorch Geometric 文档](https://pytorch-geometric.readthedocs.io/)
