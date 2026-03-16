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

## 参考资料

- [Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges](https://geometricdeeplearning.com/)
- [PyTorch Geometric 文档](https://pytorch-geometric.readthedocs.io/)
