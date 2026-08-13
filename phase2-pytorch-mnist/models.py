# -*- coding: utf-8 -*-
"""
models.py —— 手写 MLP 和 CNN 两个模型

这个文件是干什么的：
    定义两个做 MNIST 手写数字识别的模型：
      - MLP：多层感知机（只有全连接层），把图片展平成 784 个数再处理
      - CNN：卷积神经网络（卷积层 + 池化层 + 全连接层），直接看图片的"结构"
    两个模型都用 PyTorch 的 nn.Module 手写，一个类一个类搭起来。

跑完能看到什么：
    本文件不单独跑，由 train.py 导入。单独运行会打印两个模型的参数量对比，
    这个数字是面试里很关键的一个点（为什么 CNN 参数更少却更准）。

怎么跑（单独查看参数量）：
    python3 models.py
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """多层感知机：全连接层堆叠。

    做法：把 28x28 的图片"拍平"成 784 个数，然后经过几层全连接。
    缺点：拍平后，像素之间的"位置关系"就丢了——这是它在图像上不如 CNN 的根本原因。
    """

    def __init__(self, input_dim=28 * 28, hidden_dims=(128, 64), num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()  # (N, 1, 28, 28) -> (N, 784)

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))  # 全连接层
            layers.append(nn.ReLU())                        # 激活函数（引入非线性）
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))     # 最后一层输出 10 个分数
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)


class CNN(nn.Module):
    """卷积神经网络：卷积层 + 池化层 + 全连接层。

    做法：卷积层用一个小"窗口"(3x3) 在图片上滑动，提取边缘/纹理等局部特征；
    池化层把特征图缩小（保留主要信息、减少计算量）；最后全连接层做分类。
    关键优势：卷积核的参数是"共享"的——不管图片哪个位置，都用同一组权重，
    所以参数少、还能抓住"局部结构"。
    """

    def __init__(self, num_classes=10):
        super().__init__()
        # 输入形状 (N, 1, 28, 28)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # -> (32, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> (32, 14, 14)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> (64, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> (64, 7, 7)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),                              # 随机丢弃一半神经元，防过拟合
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)                 # 提取特征
        x = x.view(x.size(0), -1)            # 展平 (N, 32*7*7)
        return self.classifier(x)            # 分类


def count_parameters(model):
    """统计模型里可训练参数的个数（单位：个）。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 单独运行：对比两个模型的参数量
    mlp = MLP()
    cnn = CNN()
    print(f"MLP 参数量: {count_parameters(mlp):,}")
    print(f"CNN 参数量: {count_parameters(cnn):,}")
    print("\n注意：CNN 参数更少，但准确率更高。这正是'卷积权重共享 + 局部感受野'的价值。")

    # 快速验证前向传播的形状是否正确
    dummy = torch.randn(2, 1, 28, 28)   # 假装 2 张 28x28 的灰度图
    print("\n输入形状:", tuple(dummy.shape))
    print("MLP 输出形状:", tuple(mlp(dummy).shape), "(应为 (2, 10))")
    print("CNN 输出形状:", tuple(cnn(dummy).shape), "(应为 (2, 10))")
