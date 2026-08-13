# -*- coding: utf-8 -*-
"""
03_class_example.py —— class（类）写法

这个文件是干什么的：
    PyTorch 里定义任何模型，都要写 class 并继承 nn.Module，然后实现
    __init__ 和 forward 方法。本文件用一个"手写的全连接层"演示 class 的
    核心概念：__init__ 存参数，方法做计算。看懂这个，第二阶段写神经网络
    就非常自然了。

跑完能看到什么：
    终端打印一次"前向传播"的计算过程：输入 → 权重/偏置 → 输出。
    同时演示了两个对象互不干扰（各自有自己的参数）。

怎么跑：
    python3 03_class_example.py
"""

import numpy as np


class LinearLayer:
    """一个最简化的全连接层（Linear Layer）。

    输入 n_in 个特征，输出 n_out 个特征。
    公式：output = input @ weight + bias
    """

    def __init__(self, n_in, n_out, seed=0):
        """__init__ = 构造函数：对象创建时自动调用，用来初始化参数。

        weight(权重) 和 bias(偏置) 就是这个层要"学习"的东西。
        """
        rng = np.random.default_rng(seed)
        # 权重形状 (n_in, n_out)，用小的随机数初始化
        self.weight = rng.normal(0, 0.1, size=(n_in, n_out))
        # 偏置形状 (n_out,)，从 0 开始
        self.bias = np.zeros(n_out)

    def forward(self, x):
        """forward = 前向传播：给定输入 x，算出输出。"""
        return x @ self.weight + self.bias

    def __repr__(self):
        """让 print(对象) 时显示得清楚一点。"""
        return f"LinearLayer(in={self.weight.shape[0]}, out={self.weight.shape[1]})"


class TwoLayerNet:
    """把两个 LinearLayer 串起来，就是一个最小的"两层神经网络"。

    这已经和 PyTorch 里 nn.Sequential 的思路一模一样了：数据一层一层往前流。
    """

    def __init__(self):
        self.fc1 = LinearLayer(3, 4, seed=1)   # 3 输入 -> 4 隐藏
        self.fc2 = LinearLayer(4, 2, seed=2)   # 4 隐藏 -> 2 输出

    def forward(self, x):
        hidden = self.fc1.forward(x)
        # 加一个简单的 ReLU 激活：负数变 0（非线性，这是网络"有深度"的关键）
        hidden = np.maximum(0, hidden)
        output = self.fc2.forward(hidden)
        return output


def main():
    # 1. 创建对象（会触发 __init__）
    layer = LinearLayer(n_in=3, n_out=2, seed=0)
    print("创建了一个层:", layer)
    print("权重 weight:\n", layer.weight)
    print("偏置 bias:  ", layer.bias)

    # 2. 前向传播：一条有 3 个特征的样本
    x = np.array([[1.0, 2.0, 3.0]])
    out = layer.forward(x)
    print("\n输入 x:", x)
    print("输出 out = x @ weight + bias:", out)

    # 3. 两个对象互不干扰
    layer_a = LinearLayer(3, 2, seed=7)
    layer_b = LinearLayer(3, 2, seed=8)
    print("\n两个对象的权重是否相同:", np.array_equal(layer_a.weight, layer_b.weight))

    # 4. 两层网络串联
    print("\n" + "=" * 60)
    net = TwoLayerNet()
    print("两层网络前向传播: 输入(1,3) -> 输出(1,2)")
    print(net.forward(x))


if __name__ == "__main__":
    main()
