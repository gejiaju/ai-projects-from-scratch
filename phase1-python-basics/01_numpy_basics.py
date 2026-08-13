# -*- coding: utf-8 -*-
"""
01_numpy_basics.py —— NumPy 数组操作入门

这个文件是干什么的：
    深度学习里几乎所有数据都是"张量"（tensor），而 NumPy 的 ndarray 就是
    张量的"原型"。PyTorch 的张量用法和它几乎一模一样。本文件演示 AI 实际
    用得到的 NumPy 操作：创建数组、形状变换、索引切片、广播、矩阵运算、
    统计函数、随机数。

跑完能看到什么：
    终端里打印出每一步操作的结果，以及一段"为什么这在 AI 里有用"的说明。

怎么跑：
    python3 01_numpy_basics.py
"""

import numpy as np


def section(title):
    """打印一个分隔标题，让输出更易读。"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    # 1. 创建数组：深度学习里，一张图片就是一个 (高, 宽, 通道) 的数组
    section("1. 创建数组")
    # 全 0 数组：常用于初始化一个全零的 bias（偏置）
    zeros = np.zeros((2, 3))
    print("np.zeros((2,3)) 全零数组:\n", zeros)

    # 全 1 数组
    ones = np.ones(4)
    print("\nnp.ones(4) 全一数组:\n", ones)

    # 用 list 直接构造：模拟一张 2x2 的"灰度图"，数值代表亮度 0~255
    image = np.array([[0, 255],
                      [128, 64]])
    print("\n一张 2x2 的灰度图(数值=亮度):\n", image)

    # arange + reshape：生成 0~11 共 12 个数，摆成 3 行 4 列
    matrix = np.arange(12).reshape(3, 4)
    print("\nnp.arange(12).reshape(3,4):\n", matrix)

    # 2. 形状与索引：取出数组的某一块（切片），是数据预处理的基本功
    section("2. 形状与索引")
    print("matrix 的形状 shape:", matrix.shape)      # (3, 4)
    print("matrix 的维度 ndim:", matrix.ndim)         # 2
    print("元素总个数 size:", matrix.size)            # 12
    print("matrix[1, 2] 第2行第3列:", matrix[1, 2])   # 6
    print("matrix[:, 1] 第2列(所有行):", matrix[:, 1])  # [1 5 9]

    # 3. 广播（broadcasting）：不同形状的数组也能一起运算
    section("3. 广播 broadcast")
    a = np.array([[1, 2, 3],
                  [4, 5, 6]])          # 形状 (2, 3)
    b = np.array([10, 20, 30])         # 形状 (3,)
    # b 会被自动"复制"成 (2,3) 再逐元素相加——这就是广播
    print("a + b (b 自动扩展到两行):\n", a + b)

    # 4. 矩阵运算：神经网络的前向传播本质就是 矩阵乘法 + 偏置
    section("4. 矩阵运算")
    weights = np.random.randn(3, 2)    # 3 个输入 → 2 个输出 的权重矩阵
    inputs = np.array([[1.0, 2.0, 3.0]])  # 1 条样本，3 个特征
    # @ 是矩阵乘法： (1,3) @ (3,2) -> (1,2)
    output = inputs @ weights
    print("inputs(1,3) @ weights(3,2) = output(1,2):\n", output)
    print("说明：一次矩阵乘法就完成了一次'全连接层'的前向计算。")

    # 5. 统计函数：评估模型效果时天天用
    section("5. 统计函数")
    scores = np.array([92, 85, 78, 95, 88, 73])   # 一组"准确率"样本
    print("一组准确率 scores:", scores)
    print("均值 mean:   ", scores.mean())
    print("标准差 std:   ", scores.std())          # 波动越大 std 越大
    print("最大 max:     ", scores.max())
    print("最小 min:     ", scores.min())
    print("求和 sum:     ", scores.sum())
    # argmax：最大值在第几个位置（分类模型最后一步就用它挑概率最大的类别）
    print("最大值位置 argmax:", scores.argmax())

    # 6. 随机数：固定随机种子(seed)能让每次实验"可复现"
    section("6. 随机数")
    rng = np.random.default_rng(42)   # 固定种子 42
    print("种子42生成的正态分布样本:", rng.normal(0, 1, 5))
    rng2 = np.random.default_rng(42)
    print("同样种子42再来一次:     ", rng2.normal(0, 1, 5))
    print("说明：种子相同 → 结果相同，这是实验能复现的关键。")


if __name__ == "__main__":
    main()
