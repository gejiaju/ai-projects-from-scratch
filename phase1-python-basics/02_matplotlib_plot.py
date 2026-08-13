# -*- coding: utf-8 -*-
"""
02_matplotlib_plot.py —— Matplotlib 画图入门

这个文件是干什么的：
    训练完模型后，你需要把"损失曲线""准确率曲线""不同模型对比"画成图，
    这些图会直接放进你的技术报告和 PPT 里。本文件演示 Matplotlib 最常用的
    三种图：折线图、柱状图、散点图，并演示子图(subplot)排版。

跑完能看到什么：
    当前目录下生成 4 张 PNG 图片：
      - line_plot.png    折线图（模拟训练损失下降）
      - bar_plot.png     柱状图（不同模型准确率对比）
      - scatter_plot.png 散点图（准确率 vs 推理时间）
      - subplots.png     三合一子图排版

怎么跑：
    python3 02_matplotlib_plot.py
"""

import matplotlib
matplotlib.use("Agg")  # 无界面环境也能画图并保存
import matplotlib.pyplot as plt
import numpy as np

# 统一风格：所有图用同一套字号/配色，方便后面放进 PDF 和 PPT
plt.rcParams.update({
    "figure.figsize": (7, 4.5),
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def draw_line():
    """折线图：模拟训练过程中 loss 逐渐下降。"""
    epoch = np.arange(1, 21)                       # 第 1~20 轮
    loss = 2.3 * np.exp(-epoch / 6) + 0.15 + np.random.default_rng(0).normal(0, 0.02, 20)
    plt.figure()
    plt.plot(epoch, loss, marker="o", linewidth=2, color="#1f77b4", label="Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve (simulated)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("line_plot.png", dpi=150)
    plt.close()
    print("已生成 line_plot.png")


def draw_bar():
    """柱状图：不同模型在同一数据集上的准确率对比。"""
    models = ["MLP", "CNN", "ResNet"]
    accuracy = [97.2, 99.1, 99.5]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    plt.figure()
    bars = plt.bar(models, accuracy, color=colors, width=0.55)
    for bar, acc in zip(bars, accuracy):           # 在柱子上标数字
        plt.text(bar.get_x() + bar.get_width() / 2, acc + 0.2,
                 f"{acc}%", ha="center", fontsize=11)
    plt.xlabel("Model")
    plt.ylabel("Accuracy (%)")
    plt.title("Model Comparison on MNIST")
    plt.ylim(90, 100)
    plt.tight_layout()
    plt.savefig("bar_plot.png", dpi=150)
    plt.close()
    print("已生成 bar_plot.png")


def draw_scatter():
    """散点图：准确率 vs 推理时间，看'精度-速度'权衡。"""
    rng = np.random.default_rng(1)
    n = 30
    inference_time = rng.uniform(1, 50, n)         # 推理耗时(ms)
    accuracy = 99.5 - 0.02 * inference_time + rng.normal(0, 0.1, n)
    plt.figure()
    plt.scatter(inference_time, accuracy, c="#d62728", alpha=0.7, s=40)
    plt.xlabel("Inference time (ms)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Inference Time")
    plt.tight_layout()
    plt.savefig("scatter_plot.png", dpi=150)
    plt.close()
    print("已生成 scatter_plot.png")


def draw_subplots():
    """子图：一张图里放三个子图，适合报告排版。"""
    x = np.linspace(0, 2 * np.pi, 100)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(x, np.sin(x), color="#1f77b4")
    axes[0].set_title("sin(x)")
    axes[1].plot(x, np.cos(x), color="#2ca02c")
    axes[1].set_title("cos(x)")
    axes[2].plot(x, np.sin(x) + np.cos(x), color="#ff7f0e")
    axes[2].set_title("sin+cos")
    for ax in axes:
        ax.set_xlabel("x")
    plt.tight_layout()
    plt.savefig("subplots.png", dpi=150)
    plt.close()
    print("已生成 subplots.png")


def main():
    draw_line()
    draw_bar()
    draw_scatter()
    draw_subplots()
    print("\n全部图片生成完毕。用文件管理器打开本目录即可查看。")


if __name__ == "__main__":
    main()
