# -*- coding: utf-8 -*-
"""
compare.py —— 画 MLP vs CNN 的对比图

这个文件是干什么的：
    读 train.py 保存的两个历史记录 JSON，把 MLP 和 CNN 的
    训练损失曲线、准确率曲线画在同一张图上对比，再画一张最终准确率柱状图。
    这些图就是"CNN 比 MLP 好在哪"的最直观证据，可以直接放进报告。

跑完能看到什么：
    results/ 目录下生成：
      - loss_curve.png   两个模型的训练损失下降曲线
      - acc_curve.png    两个模型的测试准确率上升曲线
      - acc_bar.png      最终准确率柱状图

怎么跑：
    python3 compare.py
    （需要先分别跑过 train.py --model mlp 和 train.py --model cnn）
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.figsize": (7, 4.5), "font.size": 12, "axes.grid": True,
                     "grid.alpha": 0.3})

COLORS = {"mlp": "#1f77b4", "cnn": "#2ca02c"}
LABELS = {"mlp": "MLP", "cnn": "CNN"}


def load_history(model_name):
    path = f"results/{model_name}_history.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 {path}，请先运行 python3 train.py --model {model_name}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    mlp = load_history("mlp")
    cnn = load_history("cnn")

    # 图1：训练损失曲线
    plt.figure()
    plt.plot(mlp["epoch"], mlp["train_loss"], marker="o", color=COLORS["mlp"], label="MLP")
    plt.plot(cnn["epoch"], cnn["train_loss"], marker="s", color=COLORS["cnn"], label="CNN")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss: MLP vs CNN")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/loss_curve.png", dpi=150)
    plt.close()

    # 图2：测试准确率曲线
    plt.figure()
    plt.plot(mlp["epoch"], [a * 100 for a in mlp["test_acc"]], marker="o",
             color=COLORS["mlp"], label="MLP")
    plt.plot(cnn["epoch"], [a * 100 for a in cnn["test_acc"]], marker="s",
             color=COLORS["cnn"], label="CNN")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy: MLP vs CNN")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/acc_curve.png", dpi=150)
    plt.close()

    # 图3：最终准确率柱状图
    final = {"MLP": mlp["test_acc"][-1] * 100, "CNN": cnn["test_acc"][-1] * 100}
    plt.figure()
    bars = plt.bar(final.keys(), final.values(), color=[COLORS["mlp"], COLORS["cnn"]],
                   width=0.5)
    for bar, name in zip(bars, final):
        plt.text(bar.get_x() + bar.get_width() / 2, final[name] + 0.2,
                 f"{final[name]:.2f}%", ha="center", fontsize=12)
    plt.ylabel("Test Accuracy (%)")
    plt.title("Final Accuracy: MLP vs CNN")
    plt.ylim(90, 100)
    plt.tight_layout()
    plt.savefig("results/acc_bar.png", dpi=150)
    plt.close()

    print("对比图已生成：")
    print(f"  MLP 最终准确率: {final['MLP']:.2f}%")
    print(f"  CNN 最终准确率: {final['CNN']:.2f}%")
    print("  results/loss_curve.png  results/acc_curve.png  results/acc_bar.png")


if __name__ == "__main__":
    main()
