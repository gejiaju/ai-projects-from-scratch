# -*- coding: utf-8 -*-
"""
analyze_csv.py —— 读 CSV → 算统计量 → 画柱状图（第一阶段核心验证脚本）

这个文件是干什么的：
    这是第一阶段的"过关测试"：读入一个 CSV 表格，按某个分组列（默认 model）
    对某个数值列（默认 accuracy）计算均值、标准差、最大值、最小值，
    然后画一张带误差棒的柱状图。这张图可以直接放进技术报告。

跑完能看到什么：
    1. 终端打印一张统计表（每个模型的 均值/标准差/最大/最小/样本数）
    2. 当前目录生成 accuracy_bar.png 柱状图

怎么跑：
    python3 generate_data.py                       # 先造数据
    python3 analyze_csv.py                          # 用默认参数分析
    python3 analyze_csv.py --column loss            # 改成统计 loss 列
    python3 analyze_csv.py --output loss_bar.png    # 指定图片输出名
"""

import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"figure.figsize": (8, 5), "font.size": 12})


def load_csv(path):
    """读取 CSV 文件，返回 (表头, 数据行列表)。"""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    return fieldnames, rows


def compute_stats(rows, group_col, value_col):
    """按 group_col 分组，对 value_col 计算统计量。

    返回字典：{分组名: {"mean": ..., "std": ..., "min": ..., "max": ..., "count": ...}}
    """
    groups = defaultdict(list)
    for row in rows:
        groups[row[group_col]].append(float(row[value_col]))

    stats = {}
    for name, values in groups.items():
        arr = np.array(values)
        stats[name] = {
            "mean": arr.mean(),
            "std": arr.std(),
            "min": arr.min(),
            "max": arr.max(),
            "count": len(arr),
        }
    return stats


def print_stats(stats, value_col):
    """把统计表打印到终端。"""
    print(f"\n按模型统计 {value_col} 列：")
    print("-" * 58)
    print(f"{'model':<12}{'mean':>10}{'std':>10}{'min':>10}{'max':>10}{'n':>6}")
    print("-" * 58)
    for name in sorted(stats):
        s = stats[name]
        print(f"{name:<12}{s['mean']:>10.2f}{s['std']:>10.2f}"
              f"{s['min']:>10.2f}{s['max']:>10.2f}{s['count']:>6d}")
    print("-" * 58)


def plot_bar(stats, value_col, output):
    """画带误差棒的柱状图，并保存为 PNG。"""
    names = sorted(stats)
    means = [stats[n]["mean"] for n in names]
    stds = [stats[n]["std"] for n in names]

    plt.figure()
    bars = plt.bar(names, means, yerr=stds, capsize=5,
                   color="#2ca02c", width=0.6, alpha=0.85)
    for bar, mean in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width() / 2, mean + max(stds) * 0.6,
                 f"{mean:.2f}", ha="center", fontsize=10)
    plt.xlabel("Model")
    plt.ylabel(value_col)
    plt.title(f"Mean {value_col} by model (error bar = std)")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"已生成柱状图 {output}")


def main():
    parser = argparse.ArgumentParser(description="读 CSV、算统计、画柱状图")
    parser.add_argument("--input", type=str, default="experiment_results.csv",
                        help="输入 CSV 文件路径")
    parser.add_argument("--group", type=str, default="model",
                        help="分组列名（默认 model）")
    parser.add_argument("--column", type=str, default="accuracy",
                        help="要统计的数值列名（默认 accuracy）")
    parser.add_argument("--output", type=str, default="accuracy_bar.png",
                        help="柱状图输出文件名")
    args = parser.parse_args()

    fieldnames, rows = load_csv(args.input)
    print(f"已读取 {args.input}，共 {len(rows)} 行，列: {', '.join(fieldnames)}")

    stats = compute_stats(rows, args.group, args.column)
    print_stats(stats, args.column)
    plot_bar(stats, args.column, args.output)

    print("\n完成。打开", args.output, "查看柱状图。")


if __name__ == "__main__":
    main()
