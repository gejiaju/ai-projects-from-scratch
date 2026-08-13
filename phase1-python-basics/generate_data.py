# -*- coding: utf-8 -*-
"""
generate_data.py —— 生成一组"模型对比实验"的模拟数据 (CSV)

这个文件是干什么的：
    为 analyze_csv.py 提供输入数据。它模拟 5 个模型各自跑了 5 次实验，
    记录准确率、损失、参数量、推理时间，写成一个 CSV 表格。
    用这个数据练"读 CSV → 算统计 → 画柱状图"的完整流程。

跑完能看到什么：
    当前目录生成 experiment_results.csv，内容形如：
        model,accuracy,loss,params_mb,time_ms
        MLP,97.1,0.18,0.4,1.2
        ...

怎么跑：
    python3 generate_data.py                          # 用默认参数
    python3 generate_data.py --output my_data.csv      # 指定输出文件名
    python3 generate_data.py --runs 8 --seed 3         # 每个模型跑 8 次，换种子
"""

import argparse
import csv

import numpy as np


# 每个模型的基础指标（均值），下面会在此基础上加一点随机波动
MODEL_BASE = {
    "MLP":     {"accuracy": 97.0, "loss": 0.18, "params_mb": 0.4, "time_ms": 1.5},
    "CNN":     {"accuracy": 99.1, "loss": 0.08, "params_mb": 1.2, "time_ms": 3.2},
    "ResNet":  {"accuracy": 99.4, "loss": 0.05, "params_mb": 11.7, "time_ms": 8.6},
    "MobileNet": {"accuracy": 98.8, "loss": 0.10, "params_mb": 2.2, "time_ms": 2.1},
    "ViT":     {"accuracy": 99.0, "loss": 0.09, "params_mb": 86.0, "time_ms": 15.3},
}


def main():
    parser = argparse.ArgumentParser(description="生成模型对比实验模拟数据")
    parser.add_argument("--output", type=str, default="experiment_results.csv",
                        help="输出 CSV 文件路径")
    parser.add_argument("--runs", type=int, default=5,
                        help="每个模型重复实验次数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子，保证可复现")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "accuracy", "loss", "params_mb", "time_ms"])

        for model, base in MODEL_BASE.items():
            for _ in range(args.runs):
                # 在基础值上叠加随机波动，模拟多次实验的差异
                accuracy = base["accuracy"] + rng.normal(0, 0.15)
                loss = base["loss"] + abs(rng.normal(0, 0.01))
                params_mb = base["params_mb"] + abs(rng.normal(0, 0.05))
                time_ms = base["time_ms"] + abs(rng.normal(0, 0.2))
                writer.writerow([
                    model,
                    round(accuracy, 2),
                    round(loss, 4),
                    round(params_mb, 3),
                    round(time_ms, 2),
                ])

    print(f"已生成 {args.output}，共 {len(MODEL_BASE)} 个模型 × {args.runs} 次 = "
          f"{len(MODEL_BASE) * args.runs} 行数据")
    print("下一步运行: python3 analyze_csv.py --input " + args.output)


if __name__ == "__main__":
    main()
