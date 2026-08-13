# -*- coding: utf-8 -*-
"""
run_ablation.py —— 消融实验脚本（第四阶段核心）

这个文件是干什么的：
    跑一组"控制变量"的消融实验，验证"加注意力"到底有没有用、哪种更好：
      1. baseline：YOLOv8n 原版（不带注意力）
      2. +SE     ：YOLOv8n + SE 通道注意力
      3. +CBAM   ：YOLOv8n + CBAM 通道+空间注意力
    三组实验用**完全相同**的数据、轮数、学习率、随机种子，只改"注意力模块"这一个变量。
    做法：先用 yolov8n-baseline.yaml 构建 baseline，再就地注入注意力（替换 C2f 为
    C2fSE/C2fCBAM），保证除注意力外结构完全一致。

跑完能看到什么：
    每组训练完成后自动对比 mAP，打印一张对比表，并生成 results/ablation.json。
    结果示例（真实数据见 technical_report.md）：
      baseline  mAP50 = xx.x%
      +SE       mAP50 = xx.x%   (+x.x)
      +CBAM     mAP50 = xx.x%   (+x.x)

怎么跑：
    python3 run_ablation.py --epochs 30          # 全部跑一遍
    python3 run_ablation.py --exp se             # 只跑 SE 那一组
"""

import argparse
import json
import os

import torch

from attention import C2fSE, C2fCBAM, inject_attention
from ultralytics import YOLO


# 三组实验的定义
EXPERIMENTS = {
    "baseline": {"label": "YOLOv8n (baseline)", "attention": None},
    "se":       {"label": "YOLOv8n + SE",      "attention": C2fSE},
    "cbam":     {"label": "YOLOv8n + CBAM",    "attention": C2fCBAM},
}


def build_model(attention_cls):
    """构建模型：从预训练权重 yolov8n.pt 出发，或注入注意力的版本。

    三组都用同一个预训练基础，保证消融实验公平：
      - baseline 直接微调
      - se/cbam 注入注意力（保留 C2f 预训练权重，注意力层随机初始化）后微调
    """
    model = YOLO("yolov8n.pt")
    if attention_cls is not None:
        inject_attention(model.model, attention_cls)
    return model


def run_one(exp_name, epochs, seed):
    """训练一组实验，返回最佳 mAP50。"""
    cfg = EXPERIMENTS[exp_name]
    model = build_model(cfg["attention"])
    results = model.train(
        data="coco128.yaml",
        epochs=epochs,
        imgsz=640,
        batch=16,
        device="0",
        seed=seed,            # 固定种子，三组实验除注意力外完全一致
        # 微调用默认 optimizer=auto（会自动算出适合微调的小 lr，第三阶段验证过能收敛）
        project="runs/ablation",
        name=exp_name,
        verbose=False,
        plots=True,
    )
    # 返回 best 模型的 mAP50
    map50 = getattr(getattr(results, "box", None), "map50", None)
    return float(map50) if map50 is not None else None


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 注意力消融实验")
    parser.add_argument("--exp", type=str, default="all", choices=list(EXPERIMENTS) + ["all"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    exps = list(EXPERIMENTS) if args.exp == "all" else [args.exp]

    print(f"设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"实验组: {', '.join(exps)}，每组 {args.epochs} 轮，种子 {args.seed}\n")

    results = {}
    for exp in exps:
        print(f"\n===== 训练 {EXPERIMENTS[exp]['label']} =====")
        map50 = run_one(exp, args.epochs, args.seed)
        results[exp] = map50
        if map50 is not None:
            print(f"  {EXPERIMENTS[exp]['label']} 最佳 mAP50 = {map50*100:.2f}%")
        else:
            print(f"  {exp} 训练完成，mAP 请查 runs/ablation/{exp}/results.csv")

    # 打印汇总对比表
    if args.exp == "all" and all(v is not None for v in results.values()):
        base = results["baseline"]
        print("\n" + "=" * 55)
        print("消融实验结果汇总")
        print("=" * 55)
        print(f"{'实验组':<24}{'mAP50':>10}{'提升':>12}")
        print("-" * 55)
        for exp in ["baseline", "se", "cbam"]:
            v = results[exp]
            delta = v - base
            print(f"{EXPERIMENTS[exp]['label']:<24}{v*100:>9.2f}%{delta*100:>+11.2f}%")
        print("=" * 55)

        os.makedirs("results", exist_ok=True)
        with open("results/ablation.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("结果已存到 results/ablation.json")


if __name__ == "__main__":
    main()
