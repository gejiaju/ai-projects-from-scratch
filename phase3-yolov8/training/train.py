# -*- coding: utf-8 -*-
"""
train.py —— 训练 YOLOv8（第三阶段核心脚本）

这个文件是干什么的：
    用 ultralytics 框架训练 YOLOv8 目标检测模型。支持：
      - 指定数据集配置文件（data.yaml）
      - 指定模型（yolov8n / yolov8s 等，或预训练权重）
      - 指定训练轮数、图片尺寸、设备
    训练过程自动保存 loss 曲线、mAP 曲线、PR 曲线等可视化结果。

跑完能看到什么：
    1. 终端打印每轮的 loss 和 mAP 指标
    2. runs/detect/train/ 目录下生成：
       - results.csv            每轮所有指标的原始数据（画曲线用）
       - results.png            训练/验证的 loss 和 mAP 曲线总览
       - confusion_matrix.png   混淆矩阵
       - weights/best.pt        验证集上最好的模型
       - weights/last.pt        最后一轮的模型

怎么跑：
    # 用 COCO128 小数据集快速跑通（128 张图，几分钟）
    python3 train.py --data coco128.yaml --model yolov8n.pt --epochs 30

    # 用自建数据集（先把 data.yaml 里的 path 改成你的数据集路径）
    python3 train.py --data ../dataset/data.yaml --model yolov8n.pt --epochs 100
"""

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="训练 YOLOv8 目标检测模型")
    parser.add_argument("--data", type=str, default="coco128.yaml",
                        help="数据集配置文件（coco128.yaml 或自建 data.yaml）")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="模型（yolov8n.pt / yolov8s.pt / yolov8n.yaml 等）")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640, help="训练图片尺寸")
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--device", type=str, default="0", help="设备，0 表示 GPU")
    parser.add_argument("--project", type=str, default="runs")
    parser.add_argument("--name", type=str, default="train")
    args = parser.parse_args()

    # 加载模型：传 .pt 表示在预训练权重基础上微调，传 .yaml 表示从零训练
    model = YOLO(args.model)

    print(f"数据集: {args.data}")
    print(f"模型: {args.model}  轮数: {args.epochs}  尺寸: {args.imgsz}  设备: {args.device}")

    # 训练
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        # 训练中用验证集评估，保存最佳模型
        val=True,
        save=True,
    )

    print("\n训练完成。结果在 runs/detect/train/ 目录下，重点看 results.png 和 weights/best.pt")


if __name__ == "__main__":
    main()
