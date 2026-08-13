# -*- coding: utf-8 -*-
"""
export.py —— 导出 YOLOv8 模型为 ONNX / TensorRT 格式

这个文件是干什么的：
    训练好的 PyTorch 模型要部署到实际设备上，通常要先"导出"成更高效的格式：
      - ONNX：通用的模型交换格式，能被各种推理引擎加载
      - TensorRT：NVIDIA 的推理加速引擎，专门为 NVIDIA GPU 优化，速度最快
    本脚本用 ultralytics 一键导出。ONNX 本机可直接跑；TensorRT 需要先装
    tensorrt 包（安装步骤见 README.md）。

跑完能看到什么：
    当前目录生成 yolov8n.onnx 文件（以及可选 .engine 文件）。

怎么跑：
    python3 export.py --format onnx                    # 导出 ONNX（本机可跑）
    python3 export.py --format engine                  # 导出 TensorRT（需先装 tensorrt）
"""

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="导出 YOLOv8 模型")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="模型权重路径")
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "engine"])
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model = YOLO(args.model)
    path = model.export(format=args.format, imgsz=args.imgsz, simplify=True)
    print(f"\n导出完成: {path}")


if __name__ == "__main__":
    main()
