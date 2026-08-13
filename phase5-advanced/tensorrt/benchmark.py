# -*- coding: utf-8 -*-
"""
benchmark.py —— 对比 PyTorch vs ONNX 的推理速度（精度-速度权衡）

这个文件是干什么的：
    部署的核心问题：模型"变快"的同时，"精度"损失多少？这就是精度-速度权衡，
    对 EE 背景是独特的加分项（和"功耗-性能权衡"是同一套思维）。
    本脚本加载 PyTorch 和 ONNX 两个版本的模型，测同一批图片的推理耗时，
    画一张"精度 vs 速度"的对比图。

跑完能看到什么：
    results/ 目录生成 speed_benchmark.png（PyTorch vs ONNX 推理速度对比柱状图）

怎么跑：
    python3 export.py --format onnx      # 先导出 ONNX
    python3 benchmark.py                 # 再测速度对比
"""

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ultralytics import YOLO


def benchmark_pytorch(model_path, imgs, warmup=3, repeats=10):
    """测 PyTorch 模型推理耗时（ms/张）。"""
    model = YOLO(model_path)
    for _ in range(warmup):          # 预热（第一次推理有 CUDA 初始化开销）
        model.predict(imgs, imgsz=640, verbose=False)
    t0 = time.time()
    for _ in range(repeats):
        model.predict(imgs, imgsz=640, verbose=False)
    return (time.time() - t0) / repeats * 1000  # ms


def benchmark_onnx(onnx_path, imgs, warmup=3, repeats=10):
    """测 ONNX 模型推理耗时（ms/张）。"""
    import cv2
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    # 预处理：resize 到 640x640 + 归一化（速度对比只看推理部分，预处理方式不影响结论）
    def preprocess(img):
        resized = cv2.resize(img, (640, 640))
        return resized.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    blobs = [preprocess(img) for img in imgs]

    # ONNX 默认 batch=1，逐张推理
    def run_all():
        for b in blobs:
            sess.run(None, {input_name: b})

    for _ in range(warmup):
        run_all()
    t0 = time.time()
    for _ in range(repeats):
        run_all()
    return (time.time() - t0) / repeats * 1000


def main():
    # 准备测试图片（从 COCO128 取 8 张）
    import glob
    img_paths = sorted(glob.glob("/mnt/f/yolo/phase3-yolov8/analysis/datasets/coco128/images/train2017/*.jpg"))[:8]
    if not img_paths:
        img_paths = sorted(glob.glob("../phase3-yolov8/analysis/datasets/coco128/images/train2017/*.jpg"))[:8]
    import cv2
    imgs = [cv2.imread(p) for p in img_paths]
    print(f"测试图片: {len(imgs)} 张")

    print("\n测 PyTorch 推理速度...")
    pt_time = benchmark_pytorch("yolov8n.pt", imgs)
    print(f"  PyTorch: {pt_time:.2f} ms/批")

    onnx_path = "yolov8n.onnx"
    onnx_time = None
    if os.path.exists(onnx_path):
        print("测 ONNX 推理速度...")
        onnx_time = benchmark_onnx(onnx_path, imgs)
        print(f"  ONNX: {onnx_time:.2f} ms/批")
    else:
        print("未找到 yolov8n.onnx，请先运行 python3 export.py --format onnx")

    # 画对比图
    os.makedirs("results", exist_ok=True)
    labels = ["PyTorch"]
    times = [pt_time]
    if onnx_time is not None:
        labels.append("ONNX")
        times.append(onnx_time)

    plt.figure(figsize=(6, 4.5))
    colors = ["#1f77b4", "#2ca02c"]
    bars = plt.bar(labels, times, color=colors[:len(labels)], width=0.5)
    for bar, t in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width() / 2, t + max(times) * 0.03,
                 f"{t:.1f}ms", ha="center")
    plt.ylabel("Inference time (ms/batch)")
    plt.title("PyTorch vs ONNX inference speed")
    plt.tight_layout()
    plt.savefig("results/speed_benchmark.png", dpi=150)
    plt.close()
    print("\n已生成 results/speed_benchmark.png")

    if onnx_time is not None:
        speedup = pt_time / onnx_time
        print(f"ONNX 相比 PyTorch 加速: {speedup:.2f}x")


if __name__ == "__main__":
    main()
