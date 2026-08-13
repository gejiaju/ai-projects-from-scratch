# -*- coding: utf-8 -*-
"""
prepare_dataset.py —— 检查 / 准备 YOLO 格式数据集

这个文件是干什么的：
    1. 检查一个 YOLO 格式数据集是否完整（图片和标注是否一一对应）
    2. 统计图片数量、每个类别的标注框数量
    3. 可选：用 ultralytics 下载 COCO128 小数据集用于快速验证
    训练前跑一遍，能提前发现"图片没有标注"或"标注类别越界"这类问题。

跑完能看到什么：
    终端打印：图片总数、标注框总数、每个类别的框数量、是否有图片缺失标注。

怎么跑：
    python3 prepare_dataset.py                    # 下载并检查 COCO128
    python3 prepare_dataset.py --root ../datasets/coco128   # 检查你自己的数据集
"""

import argparse
import os
from collections import Counter


def download_coco128():
    """用 ultralytics 的机制下载 COCO128 小数据集。"""
    from ultralytics.utils.downloads import download
    # coco128 是 ultralytics 官方提供的 128 张图小数据集（80 类）
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
    download(url, dir="./downloads", unzip=True)
    print("COCO128 已下载到 ./downloads/coco128")


def check_dataset(root):
    """检查 YOLO 格式数据集：images/ 和 labels/ 是否对应。"""
    images_dir = os.path.join(root, "images", "train2017")
    labels_dir = os.path.join(root, "labels", "train2017")

    # 兼容常见的 train/ 目录结构
    if not os.path.isdir(images_dir):
        alt_img = os.path.join(root, "images", "train")
        if os.path.isdir(alt_img):
            images_dir = alt_img
            labels_dir = os.path.join(root, "labels", "train")

    if not os.path.isdir(images_dir):
        print(f"❌ 找不到图片目录 {images_dir}")
        print("   请确认数据集结构是 path/images/train 和 path/labels/train")
        return

    image_files = [f for f in os.listdir(images_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"图片总数: {len(image_files)}")

    missing_label = 0
    empty_label = 0
    class_counter = Counter()
    total_boxes = 0

    for img in image_files:
        label_path = os.path.join(labels_dir, os.path.splitext(img)[0] + ".txt")
        if not os.path.exists(label_path):
            missing_label += 1
            continue
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.read().strip().splitlines()
        if not lines:
            empty_label += 1
        for line in lines:
            parts = line.split()
            if parts:
                class_counter[int(parts[0])] += 1
                total_boxes += 1

    print(f"标注框总数: {total_boxes}")
    print(f"缺少标注的图片: {missing_label} 张")
    print(f"有图片但无框(空标注): {empty_label} 张")
    print("\n每个类别的框数量:")
    for cls_id, count in sorted(class_counter.items()):
        print(f"  class {cls_id}: {count} 个框")


def main():
    parser = argparse.ArgumentParser(description="检查/准备 YOLO 数据集")
    parser.add_argument("--root", type=str, default=None,
                        help="数据集根目录；不填则下载并检查 COCO128")
    args = parser.parse_args()

    if args.root is None:
        download_coco128()
        root = "./downloads/coco128"
    else:
        root = args.root

    print("=" * 60)
    print("数据集检查:", root)
    print("=" * 60)
    check_dataset(root)


if __name__ == "__main__":
    main()
