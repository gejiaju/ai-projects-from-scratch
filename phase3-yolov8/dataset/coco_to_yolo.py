# -*- coding: utf-8 -*-
"""
coco_to_yolo.py —— COCO 标注格式 转 YOLO 标注格式

这个文件是干什么的：
    很多公开数据集（COCO 等）用 JSON 格式存标注，而 YOLO 训练要用 txt 格式。
    本脚本把 COCO 的 JSON 转成 YOLO 的 txt：每张图一个 .txt，每行一个框，
    格式为 `class_id cx cy w h`（中心点坐标和宽高，都归一化到 0~1）。

跑完能看到什么：
    在 --output 目录下为每张图生成对应的 .txt 标注文件。

怎么跑：
    python3 coco_to_yolo.py --json annotations.json --output labels
"""

import argparse
import json
import os


def convert(coco_json, output_dir):
    """把 COCO 格式的标注 JSON 转成 YOLO 格式的 txt 文件。"""
    with open(coco_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # 图片 id -> (宽, 高, 文件名)
    images = {img["id"]: (img["width"], img["height"], img["file_name"])
              for img in data["images"]}

    # 按图片 id 收集标注
    from collections import defaultdict
    anns_by_image = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    count = 0
    for image_id, (width, height, file_name) in images.items():
        lines = []
        for ann in anns_by_image[image_id]:
            # COCO 用 [x_min, y_min, w, h]（左上角 + 宽高），是像素坐标
            x_min, y_min, w, h = ann["bbox"]
            # 转成中心点坐标
            cx = x_min + w / 2
            cy = y_min + h / 2
            # 归一化到 0~1（YOLO 要求）
            cx_n = cx / width
            cy_n = cy / height
            w_n = w / width
            h_n = h / height
            class_id = ann["category_id"]
            lines.append(f"{class_id} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")

        out_name = os.path.splitext(file_name)[0] + ".txt"
        with open(os.path.join(output_dir, out_name), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        count += 1

    print(f"已转换 {count} 张图片的标注，输出到 {output_dir}/")
    print("注意：COCO 的 category_id 和你的 names 列表顺序要对应好。")


def main():
    parser = argparse.ArgumentParser(description="COCO 标注转 YOLO 标注")
    parser.add_argument("--json", type=str, required=True, help="COCO 标注 JSON 文件")
    parser.add_argument("--output", type=str, default="labels", help="输出目录")
    args = parser.parse_args()
    convert(args.json, args.output)


if __name__ == "__main__":
    main()
