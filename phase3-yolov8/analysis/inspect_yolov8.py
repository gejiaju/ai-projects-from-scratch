# -*- coding: utf-8 -*-
"""
inspect_yolov8.py —— 打印 ultralytics 里 YOLOv8 的真实结构定义

这个文件是干什么的：
    直接读取本机 ultralytics 包里的 yolov8.yaml 配置，以及 Conv / C2f / SPPF / Detect
    这几个核心模块的源码定义，把 Backbone / Neck / Head 的"真实长相"打印出来。
    这是写结构分析文档和画结构图的事实依据——不凭记忆，直接读源码。

跑完能看到什么：
    终端打印：
      1. yolov8.yaml 的 backbone 和 head 定义（每层名字、参数、输出通道）
      2. 四个核心模块的类定义摘要

怎么跑：
    python3 inspect_yolov8.py
"""

import inspect
import os

import ultralytics
from ultralytics.nn.modules import Conv, C2f, SPPF, Detect

BASE = os.path.dirname(ultralytics.__file__)


def find_yaml():
    """在 ultralytics 包里找到 yolov8.yaml 的路径。"""
    for root, _dirs, files in os.walk(BASE):
        for f in files:
            if f == "yolov8.yaml":
                return os.path.join(root, f)
    return None


def print_yaml(path):
    print("=" * 70)
    print("文件:", path)
    print("=" * 70)
    with open(path, "r", encoding="utf-8") as f:
        print(f.read())


def print_class_source(cls, name):
    print("\n" + "=" * 70)
    print(f"模块源码: {name}")
    print("=" * 70)
    src = inspect.getsource(cls)
    # 只打印前 40 行，够看结构和参数了
    lines = src.splitlines()
    print("\n".join(lines[:40]))
    if len(lines) > 40:
        print(f"... (共 {len(lines)} 行，已截断)")


def main():
    yaml_path = find_yaml()
    if yaml_path is None:
        print("没找到 yolov8.yaml")
        return
    print_yaml(yaml_path)
    print_class_source(Conv, "Conv")
    print_class_source(C2f, "C2f")
    print_class_source(SPPF, "SPPF")


if __name__ == "__main__":
    main()
