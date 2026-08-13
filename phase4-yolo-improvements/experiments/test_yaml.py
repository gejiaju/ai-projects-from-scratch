# -*- coding: utf-8 -*-
"""快速验证三个消融 yaml 都能被 ultralytics 正确解析并构建模型。"""
from attention import register
register()

from ultralytics import YOLO

for name in ["yolov8n-baseline.yaml", "yolov8n-se.yaml", "yolov8n-cbam.yaml"]:
    try:
        m = YOLO(name)
        n_params = sum(p.numel() for p in m.model.parameters())
        print(f"{name}: 构建成功, 参数量={n_params:,}")
    except Exception as e:
        print(f"{name}: 构建失败 -> {type(e).__name__}: {e}")
