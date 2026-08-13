# -*- coding: utf-8 -*-
"""诊断：分别用 yolov8n.pt 和 yolo26n.pt 在 coco128 上验证，看预训练权重本身有多强。"""
from ultralytics import YOLO

for name in ["yolo26n.pt", "yolov8n.pt"]:
    print("=" * 60)
    print("验证权重:", name)
    try:
        m = YOLO(name)
        # 打印模型类别数和参数量
        nc = getattr(m.model, "nc", "?")
        n_params = sum(p.numel() for p in m.model.parameters())
        print(f"  类别数 nc={nc}, 参数量={n_params:,}")
        r = m.val(data="coco128.yaml", verbose=False)
        print(f"  mAP50 = {r.box.map50:.4f}, mAP50-95 = {r.box.map:.4f}")
    except Exception as e:
        print(f"  !! 加载失败: {type(e).__name__}: {e}")
