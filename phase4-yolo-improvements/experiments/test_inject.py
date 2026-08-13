# -*- coding: utf-8 -*-
"""快速验证：注入注意力后模型能正确构建，参数量合理。"""
from attention import C2fSE, C2fCBAM, inject_attention
from ultralytics import YOLO

for name, cls in [("baseline", None), ("se", C2fSE), ("cbam", C2fCBAM)]:
    m = YOLO("yolov8n-baseline.yaml")
    if cls is not None:
        inject_attention(m.model, cls)
    n_params = sum(p.numel() for p in m.model.parameters())
    # 统计 C2f / C2fSE / C2fCBAM 的数量，确认替换成功
    from ultralytics.nn.modules import C2f
    n_c2f = sum(1 for _ in m.model.modules() if isinstance(_, C2f))
    print(f"{name:<10}: 参数量={n_params:,}, C2f类模块数={n_c2f}")
