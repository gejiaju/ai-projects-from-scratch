# -*- coding: utf-8 -*-
"""诊断脚本：检查 yolov8n.pt 是否被正确加载为预训练权重。"""
import torch

path = "yolov8n.pt"
ckpt = torch.load(path, map_location="cpu", weights_only=False)
print("checkpoint 类型:", type(ckpt))
if isinstance(ckpt, dict):
    print("顶层 keys:", list(ckpt.keys()))
    # 预训练权重通常有 model / ema 键，且里面是 state_dict
    model_state = ckpt.get("model") or ckpt.get("ema")
    if model_state is None:
        print("!! 没有 model/ema 键，可能不是标准 checkpoint")
    else:
        print("model state_dict 参数量:", len(model_state))
        print("前 3 个参数名:")
        for i, k in enumerate(list(model_state.keys())[:3]):
            print("  ", k, tuple(model_state[k].shape))
else:
    print("!! checkpoint 不是 dict，内容:", str(ckpt)[:200])
