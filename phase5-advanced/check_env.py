# -*- coding: utf-8 -*-
"""检查第五阶段 TensorRT / LoRA 需要的库是否已安装。"""
mods = ["tensorrt", "onnx", "onnxruntime", "transformers", "peft", "accelerate", "torch"]
for m in mods:
    try:
        __import__(m)
        print(f"{m:<15} 已安装")
    except ImportError:
        print(f"{m:<15} 未安装")
