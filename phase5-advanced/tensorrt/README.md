# 第五阶段 · TensorRT 模型部署（精度-速度权衡）

> 这是 EE 背景的独特加分项：目标检测模型要"落地"，本质是**精度-速度-功耗的权衡**，
> 和你熟悉的"嵌入式 DSP vs GPU 推理"是同一套思维。

## 为什么要做部署

训练好的 PyTorch 模型直接用于推理，有几个问题：

1. **慢**：PyTorch 是"动态图"框架，推理时有很多额外开销
2. **占显存**：带着训练用的优化器状态、梯度等无关信息
3. **不能上生产**：部署到边缘设备/服务器需要高效的推理引擎

解决方案是"导出 + 加速"：把模型转成 **ONNX**（通用格式）或 **TensorRT**（NVIDIA 专用加速），
推理速度能提升数倍。

## 技术路线

```
PyTorch 模型 (.pt)
    │  export.py
    ▼
ONNX (.onnx)  ── onnxruntime 推理（本机可直接跑）
    │
    ▼  TensorRT 转换
TensorRT Engine (.engine)  ── NVIDIA GPU 上最快（需装 tensorrt 包）
```

## 如何使用

```bash
cd /mnt/f/yolo/phase5-advanced/tensorrt

# 1. 导出 ONNX（本机可跑）
python3 export.py --format onnx

# 2. 测 PyTorch vs ONNX 推理速度对比
python3 benchmark.py
```

## TensorRT 安装（进阶，需要额外配置）

TensorRT 需要手动装，且版本要和 CUDA 匹配。WSL 里安装步骤：

```bash
# 1. 装 tensorrt python 包（对应 CUDA 13）
pip install --break-system-packages tensorrt

# 2. 导出 TensorRT engine
python3 export.py --format engine
```

> 注意：TensorRT engine 是**针对特定 GPU + 特定输入尺寸**编译的，换 GPU 或换输入尺寸
> 要重新导出。这就是"静态优化"和"动态图"的差别——TensorRT 用编译期优化换运行时速度。

## 真实基准结果 + 踩坑

本机实测（8 张 640×640 图，`benchmark.py`）：

| 推理引擎 | 耗时 | 说明 |
|---------|------|------|
| PyTorch (GPU) | 30 ms/批 | 原生 GPU 推理 |
| ONNX (CPU) | 437 ms/批 | onnxruntime CPU 版，没用到 GPU |

**关键踩坑**：pip 默认装的 `onnxruntime` 是 **CPU 版**，没有 `CUDAExecutionProvider`，
所以 ONNX 反而比 GPU 的 PyTorch 慢 14 倍。这个教训说明——**部署加速不是免费的**，
要真正提速，需要：

1. 装 `onnxruntime-gpu`（GPU 版 onnxruntime，需匹配 CUDA 版本）
2. 或直接用 **TensorRT**（NVIDIA 专用，最快，但需编译）

## 精度-速度权衡的关键概念

- **FP32 → FP16 → INT8**：降低数值精度来提速，精度损失可控（YOLO 对量化不敏感）
- **算子融合**：把多个小算子合成一个大算子，减少显存读写
- **动态图 vs 静态图**：PyTorch 动态图灵活但慢，TensorRT 静态图快但不灵活
