# 第五阶段：DDPM + TensorRT + LoRA（三方向进阶）

> 前四个阶段打好了深度学习基础，这一阶段向三个热门方向延伸：**生成模型（DDPM）、
> 模型部署（TensorRT）、大模型微调（LoRA）**。每个方向有独立的可视化产物和代码仓库。

## 三个方向

| 方向 | 核心产出 | 面试亮点 |
|------|---------|---------|
| **DDPM** 扩散模型 | 从零手写 U-Net + 训练 + "噪声→图像"演变图 | 生成式 AI 底层原理 |
| **TensorRT** 部署 | ONNX 导出 + 精度-速度对比 + 部署文档 | EE 背景独特加分项 |
| **LoRA** 微调 | 大模型微调代码 + 考研面试主题数据 | 大模型应用核心技术 |

## 目录结构

```
phase5-advanced/
├── ddpm/          # 扩散模型从零实现（真实训练 + 可视化）
├── tensorrt/      # 模型部署（ONNX 导出 + 推理基准）
└── lora/          # LoRA 大模型微调
```

## 各方向怎么跑

- **DDPM**（本机已真实跑通）：`cd ddpm && python3 train.py && python3 sample.py`
  → 重点看 `diffusion_process.png`（噪声→图像演变）
- **TensorRT**（ONNX 已真实跑通）：`cd tensorrt && python3 export.py && python3 benchmark.py`
- **LoRA**（代码就绪，需装依赖 + 下载模型）：`cd lora && pip install transformers peft accelerate && python3 lora_finetune.py`

## 面试自述要点

"我在前四阶段掌握了分类和检测之后，向三个热门方向做了延伸：① 从零手写 DDPM 扩散模型，
在 MNIST 上训练，展示了噪声如何一步步变成数字——这是 Stable Diffusion 的底层原理；
② 做了模型部署的对比，理解了推理引擎要选对后端（踩了 onnxruntime CPU 版的坑）；
③ 用 LoRA 微调大模型，理解了低秩适配为什么能把微调成本降到千分之一。"

详见各子目录的 README 和 `../phase6-materials/interview-qa.md`。
