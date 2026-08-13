# 从零构建 AI 项目集

> 清深 AI 备考计划（清华大学深圳国际研究生院 · 电子信息专业 AI 方向复试备考）的工程实现。
> 从零构建递进式的 AI 项目，覆盖深度学习基础 → 目标检测 → 模型改进 → 生成模型 → 部署与大模型微调。

## 项目总览

| 阶段 | 主题 | 核心成果 |
|------|------|---------|
| [phase1-python-basics](phase1-python-basics/) | Python 基础 | NumPy / Matplotlib / 文件读写 / argparse |
| [phase2-pytorch-mnist](phase2-pytorch-mnist/) | PyTorch 基本功 | 手写 MLP vs CNN，CNN 99.15% vs MLP 98.02% |
| [phase3-yolov8](phase3-yolov8/) | YOLOv8 复现 | 结构图 + 真实训练 mAP50 80.3% |
| [phase4-yolo-improvements](phase4-yolo-improvements/) | YOLO 改进 | SE/CBAM 注意力消融实验 |
| [phase5-advanced](phase5-advanced/) | 三方向进阶 | DDPM 扩散模型 + ONNX 部署 + LoRA 微调 |
| [phase6-materials](phase6-materials/) | 面试材料 | 问答笔记 + 自述 + 技术报告 + PPT |

## 技术栈

- **硬件**：NVIDIA RTX 5060 (8GB) + WSL2 Ubuntu
- **框架**：PyTorch 2.13 + ultralytics 8.4 + transformers
- **语言**：Python 3.14

## 快速导航

- 想了解「为什么 CNN 比 MLP 强」→ `phase2-pytorch-mnist/README.md`
- 想看「YOLOv8 结构图」→ `phase3-yolov8/analysis/yolov8_architecture.png`
- 想看「注意力消融结论」→ `phase4-yolo-improvements/report/technical_report.md`
- 想看「噪声如何变成图像」→ `phase5-advanced/ddpm/diffusion_process.png`
- 想准备面试 → `phase6-materials/interview-qa.md`

## 环境复现

详见各阶段 README。核心依赖安装（WSL Ubuntu + CUDA）：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install numpy pandas matplotlib ultralytics opencv-python pillow
```

> 注：数据集（MNIST、COCO128）和模型权重（.pt/.onnx）已通过 .gitignore 排除，
> 各阶段脚本首次运行时会自动下载，或按 README 步骤重新训练得到。
