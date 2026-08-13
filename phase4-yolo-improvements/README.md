# 第四阶段：YOLOv8 改进 + 消融实验

> 在 YOLOv8 上做两个方向的改进（注意力机制），用**严格的控制变量消融实验**回答
> "加注意力到底有没有用"——用严格的控制变量消融实验给出答案。

## 项目背景

YOLOv8 的 C2f 模块提取特征时，对所有通道和空间位置一视同仁，没有"重要性"概念。
本阶段引入两种经典注意力机制（SE、CBAM），验证它们能否提升检测精度，并严谨对比。

## 方法说明

- **SE**（Squeeze-and-Excitation）：通道注意力，学"哪些通道重要"
- **CBAM**（Convolutional Block Attention Module）：通道注意力 + 空间注意力

实现方式：**注入式替换**——先构建 baseline，再递归把模型里的 C2f 换成带注意力的
C2fSE/C2fCBAM。这样三组实验除了注意力模块，其他结构完全一致。

## 实验结果

（由 `experiments/run_ablation.py` 真实训练得到，COCO128 微调 30 轮）

| 实验组 | 参数量 | mAP50 | 提升 |
|--------|--------|-------|------|
| baseline | 3,157,200 | 80.28% | — |
| +SE | 3,180,880 | 80.24% | -0.04% |
| +CBAM | 3,181,664 | 80.24% | -0.04% |

**结论：COCO128 上注意力没有提升。** 原因是数据量太小（128 张图）、baseline 已经
过拟合，注意力模块的优势在大数据集上才能体现。这个"负面结果"本身是诚实的科学结论，也说明"加模块不等于一定有效"。

## 目录结构

```
phase4-yolo-improvements/
├── experiments/
│   ├── attention.py           # SE/CBAM 模块 + 注入式替换
│   ├── run_ablation.py        # 消融实验脚本
│   ├── yolov8n-baseline.yaml  # baseline 模型配置
│   ├── test_inject.py         # 验证注入正确
│   └── show_parse.py / find_register.py  # 排查记录
└── report/
    └── technical_report.md    # 技术报告
```

## 如何使用

```bash
cd /mnt/f/yolo/phase4-yolo-improvements/experiments

# 验证注意力注入正确（应看到三个模型的参数量）
python3 test_inject.py

# 跑完整消融实验（三组，每组 30 轮）
python3 run_ablation.py --epochs 30

# 只跑一组（快速验证）
python3 run_ablation.py --exp se --epochs 10
```

## 踩坑记录

1. **自定义模块 yaml 注册失败**：直接在 yaml 里写 `C2fSE` 会报 `KeyError`。排查发现
   ultralytics 的 parse_model 用 `globals()[m]` 从 `nn.tasks` 命名空间解析模块名，
   不是 `nn.modules`（见 `find_register.py`）。

2. **参数错位报 TypeError**：即使注册成功，`C2fSE` 仍报 `empty() size must be tuple of
   ints`。原因是 parse_model 对 C2f 类模块会把 repeats 插进参数列表（`repeat_modules`
   集合），自定义模块不在其中，导致 `n` 和 `shortcut` 错位（见 `show_parse.py`）。
   最终改为**代码内注入式替换**绕开 parse_model，问题解决。
