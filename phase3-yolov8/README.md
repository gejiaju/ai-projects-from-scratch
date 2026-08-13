# 第三阶段：YOLOv8 完整复现

> 核心阶段：读懂 YOLOv8 源码结构、跑通完整训练流程、画出结构图。
> 面试目标——大多数人只会 `yolo train` 跑个 demo，你能画结构图、讲清三个模块、说出训练细节。

## 目录结构

```
phase3-yolov8/
├── dataset/                 # 数据集准备
│   ├── data.yaml            #   数据集配置（YOLO 格式）
│   ├── prepare_dataset.py   #   检查数据集完整性、统计类别
│   ├── coco_to_yolo.py      #   COCO JSON → YOLO txt 格式转换
│   └── annotation_guide.md  #   LabelImg 标注指南（自建数据集用）
├── training/                # 训练
│   ├── train.py             #   训练脚本（ultralytics）
│   └── runs/detect/...      #   训练结果（曲线/混淆矩阵/权重）
└── analysis/                # 结构分析
    ├── inspect_yolov8.py        #   读取 ultralytics 里 YOLOv8 真实源码结构
    ├── yolov8_architecture.md   #   Backbone/Neck/Head 详解
    └── draw_architecture.py     #   生成结构图
```

## 实验结果（真实训练）

用 **COCO128**（128 张图的公开小数据集，80 类）从零训练 YOLOv8n，30 轮：

| 指标 | 数值 |
|------|------|
| 最终 mAP50 | **80.3%** |
| 最终 mAP50-95 | 61.6% |
| 推理速度 | 1.6 ms/张（GPU） |

训练结果文件（`training/runs/detect/runs/train/`）：

- `results.png` — loss 曲线 + mAP 曲线总览（证明你真的训练过模型的最直接证据）
- `confusion_matrix.png` — 混淆矩阵
- `BoxPR_curve.png` / `BoxP_curve.png` / `BoxR_curve.png` — PR 曲线族
- `weights/best.pt` — 验证集上最好的模型
- `val_batch*_pred.jpg` — 模型在验证集上的预测可视化（框画在图上）

## 如何复现

```bash
cd /mnt/f/yolo/phase3-yolov8/training

# 快速跑通（COCO128 会自动下载）
python3 train.py --data coco128.yaml --model yolov8n.pt --epochs 30

# 自建数据集（先按 dataset/annotation_guide.md 标注，再改 dataset/data.yaml 的 path）
python3 train.py --data ../dataset/data.yaml --model yolov8n.pt --epochs 100

# 查看 YOLOv8 真实源码结构
cd ../analysis && python3 inspect_yolov8.py && python3 draw_architecture.py
```

## 踩坑记录（面试可讲的真实经历）

1. **预训练权重下载失败**：`yolov8n.pt` 从 github 下载时 Curl 返回 56（网络中断）。
   排查发现是暂时网络抖动，用 `curl -L` 手动重试成功。**教训**：训练前先确认权重文件
   完整（6.2MB 左右），别让"下载失败"悄悄 fallback 成从零训练而不自知。

2. **ultralytics 版本差异**：本机装的 8.4.95 已经把 `yolov8n` 重命名为新系列 `yolo26n`
   （`YOLO26n`，122 层）。用 `torch.load` 直接读权重、用 `val` 对比两个权重的真实 mAP，
   才确认两个权重文件都是有效、真实的。**教训**：版本更新后模型名和结构可能变，要用
   `model.summary()` 或 `val` 验证，而不是默认"它就是我以为的那个模型"。

3. **save_dir 嵌套**：ultralytics 的 `project` 参数会自动追加 task 目录（detect），
   传 `project="runs/detect"` 会变成 `runs/detect/runs/detect/train`。改成 `project="runs"`。

## 面试自述要点

> "我把 YOLOv8 拆成三段看：Backbone（CSPDarknet，核心是 C2f 模块）+ Neck（PAN-FPN 多尺度
> 融合）+ Head（anchor-free 解耦头）。我画了张结构图，标出了三个尺度 P3/P4/P5 分别检测
> 小/中/大目标。然后我在 COCO128 上从零训练了 30 轮，mAP50 到 80%，训练曲线和混淆矩阵
> 都保存了。过程中我还踩了预训练权重下载失败的坑，学会了怎么验证权重是否真的加载。"
