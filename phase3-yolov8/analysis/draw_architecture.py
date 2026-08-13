# -*- coding: utf-8 -*-
"""
draw_architecture.py —— 画 YOLOv8 结构示意图

这个文件是干什么的：
    用 matplotlib 画一张 YOLOv8 的"三段式"结构图：Backbone → Neck → Head。
    每个模块用带圆角的方块表示，箭头表示数据流向，关键位置标出特征图尺寸。
    这张图是面试时的"杀手锏"——大多数人只会跑代码，画不出结构图。

跑完能看到什么：
    当前目录生成 yolov8_architecture.png

怎么跑：
    python3 draw_architecture.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# 配色：Backbone 蓝、Neck 橙、Head 绿，三大块一眼分清
C_BACKBONE = "#cfe3ff"
C_NECK = "#ffe4c7"
C_HEAD = "#d5f2d5"
C_BORDER = "#444444"


def box(ax, x, y, w, h, text, facecolor, fontsize=9):
    """画一个带文字的圆角方块。"""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.02",
                                linewidth=1.2, edgecolor=C_BORDER,
                                facecolor=facecolor))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, wrap=True)


def arrow(ax, x1, y1, x2, y2, color="#555555", style="-|>"):
    """画一条箭头。"""
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=12, linewidth=1.3, color=color))


def main():
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # ===== Backbone（左侧竖排）=====
    ax.text(1.6, 9.55, "Backbone", ha="center", fontsize=13, fontweight="bold",
            color="#1f4e8c")
    backbone = [
        ("Input 640×640×3", "white"),
        ("Conv 3→64, s2\n(320×320)", C_BACKBONE),
        ("Conv 64→128, s2\n(160×160)", C_BACKBONE),
        ("C2f ×3", C_BACKBONE),
        ("Conv 128→256, s2\n(80×80)", C_BACKBONE),
        ("C2f ×6", C_BACKBONE),
        ("Conv 256→512, s2\n(40×40)", C_BACKBONE),
        ("C2f ×6", C_BACKBONE),
        ("Conv 512→1024, s2\n(20×20)", C_BACKBONE),
        ("C2f ×3", C_BACKBONE),
        ("SPPF", C_BACKBONE),
    ]
    y = 9.0
    for i, (label, color) in enumerate(backbone):
        h = 0.62
        box(ax, 0.5, y, 2.4, h, label, color)
        if i < len(backbone) - 1:
            arrow(ax, 1.7, y, 1.7, y - 0.18)
        y -= (h + 0.18)

    # ===== Neck（中间）=====
    ax.text(6.0, 9.55, "Neck (PAN-FPN)", ha="center", fontsize=13, fontweight="bold",
            color="#a06000")
    neck_boxes = [
        (4.6, 7.2, "Upsample + Concat\n(C2f 512)"),
        (4.6, 5.4, "Upsample + Concat\n(C2f 256 → P3)"),
        (4.6, 3.6, "Concat + C2f 512\n(P4)"),
        (4.6, 1.8, "Concat + C2f 1024\n(P5)"),
    ]
    for x, yy, label in neck_boxes:
        box(ax, x, yy, 2.8, 0.9, label, C_NECK)

    # Neck 内部上下箭头
    arrow(ax, 6.0, 7.2, 6.0, 6.3)
    arrow(ax, 6.0, 6.3, 6.0, 5.4)
    arrow(ax, 6.0, 4.5, 6.0, 3.6)
    arrow(ax, 6.0, 2.7, 6.0, 1.8)

    # ===== Head（右侧）=====
    ax.text(9.8, 9.55, "Head", ha="center", fontsize=13, fontweight="bold",
            color="#1f6e2f")
    head_boxes = [
        (8.6, 5.4, "Detect P3\n(80×80, small)", "P3"),
        (8.6, 3.6, "Detect P4\n(40×40, medium)", "P4"),
        (8.6, 1.8, "Detect P5\n(20×20, large)", "P5"),
    ]
    for x, yy, label, _tag in head_boxes:
        box(ax, x, yy, 3.0, 0.9, label, C_HEAD)

    # ===== Backbone → Neck / Head 的横向连线（三个尺度特征）=====
    # P3: backbone 第4行(80×80) -> neck P3
    arrow(ax, 2.9, 7.3, 4.6, 7.3, color="#1f4e8c")
    ax.text(3.75, 7.42, "P3 80×80", fontsize=8, color="#1f4e8c", ha="center")
    # P4: backbone 第6行(40×40) -> neck P4
    arrow(ax, 2.9, 4.9, 4.6, 4.9, color="#1f4e8c")
    ax.text(3.75, 5.02, "P4 40×40", fontsize=8, color="#1f4e8c", ha="center")
    # P5: backbone SPPF(20×20) -> neck P5
    arrow(ax, 2.9, 2.5, 4.6, 2.5, color="#1f4e8c")
    ax.text(3.75, 2.62, "P5 20×20", fontsize=8, color="#1f4e8c", ha="center")

    # Neck -> Head 三个检测头
    arrow(ax, 7.4, 5.85, 8.6, 5.85, color="#a06000")
    arrow(ax, 7.4, 4.05, 8.6, 4.05, color="#a06000")
    arrow(ax, 7.4, 2.25, 8.6, 2.25, color="#a06000")

    # 图例说明
    ax.text(1.5, 0.5, "Legend: Blue=Backbone (feature extraction)  "
            "Orange=Neck (multi-scale fusion)  Green=Head (detection)",
            fontsize=10, ha="center")

    plt.tight_layout()
    plt.savefig("yolov8_architecture.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("已生成 yolov8_architecture.png —— 面试时展示这张图，讲清三段式结构。")


if __name__ == "__main__":
    main()
