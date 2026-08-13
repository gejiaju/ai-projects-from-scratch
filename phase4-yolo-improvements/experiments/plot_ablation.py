# -*- coding: utf-8 -*-
"""
plot_ablation.py —— 画消融实验对比柱状图

这个文件是干什么的：
    把三组消融实验的 mAP50 画成柱状图。因为三组结果非常接近（80.28 vs 80.24），
    所以把 y 轴范围设窄（79~81），放大差异，让"加注意力没提升"这个结论一目了然。

跑完能看到什么：
    results/ablation_bar.png 柱状图

怎么跑：
    python3 plot_ablation.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 真实实验结果（由 run_ablation.py 训练得到）
MODELS = ["baseline", "+SE", "+CBAM"]
MAP50 = [80.28, 80.24, 80.24]
COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e"]


def main():
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(6.5, 4.5))
    bars = plt.bar(MODELS, MAP50, color=COLORS, width=0.5)
    for bar, v in zip(bars, MAP50):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                 f"{v:.2f}%", ha="center", fontsize=11)
    # 窄 y 轴，放大差异
    plt.ylim(79.0, 81.0)
    plt.ylabel("mAP50 (%)")
    plt.title("Ablation: SE / CBAM attention on YOLOv8n (COCO128)")
    plt.tight_layout()
    plt.savefig("results/ablation_bar.png", dpi=150)
    plt.close()
    print("已生成 results/ablation_bar.png")


if __name__ == "__main__":
    main()
