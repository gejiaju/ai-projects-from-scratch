# -*- coding: utf-8 -*-
"""
sample.py —— 用训练好的 DDPM 生成手写数字 + 画"噪声到图像"演变图

这个文件是干什么的：
    加载 train.py 训练好的模型，从纯随机噪声出发，一步步去噪，生成 16 张手写数字。
    同时记录去噪过程的中间步骤，画一张"噪声 -> 图像"的演变过程图——这是扩散模型
    最直观、最有展示力的可视化产物（面试时直接亮出来）。

跑完能看到什么：
    当前目录生成：
      - samples_grid.png        16 张生成的数字
      - diffusion_process.png   一个样本从纯噪声逐步变成数字的演变过程
      - loss_curve.png          训练损失曲线（如果 train_loss.json 存在）

怎么跑：
    python3 sample.py
    （需要先跑过 train.py）
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from model import UNet
from diffusion import DDPM


def denorm(x):
    """把 [-1, 1] 的像素反归一化到 [0, 1]，用于显示。"""
    return (x + 1) / 2


def draw_grid(samples, path, nrow=4):
    """把多张图拼成网格保存。"""
    grid = make_grid(samples, nrow=nrow, normalize=False)
    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    plt.axis("off")
    plt.title("Generated MNIST digits (DDPM)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"已生成 {path}")


def draw_process(trajectory, path):
    """画去噪演变过程：选一个样本，展示它从噪声到数字的逐步变化。"""
    # 取第一个样本，沿时间排列
    imgs = [t[0] for t in trajectory]   # 每个是 (1, 28, 28)
    n = len(imgs)
    plt.figure(figsize=(n * 1.6, 2.4))
    for i, img in enumerate(imgs):
        plt.subplot(1, n, i + 1)
        plt.imshow(denorm(img).cpu().squeeze(0).numpy(), cmap="gray")
        plt.axis("off")
        if i == 0:
            plt.title("noise")
        elif i == n - 1:
            plt.title("image")
    plt.suptitle("Diffusion denoising process (noise -> image)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"已生成 {path}")


def draw_loss(path):
    """画训练损失曲线（如果数据存在）。"""
    if not os.path.exists("train_loss.json"):
        return
    with open("train_loss.json", "r", encoding="utf-8") as f:
        hist = json.load(f)
    plt.figure(figsize=(6, 4))
    plt.plot(hist["epoch"], hist["loss"], marker="o", color="#1f77b4")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (noise prediction MSE)")
    plt.title("DDPM Training Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"已生成 {path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet(in_ch=1, base=64, time_dim=64).to(device)
    model.load_state_dict(torch.load("ddpm_mnist.pt", map_location=device))
    model.eval()
    print("已加载 ddpm_mnist.pt")

    ddpm = DDPM(model, T=300, device=device)

    # 采样 16 张 + 记录去噪过程
    samples, trajectory = ddpm.sample((16, 1, 28, 28), return_trajectory=True, save_every=25)
    samples = denorm(samples).clamp(0, 1)

    draw_grid(samples, "samples_grid.png", nrow=4)
    draw_process(trajectory, "diffusion_process.png")
    draw_loss("loss_curve.png")
    print("\n完成。重点看 diffusion_process.png —— 展示噪声如何一步步变成数字。")


if __name__ == "__main__":
    main()
