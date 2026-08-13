# -*- coding: utf-8 -*-
"""
train.py —— 训练 DDPM（在 MNIST 上学习生成手写数字）

这个文件是干什么的：
    训练扩散模型，让它学会"从纯噪声逐步还原出手写数字"。
    训练目标很简单：随机挑一张图、随机挑一个时间步加噪，让网络预测加进去的噪声，
    预测得越准，模型就越会去噪，最终就能从噪声生成图片。

跑完能看到什么：
    1. 终端打印每轮的损失（应该逐步下降）
    2. 保存 ddpm_mnist.pt 模型权重
    3. 保存 train_loss.json 训练损失曲线数据

怎么跑：
    python3 train.py                 # 默认配置
    python3 train.py --epochs 30 --batch 64
"""

import argparse
import json

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import UNet
from diffusion import DDPM


def main():
    parser = argparse.ArgumentParser(description="训练 DDPM 生成 MNIST 手写数字")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--T", type=int, default=300, help="扩散总步数")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 数据：归一化到 [-1, 1]（扩散模型标准做法）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)
    print(f"训练样本: {len(dataset)} 张，batch size: {args.batch}")

    # 模型 + 扩散过程
    model = UNet(in_ch=1, base=64, time_dim=64).to(device)
    ddpm = DDPM(model, T=args.T, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"epoch": [], "loss": []}
    print(f"\n开始训练，共 {args.epochs} 轮，扩散步数 T={args.T}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, _ in loader:
            x = x.to(device)
            loss = ddpm.train_loss(x)     # 预测噪声的 MSE
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(dataset)
        history["epoch"].append(epoch)
        history["loss"].append(round(avg_loss, 5))
        print(f"epoch {epoch:>2}/{args.epochs} | loss {avg_loss:.5f}")

    # 保存
    torch.save(model.state_dict(), "ddpm_mnist.pt")
    with open("train_loss.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print("\n训练完成。模型已存到 ddpm_mnist.pt，损失曲线存到 train_loss.json")
    print("下一步运行: python3 sample.py 生成手写数字和演变过程图")


if __name__ == "__main__":
    main()
