# -*- coding: utf-8 -*-
"""
train.py —— 训练 MLP 或 CNN（第二阶段核心脚本）

这个文件是干什么的：
    手写训练循环：加载 MNIST → 逐批前向传播 → 算 loss → 反向传播 → 更新参数。
    每训练完一轮(epoch)，在训练集和测试集上各算一次准确率，记录下来。
    MLP 和 CNN 走的是**完全相同的**训练代码和超参数，只换模型——这样对比才公平。

跑完能看到什么：
    1. 终端打印每一轮的 loss 和训练/测试准确率
    2. results/<model>_history.json  保存每轮指标（给 compare.py 画曲线用）
    3. results/<model>.pt             保存训练好的模型权重

怎么跑：
    python3 train.py --model cnn                # 训练 CNN（默认）
    python3 train.py --model mlp                # 训练 MLP
    python3 train.py --model cnn --epochs 10 --lr 0.01   # 自定义轮数和学习率
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import MLP, CNN

# 固定随机种子，保证实验可复现
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_dataloaders(batch_size):
    """加载 MNIST 数据集。第一次运行会自动下载。"""
    transform = transforms.Compose([
        transforms.ToTensor(),                 # 图片 -> 0~1 的张量
        transforms.Normalize((0.1307,), (0.3081,)),  # 用 MNIST 的均值和标准差做归一化
    ])
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def evaluate(model, loader, device):
    """在给定数据上算准确率（不更新参数）。"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train_one_epoch(model, loader, optimizer, criterion, device):
    """训练一轮，返回这一轮的平均 loss。"""
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # 清空上一轮的梯度
        outputs = model(images)        # 前向传播
        loss = criterion(outputs, labels)  # 算损失
        loss.backward()                # 反向传播（自动求梯度）
        optimizer.step()               # 更新参数

        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(description="训练 MLP 或 CNN 做 MNIST 分类")
    parser.add_argument("--model", type=str, default="cnn", choices=["mlp", "cnn"])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"设备: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. 数据
    train_loader, test_loader = get_dataloaders(args.batch_size)

    # 2. 模型（MLP 和 CNN 用同一个入口）
    if args.model == "mlp":
        model = MLP().to(device)
    else:
        model = CNN().to(device)

    # 3. 损失函数 + 优化器（两种模型完全一致，保证公平）
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    history = {"epoch": [], "train_loss": [], "train_acc": [], "test_acc": []}
    os.makedirs("results", exist_ok=True)

    print(f"\n开始训练 {args.model.upper()}，共 {args.epochs} 轮\n")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_acc = evaluate(model, train_loader, device)
        test_acc = evaluate(model, test_loader, device)
        elapsed = time.time() - start

        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["test_acc"].append(round(test_acc, 4))

        print(f"epoch {epoch:>2}/{args.epochs} | loss {train_loss:.4f} | "
              f"train_acc {train_acc*100:.2f}% | test_acc {test_acc*100:.2f}% | "
              f"{elapsed:.1f}s")

    # 4. 保存结果
    json_path = f"results/{args.model}_history.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    torch.save(model.state_dict(), f"results/{args.model}.pt")
    print(f"\n训练完成。曲线数据已存到 {json_path}，模型已存到 results/{args.model}.pt")
    print(f"最终测试准确率: {history['test_acc'][-1]*100:.2f}%")


if __name__ == "__main__":
    main()
