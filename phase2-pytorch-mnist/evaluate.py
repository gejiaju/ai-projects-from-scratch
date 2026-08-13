# -*- coding: utf-8 -*-
"""
evaluate.py —— 评估训练好的模型

这个文件是干什么的：
    加载 train.py 训练出来的模型权重，在测试集上重新评估，并输出：
      1. 总体准确率
      2. 每个数字(0~9)各自的准确率
    这能帮你发现"模型对哪个数字最容易认错"——面试时可以讲这个观察。

跑完能看到什么：
    终端打印总体准确率 + 每个数字的准确率表格。

怎么跑：
    python3 evaluate.py --model cnn
    python3 evaluate.py --model mlp
"""

import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import MLP, CNN


def main():
    parser = argparse.ArgumentParser(description="评估训练好的 MNIST 模型")
    parser.add_argument("--model", type=str, default="cnn", choices=["mlp", "cnn"])
    parser.add_argument("--weights", type=str, default=None,
                        help="权重文件路径，默认 results/<model>.pt")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 测试集（和训练时同样的归一化）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # 加载模型权重
    model = MLP().to(device) if args.model == "mlp" else CNN().to(device)
    weights = args.weights or f"results/{args.model}.pt"
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    print(f"已加载权重 {weights}")

    # 统计：总体准确率 + 每个类别的正确/总数
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            for i in range(10):
                class_correct[i] += ((predicted == labels) & (labels == i)).sum().item()
                class_total[i] += (labels == i).sum().item()

    print(f"\n总体准确率: {correct/total*100:.2f}%  ({correct}/{total})")
    print("\n每个数字的准确率：")
    print("-" * 32)
    print(f"{'数字':<6}{'正确':<8}{'总数':<8}{'准确率'}")
    print("-" * 32)
    for i in range(10):
        acc = class_correct[i] / class_total[i] * 100
        print(f"{i:<6}{class_correct[i]:<8}{class_total[i]:<8}{acc:.2f}%")
    print("-" * 32)


if __name__ == "__main__":
    main()
