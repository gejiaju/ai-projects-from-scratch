# -*- coding: utf-8 -*-
"""
04_file_io.py —— 文件读写（CSV / JSON / 文本）

这个文件是干什么的：
    训练结果要保存成文件、数据集要读进内存、配置要存成 JSON。本文件演示
    三种最常见的文件读写：CSV（表格数据）、JSON（结构化配置）、TXT（日志）。
    跑完后会生成两个文件，并重新读出来打印，验证"写进去的 = 读出来的"。

跑完能看到什么：
    终端打印读写结果；当前目录生成 result.csv 和 config.json 两个文件。

怎么跑：
    python3 04_file_io.py
"""

import csv
import json
import os


def demo_csv():
    """CSV 读写：表格数据（实验记录、标注文件）最常用。"""
    print("=" * 60)
    print("1. CSV 读写")
    # 写入：模拟一次实验记录（模型名 / 准确率 / 训练轮数）
    rows = [
        ["model", "accuracy", "epoch"],
        ["MLP", "97.2", "10"],
        ["CNN", "99.1", "10"],
    ]
    with open("result.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print("已写入 result.csv")

    # 读取：逐行读出来
    with open("result.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)          # DictReader 把每行变成字典
        for row in reader:
            print(f"  模型 {row['model']}: 准确率 {row['accuracy']}%, "
                  f"训练 {row['epoch']} 轮")


def demo_json():
    """JSON 读写：保存配置（超参数、路径），程序启动时读取。"""
    print("=" * 60)
    print("2. JSON 读写")
    config = {
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 10,
        "model": "CNN",
    }
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print("已写入 config.json")

    with open("config.json", "r", encoding="utf-8") as f:
        loaded = json.load(f)
    print(f"  读回的学习率 learning_rate = {loaded['learning_rate']}")


def demo_txt():
    """文本读写：写训练日志。"""
    print("=" * 60)
    print("3. 文本文件读写")
    lines = [
        "epoch 1: loss=1.83\n",
        "epoch 2: loss=0.94\n",
        "epoch 3: loss=0.51\n",
    ]
    with open("train_log.txt", "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("已写入 train_log.txt")

    # with 语句会自动关闭文件，也可以显式用 open/close
    f = open("train_log.txt", "r", encoding="utf-8")
    content = f.read()
    f.close()
    print("  读到的日志:\n" + content)


def main():
    demo_csv()
    demo_json()
    demo_txt()
    print("=" * 60)
    print("文件读写演示完毕。可检查当前目录下的 result.csv / config.json / train_log.txt")


if __name__ == "__main__":
    main()
