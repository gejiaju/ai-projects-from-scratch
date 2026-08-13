# 第一阶段：Python 基础（为 AI 定制）

> 这一阶段不系统学 Python，只学"AI 用得上"的那部分：NumPy 数组、Matplotlib 画图、
> class 写法、文件读写，以及用 argparse 把脚本做成"能传参数"的小工具。

## 为什么做这个

深度学习的代码 90% 是围绕"数据"打转：读数据、变成数组（张量）、算一遍、把结果画出来。
这五个文件正好覆盖这四件事，而且刻意设计成和第二阶段的 MNIST 项目能衔接——

- 这里生成的 `experiment_results.csv` 是"不同模型的准确率对比"，
  第二阶段你真的会训练 MLP 和 CNN，然后画出同样一张对比柱状图。
- 这里手写的 `LinearLayer` 类，和第二阶段 PyTorch 里的 `nn.Linear` 是同一个东西。

## 文件清单

| 文件 | 干什么 | 跑完看到什么 |
|------|--------|--------------|
| `01_numpy_basics.py` | NumPy 数组操作（创建/切片/广播/矩阵乘/统计/随机数） | 终端打印每一步结果 |
| `02_matplotlib_plot.py` | 画折线图/柱状图/散点图/子图 | 生成 4 张 PNG 图 |
| `03_class_example.py` | class 写法（手写全连接层 + 两层网络） | 终端打印前向传播过程 |
| `04_file_io.py` | CSV/JSON/文本文件读写 | 生成并读回 3 个文件 |
| `generate_data.py` | 用 argparse 生成模拟实验数据 CSV | 生成 `experiment_results.csv` |
| `analyze_csv.py` | **核心脚本**：读 CSV → 算统计 → 画柱状图 | 打印统计表 + 生成柱状图 |

## 怎么跑

在 WSL 终端里，进入本目录后逐条执行：

```bash
cd /mnt/f/yolo/phase1-python-basics

# 1. 逐个跑基础脚本（只看终端输出）
python3 01_numpy_basics.py
python3 02_matplotlib_plot.py
python3 03_class_example.py
python3 04_file_io.py

# 2. 完整验证流程：造数据 → 分析 → 画图
python3 generate_data.py              # 生成 experiment_results.csv
python3 analyze_csv.py                # 读它、算统计、画柱状图
```

## 过关标准（对照 CLAUDE.md）

- [x] 能读 CSV 文件
- [x] 能计算统计量（均值/标准差/最大最小）
- [x] 能画柱状图并保存为图片
- [x] 用 `argparse` 传参数，脚本可复用

全部跑通后，你已经掌握了深度学习代码里 80% 的"体力活"，可以直接进入第二阶段。
