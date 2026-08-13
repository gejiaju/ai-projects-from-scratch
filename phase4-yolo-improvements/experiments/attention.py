# -*- coding: utf-8 -*-
"""
attention.py —— SE 和 CBAM 注意力模块（第四阶段改进方向）

这个文件是干什么的：
    定义两种经典注意力模块，并把它们"塞进"YOLOv8 的 C2f 模块，得到 C2fSE / C2fCBAM。
      - SE（Squeeze-and-Excitation）：通道注意力——自动学习"哪些通道更重要"
      - CBAM（Convolutional Block Attention Module）：通道注意力 + 空间注意力
        既学"哪些通道重要"，又学"图上哪些位置重要"
    这两个是目标检测里最常用的"加注意力提升精度"手段。

跑完能看到什么：
    本文件不单独跑，由 run_ablation.py 导入。单独运行会验证模块前向形状正确。

怎么跑（单独验证）：
    python3 attention.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import C2f


class SE(nn.Module):
    """Squeeze-and-Excitation：给每个通道学一个 0~1 的权重，加权到特征图上。

    思路：先"挤压"（全局平均池化，得到每个通道的一个数），再"激励"
    （两个全连接层学出权重），最后把权重乘回每个通道。
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # 每个通道 -> 1 个数
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),                     # 权重压到 0~1
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        weight = self.fc(x).view(b, c, 1, 1)
        return x * weight                    # 按通道加权


class CBAM(nn.Module):
    """CBAM：先做通道注意力，再做空间注意力，两个维度都加权。"""

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        # 通道注意力：用 max-pool 和 avg-pool 两路，共享一个 MLP
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        # 空间注意力：沿通道做 max/avg，再用 7x7 卷积学空间权重
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        # ---- 通道注意力 ----
        avg = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        max_ = F.adaptive_max_pool2d(x, 1).view(x.size(0), -1)
        channel_att = torch.sigmoid(self.mlp(avg) + self.mlp(max_))
        x = x * channel_att.view(x.size(0), x.size(1), 1, 1)

        # ---- 空间注意力 ----
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * spatial_att


class C2fSE(C2f):
    """在 C2f 之后加 SE 注意力。继承 C2f，参数签名和原始 C2f 完全一致。"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.attention = SE(c2)

    def forward(self, x):
        return self.attention(super().forward(x))

    @classmethod
    def from_c2f(cls, c2f):
        """从已有的 C2f 实例提取参数，构造结构等价且**保留预训练权重**的 C2fSE。

        只复制 C2f 本体（cv1/cv2/bottleneck）的权重，新增的 SE 注意力层随机初始化。
        这样微调时模型从预训练基础出发，而不是从零开始。
        """
        c1 = c2f.cv1.conv.in_channels
        c2 = c2f.cv2.conv.out_channels
        n = len(c2f.m)
        shortcut = bool(c2f.m[0].add) if n > 0 else False
        obj = cls(c1, c2, n=n, shortcut=shortcut)
        obj.cv1.load_state_dict(c2f.cv1.state_dict())
        obj.cv2.load_state_dict(c2f.cv2.state_dict())
        for i, bottleneck in enumerate(c2f.m):
            obj.m[i].load_state_dict(bottleneck.state_dict())
        return obj


class C2fCBAM(C2f):
    """在 C2f 之后加 CBAM 注意力。"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.attention = CBAM(c2)

    def forward(self, x):
        return self.attention(super().forward(x))

    @classmethod
    def from_c2f(cls, c2f):
        """从已有的 C2f 实例提取参数，构造结构等价且**保留预训练权重**的 C2fCBAM。"""
        c1 = c2f.cv1.conv.in_channels
        c2 = c2f.cv2.conv.out_channels
        n = len(c2f.m)
        shortcut = bool(c2f.m[0].add) if n > 0 else False
        obj = cls(c1, c2, n=n, shortcut=shortcut)
        obj.cv1.load_state_dict(c2f.cv1.state_dict())
        obj.cv2.load_state_dict(c2f.cv2.state_dict())
        for i, bottleneck in enumerate(c2f.m):
            obj.m[i].load_state_dict(bottleneck.state_dict())
        return obj


def inject_attention(module, attention_cls):
    """递归遍历模型，把里面所有的 C2f 替换成 attention_cls（C2fSE 或 C2fCBAM）。

    为什么这么做：ultralytics 的 parse_model 对 C2f 有特殊处理（会把 repeats 插进
    参数里），但自定义的 C2fSE/C2fCBAM 不享受这个待遇，直接写 yaml 会参数错位。
    所以改成先构建 baseline，再在代码里把 C2f 就地替换成带注意力的版本。
    这样结构完全等价，只有"加不加注意力"这一个变量，消融实验才公平。
    """
    for name, child in list(module.named_children()):
        if isinstance(child, C2f):
            setattr(module, name, attention_cls.from_c2f(child))
        else:
            inject_attention(child, attention_cls)


if __name__ == "__main__":
    # 单独运行：验证模块前向形状正确
    x = torch.randn(2, 64, 32, 32)
    se = SE(64)
    cbam = CBAM(64)
    c2f_se = C2fSE(64, 64, n=3)
    c2f_cbam = C2fCBAM(64, 64, n=3)
    print("输入形状:", tuple(x.shape))
    print("SE 输出形状:", tuple(se(x).shape), "(应不变)")
    print("CBAM 输出形状:", tuple(cbam(x).shape), "(应不变)")
    print("C2fSE 输出形状:", tuple(c2f_se(x).shape))
    print("C2fCBAM 输出形状:", tuple(c2f_cbam(x).shape))
    print("\nSE 参数量:", sum(p.numel() for p in se.parameters()))
    print("CBAM 参数量:", sum(p.numel() for p in cbam.parameters()))
    print("说明：注意力模块只加了极少量参数，几乎不影响推理速度。")
