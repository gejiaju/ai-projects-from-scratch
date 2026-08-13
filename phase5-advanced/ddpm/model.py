# -*- coding: utf-8 -*-
"""
model.py —— DDPM 的 U-Net 主干网络

这个文件是干什么的：
    扩散模型的核心是一个"噪声预测网络"，通常用 U-Net 结构。
    U-Net 的特点是：先下采样（压缩、提取全局信息），再上采样（恢复分辨率、还原细节），
    中间用 skip connection 把浅层细节直接传给深层，避免信息丢失。
    本文件还实现了"时间步 embedding"——把"当前是第几步加噪"这个信息注入网络，
    让网络知道该预测多大程度的噪声。

跑完能看到什么：
    本文件不单独跑，由 train.py / sample.py 导入。单独运行会打印网络参数量并
    验证前向传播的形状正确。

怎么跑（单独验证）：
    python3 model.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gnorm(channels):
    """构造 GroupNorm，group 数自适应通道数（1 通道图也能用）。"""
    groups = 32
    while channels % groups != 0 and groups > 1:
        groups //= 2
    return nn.GroupNorm(groups, channels)


def sinusoidal_embedding(timesteps, dim):
    """把时间步 t（整数）编码成 dim 维的向量（正弦位置编码）。

    为什么需要它：U-Net 要"知道"自己正在给第几步的图去噪——第 1 步和第 500 步
    要预测的噪声完全不同。用正弦编码把 t 变成一个高维向量，网络就能区分不同步数。
    """
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
    freqs = freqs.to(timesteps.device)
    args = timesteps.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResidualBlock(nn.Module):
    """带时间注入的残差块：GroupNorm -> Conv -> 加时间信息 -> Conv -> 残差连接。"""

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = _gnorm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        # 时间信息通过一个 MLP 变成 (out_ch,) 的向量，加到特征图的每个像素上
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.norm2 = _gnorm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        # 残差连接：通道数不变就直连，变了就用 1x1 卷积对齐
        self.residual = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, time_emb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        # 时间信息注入：time_emb -> (B, out_ch) -> (B, out_ch, 1, 1) 广播到每个位置
        h = h + self.time_mlp(time_emb)[:, :, None, None]
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.residual(x)


class UNet(nn.Module):
    """U-Net：下采样(编码) -> 中间 -> 上采样(解码)，带 skip connection。"""

    def __init__(self, in_ch=1, base=64, time_dim=64):
        super().__init__()
        self.time_dim = time_dim

        # 下采样路径（通道逐渐变多，分辨率逐渐变小）
        self.enc1 = ResidualBlock(in_ch, base, time_dim)          # 28x28
        self.enc2 = ResidualBlock(base, base * 2, time_dim)       # 14x14
        self.enc3 = ResidualBlock(base * 2, base * 4, time_dim)   # 7x7

        # 中间层（7x7，通道最多）
        self.mid = ResidualBlock(base * 4, base * 4, time_dim)

        # 上采样路径：和"同尺度"的编码特征做 skip 拼接，所以通道要相加
        self.dec2 = ResidualBlock(base * 4 + base * 2, base * 2, time_dim)  # 14x14
        self.dec1 = ResidualBlock(base * 2 + base, base, time_dim)          # 28x28

        self.down = nn.MaxPool2d(2)          # 下采样（减半）
        self.out = nn.Conv2d(base, in_ch, 1)  # 输出：预测的噪声（和输入同形状）

    def forward(self, x, t):
        # 时间步 -> 正弦编码
        time_emb = sinusoidal_embedding(t, self.time_dim)

        # 编码器（逐级下采样）
        e1 = self.enc1(x, time_emb)                 # (B, 64, 28, 28)
        e2 = self.enc2(self.down(e1), time_emb)     # (B, 128, 14, 14)
        e3 = self.enc3(self.down(e2), time_emb)     # (B, 256, 7, 7)

        # 中间（保持 7x7）
        m = self.mid(e3, time_emb)                  # (B, 256, 7, 7)

        # 解码器（上采样，并与同尺度编码特征拼接）
        d2 = F.interpolate(m, scale_factor=2, mode="nearest")     # (B, 256, 14, 14)
        d2 = self.dec2(torch.cat([d2, e2], dim=1), time_emb)      # (B, 128, 14, 14)
        d1 = F.interpolate(d2, scale_factor=2, mode="nearest")    # (B, 128, 28, 28)
        d1 = self.dec1(torch.cat([d1, e1], dim=1), time_emb)      # (B, 64, 28, 28)

        return self.out(d1)   # (B, 1, 28, 28) = 预测的噪声


if __name__ == "__main__":
    model = UNet(in_ch=1, base=64, time_dim=64)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net 参数量: {n_params:,}")

    # 验证前向传播形状
    x = torch.randn(4, 1, 28, 28)
    t = torch.randint(0, 1000, (4,))
    out = model(x, t)
    print("输入形状:", tuple(x.shape))
    print("输出形状:", tuple(out.shape), "(应与输入一致，因为预测的是噪声)")
