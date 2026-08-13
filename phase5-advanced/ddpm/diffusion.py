# -*- coding: utf-8 -*-
"""
diffusion.py —— DDPM 前向扩散 + 反向去噪

这个文件是干什么的：
    实现 DDPM（Denoising Diffusion Probabilistic Models）的核心数学：
      - 前向过程：把干净图片逐步加噪，直到变成纯高斯噪声
      - 反向过程：训练好的网络从噪声逐步去噪，还原出图片
    核心公式（面试要能讲）：
      前向  x_t = sqrt(ᾱ_t) · x_0 + sqrt(1-ᾱ_t) · ε        (ε 是标准高斯噪声)
      训练  让网络预测 ε，最小化 ||ε - ε_θ(x_t, t)||²
      采样  用预测的噪声一步步把 x_t 变回 x_{t-1}

跑完能看到什么：
    本文件不单独跑，由 train.py / sample.py 导入。

怎么跑：
    python3 train.py   # 训练
    python3 sample.py  # 采样生成图片
"""

import torch
import torch.nn.functional as F


class DDPM:
    def __init__(self, model, T=300, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.model = model
        self.T = T
        self.device = device

        # 噪声 schedule：β 从 1e-4 线性增加到 0.02（越往后噪声越大）
        self.betas = torch.linspace(beta_start, beta_end, T).to(device)
        self.alphas = 1.0 - self.betas                    # α_t = 1 - β_t
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)  # ᾱ_t = Π α_s

    def forward_diffuse(self, x0, t):
        """前向加噪：把干净图 x0 加噪到第 t 步，返回 (x_t, 真实噪声 ε)。"""
        sqrt_ab = torch.sqrt(self.alpha_bars[t])[:, None, None, None]
        sqrt_1m_ab = torch.sqrt(1 - self.alpha_bars[t])[:, None, None, None]
        eps = torch.randn_like(x0)
        xt = sqrt_ab * x0 + sqrt_1m_ab * eps
        return xt, eps

    def train_loss(self, x0):
        """训练一步的损失：随机选 t，加噪，预测噪声，算 MSE。"""
        t = torch.randint(0, self.T, (x0.shape[0],), device=self.device)
        xt, eps = self.forward_diffuse(x0, t)
        eps_pred = self.model(xt, t)
        return F.mse_loss(eps_pred, eps)

    @torch.no_grad()
    def sample(self, shape, return_trajectory=False, save_every=25):
        """反向去噪采样：从纯噪声逐步去噪，得到干净图片。

        如果 return_trajectory=True，会返回采样过程中的中间结果（每隔 save_every 步
        存一张），用来画"噪声 -> 图像"的演变过程图。
        """
        x = torch.randn(shape, device=self.device)
        trajectory = []

        for t in reversed(range(self.T)):
            t_tensor = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            eps_pred = self.model(x, t_tensor)

            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]

            # DDPM 采样更新公式
            coef = (1 - alpha) / torch.sqrt(1 - alpha_bar)
            mean = (x - coef * eps_pred) / torch.sqrt(alpha)

            if t > 0:
                z = torch.randn_like(x)
                sigma = torch.sqrt(self.betas[t])
            else:
                z = torch.zeros_like(x)
                sigma = 0.0
            x = mean + sigma * z

            if return_trajectory and (t % save_every == 0 or t == 0):
                trajectory.append(x.clone())

        if return_trajectory:
            return x, trajectory
        return x
