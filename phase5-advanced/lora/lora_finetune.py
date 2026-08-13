# -*- coding: utf-8 -*-
"""
lora_finetune.py —— 用 LoRA 微调大语言模型

这个文件是干什么的：
    LoRA（Low-Rank Adaptation，低秩适配）是大模型微调的核心技术：
    不更新原模型的全部参数，而是给每层加一个"低秩"的小矩阵，只训练这个小矩阵。
    这样能把几百亿参数的模型微调成本降到"几百万参数"，普通显卡就能跑。

    本脚本用 Qwen2.5 小模型 + 一份"电子信息考研面试"主题的数据做 LoRA 微调，
    让模型学会用"考研面试问答"的风格回答问题——和备考目标直接相关。

跑完能看到什么：
    1. 终端打印训练进度和损失
    2. 保存微调后的 LoRA 权重到 lora_output/
    3. 微调前后各问同一个问题，对比模型回答的变化

怎么跑：
    # 先装依赖（本机未装，需先安装）
    pip install --break-system-packages transformers peft accelerate datasets

    # 再跑微调（需要联网下载 Qwen2.5-0.5B 模型，约 1GB）
    python3 lora_finetune.py
"""

import argparse


# 微调用的小数据集：电子信息考研面试问答（和备考目标直接相关）
TRAINING_DATA = [
    {"instruction": "什么是卷积神经网络？", "output": "卷积神经网络（CNN）是一种专门处理网格状数据（如图像）的深度学习模型，核心是卷积层、池化层和全连接层。卷积层用共享的卷积核在输入上滑动提取局部特征，池化层下采样降低计算量，全连接层做最终分类。"},
    {"instruction": "解释一下梯度下降", "output": "梯度下降是一种优化算法，通过沿着损失函数梯度的反方向迭代更新参数，使损失逐步减小。学习率决定每步更新多大，太大可能震荡，太小收敛慢。"},
    {"instruction": "什么是过拟合？怎么解决？", "output": "过拟合是指模型在训练集上表现很好但在测试集上表现差，说明模型记住了训练数据的噪声。解决方法包括增加数据、正则化、Dropout、早停和降低模型复杂度。"},
    {"instruction": "介绍一下 YOLO 目标检测", "output": "YOLO 是一种单阶段目标检测算法，把检测看作回归问题，一次前向传播同时预测目标框和类别，速度快、适合实时场景。核心思想是把图分成网格，每个网格负责预测落在其中的目标。"},
]


def load_model_and_tokenizer(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
    return model, tokenizer


def apply_lora(model):
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(
        r=8,              # LoRA 秩（越小参数越少）
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 只微调注意力层
        lora_dropout=0.1,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()  # 打印"可训练参数 / 总参数"，体现 LoRA 的价值
    return model


def format_prompt(instruction):
    """构造 chat 模板。"""
    return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"


def main():
    parser = argparse.ArgumentParser(description="LoRA 微调大语言模型")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    import torch
    print(f"设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    print(f"加载模型 {args.model} ...")
    model, tokenizer = load_model_and_tokenizer(args.model)

    print("应用 LoRA ...")
    model = apply_lora(model)

    # 把数据转成 token id
    from torch.utils.data import DataLoader, Dataset

    class QADataset(Dataset):
        def __init__(self, data, tokenizer):
            self.samples = []
            for d in data:
                text = format_prompt(d["instruction"]) + d["output"] + "<|im_end|>"
                self.samples.append(tokenizer(text, truncation=True, max_length=256)["input_ids"])

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, i):
            return torch.tensor(self.samples[i], dtype=torch.long)

    dataset = QADataset(TRAINING_DATA, tokenizer)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"\n开始 LoRA 微调，共 {args.epochs} 轮\n")
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in loader:
            batch = batch.to(model.device)
            outputs = model(input_ids=batch, labels=batch)  # 自回归：预测下一个 token
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch {epoch+1}/{args.epochs} | loss {total_loss/len(loader):.4f}")

    model.save_pretrained("lora_output")
    print("\nLoRA 权重已保存到 lora_output/")

    # 微调后问一个问题，对比效果
    model.eval()
    q = "什么是过拟合？"
    inputs = tokenizer(format_prompt(q), return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100)
    answer = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"\n微调后回答「{q}」：\n{answer}")


if __name__ == "__main__":
    main()
