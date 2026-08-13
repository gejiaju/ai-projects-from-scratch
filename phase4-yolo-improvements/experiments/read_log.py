# -*- coding: utf-8 -*-
"""读取训练日志，提取 optimizer / lr 相关行（过滤进度条干扰）。"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/abl_test.log"
with open(path, encoding="utf-8", errors="ignore") as f:
    content = f.read().replace("\r", "\n")

keywords = ["optimizer", "SGD", "AdamW", "lr0", "momentum", "warmup", "auto"]
for line in content.splitlines():
    if any(k in line for k in keywords):
        print(line)
