# -*- coding: utf-8 -*-
"""查 ultralytics parse_model 如何解析模块名，以及有没有 add_custom_module。"""
import inspect

import ultralytics.nn.tasks as tasks

src = inspect.getsource(tasks.parse_model)
print("=== parse_model 里和模块名解析相关的行 ===")
for i, line in enumerate(src.splitlines()):
    low = line.lower()
    if any(k in low for k in ["getattr", "globals", "custom", "eval", "m =", "import"]):
        print(f"{i:>3}: {line}")

# 查 add_custom_module 是否存在
print("\n=== 查找 add_custom_module ===")
import ultralytics.utils as u
print("ultralytics.utils 有 add_custom_module:", hasattr(u, "add_custom_module"))
import ultralytics.nn.modules as m
print("ultralytics.nn.modules 有 add_custom_module:", hasattr(m, "add_custom_module"))
