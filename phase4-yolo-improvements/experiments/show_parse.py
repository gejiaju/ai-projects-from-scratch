# -*- coding: utf-8 -*-
"""打印 parse_model 关键源码，理解它如何给 C2f 类模块传参。"""
import inspect
import ultralytics.nn.tasks as tasks

src = inspect.getsource(tasks.parse_model).splitlines()
for i, line in enumerate(src):
    if 55 <= i <= 150:
        print(f"{i:>3}: {line}")
