#!/usr/bin/env python3
"""
运行 example.py，并开启加密与可视化追踪。

使用：
  python scripts/run_example_with_trace.py
可选：将 example.py 中的模型路径修改为你的本地路径。
"""

import os
import sys
from nanovllm.utils.secure import set_security_config
from nanovllm.utils.trace import set_trace_config

# 确保能从项目根目录导入 example.py
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import example


def main():
    # 开启两条加密链路，并在 CPU 上执行加密/解密（TEE 仿真）
    set_security_config(
        enable_softmax_encrypt=True,
        enable_linear_noise=True,
        encrypt_on_cpu=True,
        decrypt_on_cpu=True,
        noise_pool_size=16,
        noise_scale=0.05,
        seed=1234,
    )
    # 开启可视化追踪：仅打印第 0 层，每个关键模块仅打印一次，避免刷屏
    set_trace_config(enabled=True, head_items=4, max_calls_per_key=1, layer_filter=0)

    example.main()


if __name__ == "__main__":
    main()
