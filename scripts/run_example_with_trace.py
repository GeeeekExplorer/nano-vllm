#!/usr/bin/env python3
"""
运行 example.py，并开启加密与可视化追踪。

使用：
  python scripts/run_example_with_trace.py
可选：将 example.py 中的模型路径修改为你的本地路径。
"""

from nanovllm.utils.secure import set_security_config
from nanovllm.utils.trace import set_trace_config
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
    # 开启可视化追踪：每个关键模块仅打印一次，避免刷屏
    set_trace_config(enabled=True, head_items=6, max_calls_per_key=1)

    example.main()


if __name__ == "__main__":
    main()

