import math
import os
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SecurityConfig:
    enable_softmax_encrypt: bool = False
    enable_linear_noise: bool = False
    encrypt_on_cpu: bool = True
    decrypt_on_cpu: bool = True
    noise_pool_size: int = 16
    noise_scale: float = 0.05
    seed: int = 1234


_CONFIG = SecurityConfig()


def get_security_config() -> SecurityConfig:
    return _CONFIG


def set_security_config(**kwargs):
    global _CONFIG
    for k, v in kwargs.items():
        if hasattr(_CONFIG, k):
            setattr(_CONFIG, k, v)


def _manual_seed(seed: int):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


def orthogonal_matrix(dim: int, dtype: torch.dtype, device: torch.device, seed: Optional[int] = None) -> torch.Tensor:
    """
    生成维度为 dim 的正交矩阵 R，使得 R^{-1} = R^T。
    使用 QR 分解保证数值稳定，R 取 Q。
    """
    g = _manual_seed(_CONFIG.seed if seed is None else seed)
    # 注意：在模型构建阶段 global default device 可能被设为 cuda
    # 这里显式在 CPU 上采样，避免与 CPU generator 不匹配
    a = torch.randn((dim, dim), dtype=torch.float32, device="cpu", generator=g)
    q, r = torch.linalg.qr(a)
    # 提升到目标 dtype/device
    q = q.to(dtype=dtype, device=device)
    return q


class NoisePool:
    """
    噪声池：提供固定数量的输入侧噪声向量 r（形状 [in_features]），
    并在权重确定后预计算补偿项 rW（形状 [out_features]）。
    在前向中：x' = x - r，GPU 计算 y' = x' W^T + b，CPU 端（或 GPU）加回 rW 以恢复 y。
    """

    def __init__(self, in_features: int, out_features: int, pool_size: int, noise_scale: float, seed: int = 1234):
        self.in_features = in_features
        self.out_features = out_features
        self.pool_size = pool_size
        self.noise_scale = noise_scale
        self.seed = seed
        self._rng = _manual_seed(seed)

        # r_pool: [P, in_features] 存在 CPU 上，避免与 GPU 权重生命周期耦合
        # 显式在 CPU 上采样，避免 default device=CUDA 时与 CPU generator 冲突
        self.r_pool_cpu = torch.randn((pool_size, in_features), device="cpu", generator=self._rng) * noise_scale
        self.r_pool_cpu = self.r_pool_cpu.to(dtype=torch.float32, device="cpu")

        # rW_pool: [P, out_features]，在 set_weight 之后计算
        self.rw_pool_cpu: Optional[torch.Tensor] = None

    def set_weight(self, weight_shard: torch.Tensor):
        """
        在权重装载/更新后调用，预计算 rW。
        weight_shard: 形状 [out_features, in_features] 的本 rank 权重切片。
        计算：rW = r @ W^T -> [P, out_features]
        """
        assert weight_shard.dim() == 2
        out_features, in_features = weight_shard.shape
        assert out_features == self.out_features and in_features == self.in_features
        # 使用 CPU 计算，符合“CPU 可用于解密/补偿”的需求
        w_cpu = weight_shard.detach().to(dtype=torch.float32, device="cpu")
        self.rw_pool_cpu = self.r_pool_cpu @ w_cpu.T

    def sample(self, index: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        采样一个噪声 r 及其补偿 rW（均在 CPU 上），返回 (r_cpu, rW_cpu, idx)。
        """
        assert self.rw_pool_cpu is not None, "NoisePool: rW 尚未预计算，请在权重加载后调用 set_weight()。"
        if index is None:
            # 显式在 CPU 上采样索引，避免默认设备为 CUDA 时与 CPU generator 冲突
            idx = int(torch.randint(0, self.pool_size, (1,), device="cpu", generator=self._rng).item())
        else:
            idx = int(index)
        r_cpu = self.r_pool_cpu[idx]
        rw_cpu = self.rw_pool_cpu[idx]
        return r_cpu, rw_cpu, idx
