# -*- coding: utf-8 -*-
import torch
from typing import Optional, Sequence, Tuple, Dict

class EncryptedRotation:
    """
    生成并应用 R = P @ B 的块对角+置换旋转矩阵（强制 d % n == 0）：
      - B: 对角线上堆叠若干 n×n 的正交旋转块（det=+1）
      - P: 置换矩阵（每行/列恰一 个 1），可选偶置换（det=+1）
      - R: 正交 => R^{-1} = R^T
    """

    def __init__(self, d: int, n: int, det_one: bool = True,
                 device: str = "cpu", dtype: torch.dtype = torch.float32):
        assert 1 <= n <= d, "n 必须在 [1, d] 内"
        assert d % n == 0, "要求 d % n == 0"
        self.d, self.n = d, n
        self.det_one = det_one
        self.device, self.dtype = device, dtype

    @staticmethod
    def _random_rotation_block(n: int, device="cpu", dtype=torch.float32) -> torch.Tensor:
        A = torch.randn(n, n, device=device, dtype=dtype)
        Q, _ = torch.linalg.qr(A)
        if torch.linalg.det(Q) < 0:
            Q[:, -1] = -Q[:, -1]
        return Q

    def _make_blockdiag_rotation(self) -> torch.Tensor:
        d, n = self.d, self.n
        B = torch.zeros((d, d), device=self.device, dtype=self.dtype)
        for i in range(0, d, n):
            blk = self._random_rotation_block(n, device=self.device, dtype=self.dtype)
            B[i:i+n, i:i+n] = blk
        return B  # det(B)=+1

    @staticmethod
    def _perm_parity_is_odd(sigma: torch.Tensor) -> bool:
        d = sigma.numel()
        visited = torch.zeros(d, dtype=torch.bool, device=sigma.device)
        transpositions = 0
        for i in range(d):
            if not visited[i]:
                j, cyc = i, 0
                while not visited[j]:
                    visited[j] = True
                    j = int(sigma[j])
                    cyc += 1
                if cyc > 0:
                    transpositions += cyc - 1
        return (transpositions % 2 == 1)

    def _make_permutation(self) -> Tuple[torch.Tensor, torch.Tensor]:
        d = self.d
        sigma = torch.randperm(d, device=self.device)
        if self.det_one and d >= 2 and self._perm_parity_is_odd(sigma):
            sigma[-1], sigma[-2] = sigma[-2], sigma[-1]  # 变偶置换
        P = torch.zeros((d, d), device=self.device, dtype=self.dtype)
        rows = torch.arange(d, device=self.device)
        P[rows, sigma] = 1.0
        return P, sigma

    def make_rotation(self, seed: Optional[int] = None) -> Tuple[torch.Tensor, Dict]:
        if seed is not None:
            torch.manual_seed(seed)
        B = self._make_blockdiag_rotation()
        P, sigma = self._make_permutation()
        R = P @ B
        I = torch.eye(self.d, device=self.device, dtype=self.dtype)
        info = {
            "sigma": sigma,
            "orth_err": torch.linalg.norm(R.T @ R - I).item(),
            "det": float(torch.linalg.det(R.to(torch.float64)))
        }
        return R, info

    def make_rotation_batch(self, num_heads: int,
                            seeds: Optional[Sequence[int]] = None) -> Tuple[torch.Tensor, Dict]:
        H = num_heads
        if seeds is None: seeds = [None] * H
        assert len(seeds) == H
        Rs, sigmas, dets, errs = [], [], [], []
        for s in seeds:
            R, info = self.make_rotation(seed=s)
            Rs.append(R); sigmas.append(info["sigma"]); dets.append(info["det"]); errs.append(info["orth_err"])
        R_stack = torch.stack(Rs, 0)              # [H, d, d]
        info = {"sigma": torch.stack(sigmas, 0), "det": torch.tensor(dets), "orth_err": torch.tensor(errs)}
        return R_stack, info

    @staticmethod
    def apply(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        if R.dim() == 2:        # [..., d] @ [d, d]
            return torch.matmul(x, R)
        elif R.dim() == 3:      # ...hd, hdd -> ...he
            return torch.einsum("...hd,hde->...he", x, R)
        raise ValueError("R 的维度应为 2 或 3")

    @staticmethod
    def verify_orthogonal(R: torch.Tensor) -> Dict[str, float]:
        d = R.shape[-1]
        I = torch.eye(d, device=R.device, dtype=R.dtype)
        return {
            "orth_err": torch.linalg.norm(R.transpose(-1, -2) @ R - I).item(),
            "inv_eq_transpose_err": torch.linalg.norm(torch.linalg.inv(R) - R.transpose(-1, -2)).item(),
            "det": float(torch.linalg.det(R.to(torch.float64))) if R.dim() == 2 else None
        }
