#!/usr/bin/env python3
"""
演示两条加密链路：
1) 线性层“输入侧加噪 + rW 预计算”的 TEE 流程（CPU 加密/解密，GPU 计算密文）
2) 注意力 Q/K 的正交加密（R 矩阵），验证分数不变与 RMS 不变

运行：
  python scripts/demo_secure_flow.py --device cuda   # 若有 GPU
  python scripts/demo_secure_flow.py --device cpu    # 无 GPU 时也可演示（用 CPU 模拟）
"""

import argparse
import torch
import torch.nn.functional as F

from nanovllm.utils.secure import NoisePool, orthogonal_matrix


def _to_cpu_f32(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to(device="cpu", dtype=torch.float32)


def _print_tensor(name: str, t: torch.Tensor, max_items: int = 6):
    flat = t.flatten()
    vals = flat[:max_items].tolist()
    print(f"{name} shape={tuple(t.shape)} device={t.device} dtype={t.dtype} head={vals}")


@torch.inference_mode()
def demo_linear_noise(device: torch.device, *, B=2, Din=8, Dout=6, noise_scale=0.05):
    print("\n=== 线性层：输入侧加噪 + rW 预计算（TEE: CPU | Untrusted: GPU）===")
    has_gpu = device.type == "cuda"
    tee = torch.device("cpu")
    accel = device

    # 准备明文输入与权重
    torch.manual_seed(0)
    x_cpu = torch.randn(B, Din, dtype=torch.float16, device=tee)  # 明文在 CPU(TEE)
    W = torch.randn(Dout, Din, dtype=torch.float16, device=accel)  # 权重在 GPU(不可信)
    b = torch.randn(Dout, dtype=torch.float16, device=accel)

    # 参考输出（明文在不可信端仅用于验证，真实 TEE 中不应直接发送明文）
    y_ref = F.linear(x_cpu.to(accel), W, b).float().cpu()

    # 构建噪声池并预计算 rW（在 CPU）
    pool = NoisePool(Din, Dout, pool_size=16, noise_scale=noise_scale, seed=1234)
    pool.set_weight(W)  # 会复制权重到 CPU 进行 rW 预计算
    r_cpu, rw_cpu, idx = pool.sample()

    # CPU(TEE) 侧加密：x' = x - r
    x_cpu_f32 = _to_cpu_f32(x_cpu)
    r = r_cpu.to(dtype=x_cpu_f32.dtype)
    x_masked_cpu = x_cpu_f32 - r.unsqueeze(0)

    # 发送密文到不可信加速器做线性
    x_masked = x_masked_cpu.to(accel, dtype=x_cpu.dtype)
    y_masked = F.linear(x_masked, W, b).float()  # y' = (x - r) W^T + b

    # CPU(TEE) 侧解密补偿：y = y' + rW
    y_cpu = _to_cpu_f32(y_masked)
    y_rec = y_cpu + rw_cpu.unsqueeze(0)

    # 对比
    max_abs_err = (y_ref - y_rec).abs().max().item()
    print(f"噪声样本 idx={idx}")
    _print_tensor("x (明文, CPU)", x_cpu)
    _print_tensor("r (TEE, CPU)", r_cpu)
    _print_tensor("x_masked (密文, 发送到不可信端)", x_masked)
    _print_tensor("y_masked (不可信端计算产物)", y_masked)
    _print_tensor("rW (TEE 预计算)", rw_cpu)
    _print_tensor("y_rec (TEE 解密后)", y_rec)
    print(f"与明文直算 y_ref 的最大绝对误差: {max_abs_err:.3e}")
    print("判定: ", "通过" if torch.allclose(y_ref, y_rec, atol=1e-3, rtol=1e-3) else "未通过")


@torch.inference_mode()
def demo_qk_orthogonal(device: torch.device, *, Nq=4, Nk=3, H=2, D=8):
    print("\n=== 注意力：Q/K 正交加密（TEE: CPU | Untrusted: GPU）===")
    tee = torch.device("cpu")
    accel = device

    torch.manual_seed(1)
    # 构造 Query/Key 明文（在 TEE）
    Q_cpu = torch.randn(Nq, H, D, dtype=torch.float32, device=tee)
    K_cpu = torch.randn(Nk, H, D, dtype=torch.float32, device=tee)

    # 分数（明文对照）S = Q @ K^T，按 head 维度独立计算
    # 变换成 [H, Nq, D] 和 [H, Nk, D]
    Qh = Q_cpu.permute(1, 0, 2).contiguous()  # [H, Nq, D]
    Kh = K_cpu.permute(1, 0, 2).contiguous()  # [H, Nk, D]
    S_ref = torch.matmul(Qh, Kh.transpose(-1, -2))  # [H, Nq, Nk]

    # 生成正交矩阵 R（TEE）
    R = orthogonal_matrix(D, dtype=torch.float32, device=tee)

    # TEE 侧加密：Q' = Q R, K' = K R
    Q_enc_cpu = torch.matmul(Q_cpu, R)
    K_enc_cpu = torch.matmul(K_cpu, R)

    # 发送到不可信端（只看到密文 Q'/K'）并计算分数
    Qh_enc = Q_enc_cpu.permute(1, 0, 2).contiguous().to(accel)
    Kh_enc = K_enc_cpu.permute(1, 0, 2).contiguous().to(accel)
    S_enc = torch.matmul(Qh_enc, Kh_enc.transpose(-1, -2)).cpu()  # 回到 CPU 对比

    max_abs_err = (S_ref - S_enc).abs().max().item()

    # 验证 RMS 不变（取一个向量）
    x = Q_cpu[0, 0]  # [D]
    xr = Q_enc_cpu[0, 0]
    rms = torch.sqrt((x.pow(2)).mean()).item()
    rms_r = torch.sqrt((xr.pow(2)).mean()).item()

    _print_tensor("R (TEE, 正交矩阵, 仅展示前几项)", R)
    _print_tensor("Q 明文(TEE)", Q_cpu)
    _print_tensor("Q' 密文(发送到不可信端)", Qh_enc)
    print(f"分数矩阵不变性 max_abs_err={max_abs_err:.3e} -> ", "通过" if max_abs_err < 1e-5 else "未通过")
    print(f"RMS 不变性: rms(x)={rms:.6f}, rms(xR)={rms_r:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cuda", "cpu"], help="不可信加速器设备类型（无 GPU 可用 cpu 模拟）")
    parser.add_argument("--din", type=int, default=8)
    parser.add_argument("--dout", type=int, default=6)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"使用不可信加速器: {device}")
    demo_linear_noise(device, B=args.batch, Din=args.din, Dout=args.dout)
    demo_qk_orthogonal(device)


if __name__ == "__main__":
    main()

