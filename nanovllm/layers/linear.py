import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from nanovllm.utils.secure import NoisePool, get_security_config
from nanovllm.utils.trace import should_trace, print_tensor, print_line


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator

#实现并行线性层，在分布式环境中高效处理大规模模型的线性变换
#可以生成注意力机制中的QKV矩阵
#在MLP中进行前向传播
class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

#继承这个类，增加一步对x进行加密的操作
class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        enable_mask: bool = True,      # 是否启用加性噪声掩码（x -> x - r）
        mask_scale: float = 0.05,      # 噪声强度
        layer_id: int | None = None,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)

        # 计算输出维度
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        
        # 调用父类初始化（列并行，内部会按 TP 分片输出尺寸）
        super().__init__(hidden_size, output_size, bias)

        # 掩码/噪声配置
        self.enable_mask = enable_mask
        self.mask_scale = mask_scale
        self.hidden_size = hidden_size
        self.layer_id = layer_id

        # 安全配置：是否在 CPU 上执行解密补偿
        sec = get_security_config()
        self.decrypt_on_cpu = sec.decrypt_on_cpu
        pool_size = sec.noise_pool_size
        # 噪声池（输入维度 hidden_size，输出维度为本 rank 的 out_features）
        if self.enable_mask:
            self._noise_pool = NoisePool(
                in_features=hidden_size,
                out_features=self.weight.shape[0],
                pool_size=pool_size,
                noise_scale=self.mask_scale,
                seed=sec.seed,
            )
        else:
            self._noise_pool = None

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)
        # 权重更新后，更新噪声池的 rW 预计算
        if self._noise_pool is not None and param is self.weight:
            # 注意：这里每次装载一段权重后都会更新一次 rW，待全部装载完成后会稳定。
            self._noise_pool.set_weight(self.weight.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 如果未启用掩码，直接返回原始计算
        if not self.enable_mask or self._noise_pool is None:
            return F.linear(x, self.weight, self.bias)

        # 采样噪声 r 及补偿 rW（均为 CPU 张量，形状：[in_features], [out_features]）
        r_cpu, rw_cpu, _ = self._noise_pool.sample()

        # 在 CPU 进行加密（TEE）：x' = x - r
        sec = get_security_config()
        if sec.encrypt_on_cpu:
            x_cpu = x.detach().to(device="cpu", dtype=torch.float32)
            r = r_cpu.to(dtype=x_cpu.dtype)
            if x_cpu.dim() == 2:
                r_b = r.unsqueeze(0)
            else:
                view_shape = [1] * (x_cpu.dim() - 1) + [r.shape[0]]
                r_b = r.view(*view_shape)
            x_masked_cpu = x_cpu - r_b
            # 传送给不安全 GPU 做线性
            x_masked = x_masked_cpu.to(device=self.weight.device, dtype=x.dtype)
        else:
            # 在当前设备直接加密（不安全环境），仅用于测试或性能对比
            r = r_cpu.to(device=x.device, dtype=x.dtype)
            if x.dim() == 2:
                r_b = r.unsqueeze(0)
            else:
                view_shape = [1] * (x.dim() - 1) + [r.shape[0]]
                r_b = r.view(*view_shape)
            x_masked = x - r_b

        # 在 GPU 上计算加密后的线性：y' = (x - r) W^T + b
        y_masked = F.linear(x_masked, self.weight, self.bias)

        # 解密补偿：添加 rW
        if sec.decrypt_on_cpu:
            # 将 y' 回传到 CPU，在 CPU 上做补偿，再返回设备
            y_cpu = y_masked.detach().to(device="cpu", dtype=torch.float32)
            rw = rw_cpu.to(dtype=y_cpu.dtype)
            if y_cpu.dim() == 2:
                y_cpu = y_cpu + rw.unsqueeze(0)
            else:
                view_shape = [1] * (y_cpu.dim() - 1) + [rw.shape[0]]
                y_cpu = y_cpu + rw.view(*view_shape)
            y = y_cpu.to(device=y_masked.device, dtype=y_masked.dtype)
        else:
            # 在 GPU 上完成补偿
            rw = rw_cpu.to(device=y_masked.device, dtype=y_masked.dtype)
            if y_masked.dim() == 2:
                y = y_masked + rw.unsqueeze(0)
            else:
                view_shape = [1] * (y_masked.dim() - 1) + [rw.shape[0]]
                y = y_masked + rw.view(*view_shape)

        # 可视化与正确性对比（仅打印一次）
        from nanovllm.utils.trace import layer_enabled, get_trace_config
        if layer_enabled(self.layer_id) and should_trace(f"QKVParallelLinear:{id(self)}"):
            cfg = get_trace_config()
            print_line(f"[TRACE][QKV][L{self.layer_id}] 线性加密流程")
            try:
                # 参考对比（纯 CPU-fp32 代数等价 + 运行路径）
                w_cpu = self.weight.detach().to(device="cpu", dtype=torch.float32)
                b_cpu = None if self.bias is None else self.bias.detach().to(device="cpu", dtype=torch.float32)
                x_plain_cpu = x.detach().to(device="cpu", dtype=torch.float32)
                y_ref_cpu = F.linear(x_plain_cpu, w_cpu, b_cpu)
                if x_plain_cpu.dim() == 2:
                    r_b_cpu = r_cpu.to(dtype=x_plain_cpu.dtype).unsqueeze(0)
                else:
                    view_shape = [1] * (x_plain_cpu.dim() - 1) + [r_cpu.numel()]
                    r_b_cpu = r_cpu.to(dtype=x_plain_cpu.dtype).view(*view_shape)
                y_rec_cpu_true = F.linear(x_plain_cpu - r_b_cpu, w_cpu, b_cpu) + rw_cpu.to(dtype=torch.float32).unsqueeze(0)
                alg_abs_err = (y_ref_cpu - y_rec_cpu_true).abs().max().item()
                y_rec_cpu = y.detach().to(device="cpu", dtype=torch.float32)
                run_abs_err = (y_ref_cpu - y_rec_cpu).abs().max().item()

                # 同时给出相对误差，便于不同尺度比较
                denom = y_ref_cpu.abs().max().item() + 1e-6
                alg_rel = alg_abs_err / denom
                run_rel = run_abs_err / denom

                # 简洁输出
                if cfg.summary_only:
                    # 阈值策略：代数等价应近似 0；运行路径相对误差在 bf16 下放宽到 1e-1
                    pass_alg = alg_abs_err <= 1e-6 or alg_rel <= 1e-6
                    pass_run = run_rel <= 1e-1
                    print_line(f"[QKV][alg] PASS={pass_alg} abs={alg_abs_err:.2e} rel={alg_rel:.2e}")
                    print_line(f"[QKV][run] PASS={pass_run} abs={run_abs_err:.2e} rel={run_rel:.2e}")
                else:
                    print_line(f"x: {tuple(x.shape)} -> x' {tuple(x_masked.shape)} | W {tuple(self.weight.shape)} | y' {tuple(y_masked.shape)} -> y {tuple(y.shape)}")
                    print_line(f"r: {tuple(r_cpu.shape)} | rW: {tuple(rw_cpu.shape)} (CPU)")
                    print_line(f"equiv_err(alg) abs={alg_abs_err:.2e} rel={alg_rel:.2e} | equiv_err(run) abs={run_abs_err:.2e} rel={run_rel:.2e}")
            except Exception as e:
                print_line(f"trace 对比失败: {e}")

        return y
    
class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
