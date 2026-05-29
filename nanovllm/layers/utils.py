import torch 
def shard_slice(loaded_weight: torch.Tensor, dim: int, start: int, size: int) -> torch.Tensor:
    """只读取 [start:start+size] 这一片。

    loaded_weight 可以是完整的 torch.Tensor，也可以是 safetensors 的 slice 句柄
    (safe_open(...).get_slice(name))。后者只会从磁盘读出需要的这一片，从而避免每个
    rank 都把整张权重读出来再丢弃 1-1/tp，多卡加载会显著加速。
    """
    if dim == 0:
        return loaded_weight[start:start + size]
    if dim == 1:
        return loaded_weight[:, start:start + size]
    raise ValueError(f"unsupported shard dim: {dim}")


def materialize_full(loaded_weight):
    """把完整权重读出来（用于不切分、需整张拷贝的参数）。对 Tensor 和 slice 句柄都适用。"""
    return loaded_weight[:]

