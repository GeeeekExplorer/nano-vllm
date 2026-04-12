"""CPU Offload 元数据定义模块。

定义 KV Cache 在 GPU 与 CPU 之间卸载时所需的元数据结构。
对应 vllm-ascend PR #1659 中的 metadata.py。
"""

from dataclasses import dataclass, field


@dataclass
class OffloadMetadata:
    """单次 KV Cache 卸载/加载操作的元数据。

    Attributes:
        seq_id:       请求序列的唯一标识
        layer_name:   模型层名称，如 "layers.0.self_attn"
        block_ids:    需要卸载/加载的物理 block ID 列表
        token_length: 该序列当前的 token 总长度
        is_prefill:   当前是否处于 prefill 阶段
    """
    seq_id: int = -1
    layer_name: str = ""
    block_ids: list[int] = field(default_factory=list)
    token_length: int = 0
    is_prefill: bool = False


@dataclass
class SwapRequest:
    """描述一次 GPU ↔ CPU 交换请求。

    Attributes:
        metadata:  本次交换对应的元数据
        direction: 交换方向，"swap_out" 表示 GPU→CPU，"swap_in" 表示 CPU→GPU
        priority:  优先级，数值越小优先级越高
    """
    metadata: OffloadMetadata = field(default_factory=OffloadMetadata)
    direction: str = "swap_out"
    priority: int = 0


def build_offload_metadata(
    seq_id: int,
    layer_name: str,
    block_ids: list[int],
    token_length: int,
    is_prefill: bool,
) -> OffloadMetadata:
    """根据当前推理状态构建卸载元数据。

    Args:
        seq_id:       请求序列 ID
        layer_name:   模型层名称
        block_ids:    涉及的物理 block ID 列表
        token_length: 序列当前 token 总长度
        is_prefill:   是否处于 prefill 阶段

    Returns:
        OffloadMetadata: 填充完毕的元数据对象
    """
    # --- 执行逻辑 ---
    # 1. 将传入参数直接封装为 OffloadMetadata 数据类
    # 2. 返回该元数据实例，供 Connector 和 Manager 在后续
    #    swap_out / swap_in 操作中使用
    return OffloadMetadata(
        seq_id=seq_id,
        layer_name=layer_name,
        block_ids=block_ids,
        token_length=token_length,
        is_prefill=is_prefill,
    )
