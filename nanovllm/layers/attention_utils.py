"""Attention 层与 CPU Offload Connector 的集成工具函数。

提供在 Attention forward 前后调用的钩子函数，用于：
- prefill 前：从 CPU 换入（swap in）该层 KV Cache
- prefill 后：将该层 KV Cache 卸载（swap out）到 CPU

对应 vllm-ascend PR #1659 中从 attention_v1.py 提取到 utils.py 的两个函数。
"""

import torch

# 全局 connector 引用，由 ModelRunner 初始化时设置
_CONNECTOR = None


def set_connector(connector) -> None:
    """设置全局 CPU Offload Connector 实例。

    Args:
        connector: CPUOffloadConnector 实例，或 None 表示不启用 offload
    """
    global _CONNECTOR
    _CONNECTOR = connector


def get_connector():
    """获取全局 CPU Offload Connector 实例。

    Returns:
        CPUOffloadConnector | None
    """
    return _CONNECTOR


def wait_for_kv_layer_from_connector(
    layer_name: str,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> None:
    """在 Attention forward 之前调用，等待从 CPU 换入该层的 KV Cache。

    若未启用 CPU Offload（connector 为 None），则直接返回不做任何操作。

    Args:
        layer_name: 当前层名称，如 "layers.0.self_attn"
        k_cache:    GPU 端该层的 K cache 张量
        v_cache:    GPU 端该层的 V cache 张量

    Returns:
        None（原地写入 k_cache / v_cache）
    """
    # --- 执行逻辑 ---
    # 1. 检查全局 _CONNECTOR 是否为 None，若是则直接 return
    # 2. 从当前推理上下文（get_context()）中获取 seq_id、block_ids 等信息
    # 3. 检查该层对应的 block 是否已被 swap out 到 CPU
    # 4. 若是，调用 connector.recv_kv_caches_and_hidden_states(
    #        layer_name, seq_id, k_cache, v_cache, block_ids, token_length, is_prefill)
    #    将 CPU 上缓存的 KV 数据拷贝回 GPU 端的 k_cache / v_cache
    # 5. 等待异步拷贝完成（CUDA stream 同步）
    pass


def maybe_save_kv_layer_to_connector(
    layer_name: str,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> None:
    """在 prefill Attention forward 之后调用，尝试将该层 KV Cache 卸载到 CPU。

    仅在 prefill 阶段且启用了 CPU Offload 时生效。

    Args:
        layer_name: 当前层名称
        k_cache:    GPU 端该层的 K cache 张量
        v_cache:    GPU 端该层的 V cache 张量

    Returns:
        None
    """
    # --- 执行逻辑 ---
    # 1. 检查全局 _CONNECTOR 是否为 None，若是则直接 return
    # 2. 从当前推理上下文中获取 seq_id、block_ids、token_length、is_prefill
    # 3. 仅在 is_prefill == True 时执行卸载
    # 4. 调用 connector.send_kv_caches_and_hidden_states(
    #        layer_name, seq_id, k_cache, v_cache, block_ids, token_length, is_prefill)
    #    通过异步拷贝将 GPU 上的 KV Cache 数据写入 CPU pin_memory 区域
    # 5. 无需等待完成——异步执行以减少对推理延迟的影响
    pass
