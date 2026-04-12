"""Attention 层与 CPU Offload Connector 的集成工具函数。

提供在 Attention forward 前后调用的钩子函数，用于：
- prefill 前：从 CPU 换入（swap in）该层 KV Cache
- prefill 后：将该层 KV Cache 卸载（swap out）到 CPU

支持两种 KV Cache 格式：
- 标准 MHA：传入 (k_cache, v_cache) 两个张量
- MLA 压缩格式：传入 (kv_c_cache, k_pe_cache) 两个张量
两者接口统一，通过 *kv_cache_tensors 可变参数适配。

昇腾 910C 上，检测到 NPU device 时自动选择 scatter+burst 传输路径。

对应 vllm-ascend PR #1659 中从 attention_v1.py 提取到 utils.py 的两个函数。
"""

import torch

# 全局 connector 引用，由 ModelRunner 初始化时设置
_CONNECTOR = None

# 是否使用昇腾 scatter+burst 路径（由 _maybe_init 时根据 device 类型设置）
_USE_ASCEND_BURST = False


def set_connector(connector, use_ascend_burst: bool = False) -> None:
    """设置全局 CPU Offload Connector 实例。

    Args:
        connector:        CPUOffloadConnector 实例，或 None 表示不启用 offload
        use_ascend_burst: 是否启用昇腾 scatter+burst 传输路径
    """
    global _CONNECTOR, _USE_ASCEND_BURST
    _CONNECTOR = connector
    _USE_ASCEND_BURST = use_ascend_burst


def get_connector():
    """获取全局 CPU Offload Connector 实例。

    Returns:
        CPUOffloadConnector | None
    """
    return _CONNECTOR


def wait_for_kv_layer_from_connector(
    layer_name: str,
    *kv_cache_tensors: torch.Tensor,
) -> None:
    """在 Attention forward 之前调用，等待从 CPU 换入该层的 KV Cache。

    若未启用 CPU Offload（connector 为 None），则直接返回不做任何操作。

    支持两种调用方式：
      - 标准 MHA：wait_for_kv_layer_from_connector(name, k_cache, v_cache)
      - MLA 格式：wait_for_kv_layer_from_connector(name, kv_c_cache, k_pe_cache)

    Args:
        layer_name:        当前层名称，如 "layers.0.self_attn"
        *kv_cache_tensors: GPU/NPU 端该层的 KV cache 张量（2 个）

    Returns:
        None（原地写入 kv_cache_tensors）
    """
    # --- 执行逻辑 ---
    # 1. 检查全局 _CONNECTOR 是否为 None，若是则直接 return
    # 2. 从当前推理上下文（get_context()）中获取 seq_id、block_ids 等信息
    # 3. 检查该层对应的 block 是否已被 swap out 到 CPU
    # 4. 根据 _USE_ASCEND_BURST 选择传输路径：
    #    a. 昇腾路径：调用 connector.ascend_swap_in_with_burst_gather(
    #           layer_name, seq_id, kv_cache_tensors[0], kv_cache_tensors[1],
    #           block_ids, token_length, is_prefill)
    #       → burst DMA 传输 + AI Core gather 分发
    #    b. 通用路径：调用 connector.recv_kv_caches_and_hidden_states(
    #           layer_name, seq_id, kv_cache_tensors[0], kv_cache_tensors[1],
    #           block_ids, token_length, is_prefill)
    #       → 逐 block CUDA memcpy
    # 5. 等待传输完成（stream 同步），确保后续算子读到有效数据
    pass


def maybe_save_kv_layer_to_connector(
    layer_name: str,
    *kv_cache_tensors: torch.Tensor,
) -> None:
    """在 prefill Attention forward 之后调用，尝试将该层 KV Cache 卸载到 CPU。

    仅在 prefill 阶段且启用了 CPU Offload 时生效。

    支持两种调用方式：
      - 标准 MHA：maybe_save_kv_layer_to_connector(name, k_cache, v_cache)
      - MLA 格式：maybe_save_kv_layer_to_connector(name, kv_c_cache, k_pe_cache)

    Args:
        layer_name:        当前层名称
        *kv_cache_tensors: GPU/NPU 端该层的 KV cache 张量（2 个）

    Returns:
        None
    """
    # --- 执行逻辑 ---
    # 1. 检查全局 _CONNECTOR 是否为 None，若是则直接 return
    # 2. 从当前推理上下文中获取 seq_id、block_ids、token_length、is_prefill
    # 3. 仅在 is_prefill == True 时执行卸载
    # 4. 根据 _USE_ASCEND_BURST 选择传输路径：
    #    a. 昇腾路径：调用 connector.ascend_swap_out_with_scatter_burst(
    #           layer_name, seq_id, kv_cache_tensors[0], kv_cache_tensors[1],
    #           block_ids, token_length, is_prefill)
    #       → AI Core scatter 打包 + burst DMA 传输
    #       → 对 MLA 格式尤其高效：kv_c + k_pe 数据量远小于完整 K+V
    #    b. 通用路径：调用 connector.send_kv_caches_and_hidden_states(
    #           layer_name, seq_id, kv_cache_tensors[0], kv_cache_tensors[1],
    #           block_ids, token_length, is_prefill)
    # 5. 异步执行不等待——减少对推理延迟的影响
    pass
