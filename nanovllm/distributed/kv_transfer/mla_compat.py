"""
难点四: MLA（Multi-head Latent Attention）模型兼容性。

问题:
    DeepSeek 等模型使用 MLA 注意力机制，其 KV Cache 格式与标准 MHA 截然不同:
    - 标准 MHA: K=[num_blocks, block_size, num_kv_heads, head_dim],
                V=[num_blocks, block_size, num_kv_heads, head_dim]（K 和 V 独立存储）
    - MLA: latent=[num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
           （K 和 V 共享一个低秩压缩表示）

    这导致:
    1. block 字节大小计算公式不同
    2. 存储 key 生成方式不同（单 latent 而非 K+V 两份）
    3. 传输线程中计算设备地址和大小的逻辑不同
    4. 条件判断从 torchair_graph_enabled 改为 is_deepseek_mla（更精确）

解决方案:
    1. detect_mla_model(): 检测模型是否使用 MLA 注意力
    2. compute_mla_kv_cache_shape(): 计算 MLA 模型的 KV Cache 形状
    3. compute_mla_block_bytes(): 计算 MLA block 的字节大小（考虑 4MB 对齐）
    4. build_mla_kv_cache_dict(): 构建 MLA 模式下的 KV Cache 注册字典
    5. compute_mla_block_addr(): 计算 MLA block 在设备内存中的地址
    6. choose_decode_attention_op(): 选择合适的 decode 注意力算子
"""

import torch


def detect_mla_model(hf_config) -> bool:
    """
    检测模型是否使用 MLA（Multi-head Latent Attention）。

    输入:
        hf_config: AutoConfig - HuggingFace 模型配置

    输出:
        bool - True 如果模型使用 MLA

    执行逻辑:
        # MLA 模型的特征:
        # 1. 配置中包含 kv_lora_rank 字段（KV 的低秩维度）
        # 2. 配置中包含 qk_rope_head_dim 字段（旋转位置编码的 head 维度）
        # 3. 或直接检查 model_type == "deepseek_v2" / "deepseek_v3"
        #
        # 注意:
        #   PR #950 中将条件从 torchair_graph_enabled 改为 is_deepseek_mla，
        #   因为后来其他模型（如盘古）也支持了 TorchAIR，而 MLA 的 KV Cache 格式
        #   是 DeepSeek 特有的，需要更精确的条件判断。
        #
        # return (
        #     hasattr(hf_config, "kv_lora_rank")
        #     and hasattr(hf_config, "qk_rope_head_dim")
        #     and hf_config.kv_lora_rank > 0
        # )
    """
    return (
        hasattr(hf_config, "kv_lora_rank")
        and hasattr(hf_config, "qk_rope_head_dim")
        and getattr(hf_config, "kv_lora_rank", 0) > 0
    )


def get_mla_latent_dim(hf_config) -> int:
    """
    获取 MLA 模型的 latent 维度（= kv_lora_rank + qk_rope_head_dim）。

    输入:
        hf_config: AutoConfig - HuggingFace 模型配置

    输出:
        int - MLA latent 维度

    执行逻辑:
        # MLA 的压缩 KV 表示维度 = kv_lora_rank + qk_rope_head_dim
        # 例如 DeepSeek-V2: kv_lora_rank=512, qk_rope_head_dim=64 → latent_dim=576
        #
        # return hf_config.kv_lora_rank + hf_config.qk_rope_head_dim
    """
    return getattr(hf_config, "kv_lora_rank", 0) + getattr(hf_config, "qk_rope_head_dim", 0)


def compute_mla_kv_cache_shape(
    num_blocks: int,
    block_size: int,
    hf_config,
) -> tuple[int, ...]:
    """
    计算 MLA 模型的单层 KV Cache tensor 形状。

    输入:
        num_blocks: int - block 总数
        block_size: int - 每 block 的 token 数
        hf_config: AutoConfig - 模型配置

    输出:
        tuple[int, ...] - KV Cache tensor 形状

    执行逻辑:
        # 标准 MHA 的 KV Cache:
        #   K: [num_blocks, block_size, num_kv_heads, head_dim]
        #   V: [num_blocks, block_size, num_kv_heads, head_dim]
        #   总共需要 2 份（K 和 V 分开存储）
        #
        # MLA 的 KV Cache:
        #   latent: [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
        #   只需要 1 份（K 和 V 共享压缩表示）
        #   在推理时，从 latent 通过上投影恢复出 K 和 V
        #
        # latent_dim = hf_config.kv_lora_rank + hf_config.qk_rope_head_dim
        # return (num_blocks, block_size, latent_dim)
    """
    latent_dim = get_mla_latent_dim(hf_config)
    return (num_blocks, block_size, latent_dim)


def compute_mla_block_bytes(
    block_size: int,
    hf_config,
    dtype: torch.dtype,
    apply_alignment: bool = True,
) -> int:
    """
    计算 MLA 模型单个 KV block 的字节大小。

    输入:
        block_size: int - 每 block 的 token 数
        hf_config: AutoConfig - 模型配置
        dtype: torch.dtype - 数据类型
        apply_alignment: bool - 是否应用 4MB 对齐

    输出:
        int - 单个 block 的字节大小

    执行逻辑:
        # MLA 模型:
        #   latent_dim = kv_lora_rank + qk_rope_head_dim
        #   raw_bytes = block_size * latent_dim * dtype.itemsize
        #   # MLA 只需要 1 份（共享表示），而 MHA 需要 2 份（K + V）
        #
        # 标准 MHA 模型:
        #   raw_bytes = 2 * block_size * num_kv_heads * head_dim * dtype.itemsize
        #   # 2 份 = K + V
        #
        # 对齐:
        #   if apply_alignment:
        #       return align_to_4mb(raw_bytes)
        #   return raw_bytes
    """
    latent_dim = get_mla_latent_dim(hf_config)
    raw_bytes = block_size * latent_dim * dtype.itemsize
    if apply_alignment:
        from nanovllm.distributed.kv_transfer.memory_alignment import align_to_4mb
        return align_to_4mb(raw_bytes)
    return raw_bytes


def build_mla_kv_cache_dict(
    latent_caches: list[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    构建 MLA 模式下供 KVPoolWorker.register_kv_caches() 使用的字典。

    与标准 MHA 不同，MLA 每层只有一个 latent cache（而非 K + V 两个）。

    输入:
        latent_caches: list[torch.Tensor] - 每层的 latent cache tensor
            shape: [num_blocks, block_size, latent_dim]

    输出:
        dict[str, torch.Tensor] - key 为 "layer.{i}.latent"

    执行逻辑:
        # kv_dict = {}
        # for i, cache in enumerate(latent_caches):
        #     kv_dict[f"layer.{i}.latent"] = cache
        # return kv_dict
        #
        # 传输线程在处理 MLA 字典时:
        #   - 每个 block 只需传输 1 份数据（latent），而非 MHA 的 2 份（K + V）
        #   - block 地址计算: base_ptr + block_id * aligned_latent_block_bytes
    """
    kv_dict = {}
    for i, cache in enumerate(latent_caches):
        kv_dict[f"layer.{i}.latent"] = cache
    return kv_dict


def compute_mla_block_addr(
    latent_cache: torch.Tensor,
    block_id: int,
    block_size: int,
    latent_dim: int,
    itemsize: int,
    apply_alignment: bool = True,
) -> tuple[int, int]:
    """
    计算 MLA 模型中单个 block 在设备内存中的地址和大小。

    输入:
        latent_cache: torch.Tensor - 该层的 latent cache tensor
        block_id: int - block 编号
        block_size: int - 每 block 的 token 数
        latent_dim: int - latent 维度
        itemsize: int - 元素字节大小
        apply_alignment: bool - 是否使用对齐后的偏移

    输出:
        tuple[int, int] - (设备内存地址, block 字节大小)

    执行逻辑:
        # raw_block_bytes = block_size * latent_dim * itemsize
        #
        # if apply_alignment:
        #     aligned_bytes = align_to_4mb(raw_block_bytes)
        # else:
        #     aligned_bytes = raw_block_bytes
        #
        # addr = latent_cache.data_ptr() + block_id * aligned_bytes
        # return addr, raw_block_bytes
        #
        # 注意: 地址偏移使用 aligned_bytes（对齐后），
        #       但实际数据大小使用 raw_block_bytes（不含 padding）
    """
    return 0, 0


def choose_decode_attention_op(
    is_mla: bool,
    mla_pa_enabled: bool = False,
) -> str:
    """
    根据模型类型选择合适的 decode 阶段注意力算子。

    输入:
        is_mla: bool - 是否为 MLA 模型
        mla_pa_enabled: bool - 是否启用 MLA Paged Attention
            （通过环境变量 VLLM_ASCEND_MLA_PA 控制）

    输出:
        str - 算子名称

    执行逻辑:
        # if is_mla:
        #     if mla_pa_enabled:
        #         # 使用昇腾专用的 MLA 分页注意力算子
        #         # 该算子直接在 latent 空间进行注意力计算，无需先上投影
        #         return "npu_multi_head_latent_attention"
        #     else:
        #         # 使用通用的注意力算子（先从 latent 上投影出 K/V，再计算注意力）
        #         return "flash_attn_with_kvcache_mla_unproject"
        # else:
        #     # 标准 MHA 模型使用 flash attention
        #     return "flash_attn_with_kvcache"
        #
        # 注意:
        #   npu_multi_head_latent_attention 在公开版 torch_npu 中可能不可用，
        #   因此通过 VLLM_ASCEND_MLA_PA 环境变量控制是否启用。
        #   PR #950 中的相关 commit: "enable mla_pa for deepseek mla decode"
    """
    if is_mla:
        if mla_pa_enabled:
            return "npu_multi_head_latent_attention"
        else:
            return "flash_attn_with_kvcache_mla_unproject"
    else:
        return "flash_attn_with_kvcache"


def adapt_kv_transfer_for_mla(
    req_meta,
    kv_caches: dict[str, torch.Tensor],
    block_size: int,
    is_mla: bool,
) -> tuple[list[str], list[list[int]], list[list[int]]]:
    """
    根据模型类型适配 KV 传输线程的 key/addr/size 生成逻辑。

    输入:
        req_meta: ReqMeta - 请求传输元数据
        kv_caches: dict[str, torch.Tensor] - 已注册的 KV Cache 字典
        block_size: int - block 大小
        is_mla: bool - 是否为 MLA 模型

    输出:
        tuple[list[str], list[list[int]], list[list[int]]]:
            - keys: 存储 key 列表
            - addrs: 设备地址列表
            - sizes: 字节大小列表

    执行逻辑:
        # if is_mla:
        #     # MLA 模式: 每个 block 只有一个 latent cache
        #     # key 格式: "{model}@tp:{rank}@{hash}"（无 .k/.v 后缀）
        #     # 地址: 从 "layer.{i}.latent" tensor 中计算
        #     # 大小: block_size * latent_dim * itemsize
        #     for block_id, block_hash in zip(req_meta.block_ids, req_meta.block_hashes):
        #         layer_addrs = []
        #         layer_sizes = []
        #         for layer_name, cache in kv_caches.items():
        #             if "latent" not in layer_name:
        #                 continue
        #             addr, size = compute_mla_block_addr(cache, block_id, ...)
        #             layer_addrs.append(addr)
        #             layer_sizes.append(size)
        #         keys.append(block_hash_to_key(block_hash))
        #         addrs.append(layer_addrs)
        #         sizes.append(layer_sizes)
        #
        # else:
        #     # 标准 MHA 模式: 每个 block 有 K 和 V 两份
        #     # key 格式同上
        #     # 地址: 从 "layer.{i}.k" 和 "layer.{i}.v" tensor 中分别计算
        #     # 大小: block_size * num_kv_heads * head_dim * itemsize（K 和 V 各一份）
        #     for block_id, block_hash in zip(req_meta.block_ids, req_meta.block_hashes):
        #         layer_addrs = []
        #         layer_sizes = []
        #         for layer_name, cache in kv_caches.items():
        #             addr = cache.data_ptr() + block_id * block_bytes
        #             layer_addrs.append(addr)
        #             layer_sizes.append(block_bytes)
        #         keys.append(block_hash_to_key(block_hash))
        #         addrs.append(layer_addrs)
        #         sizes.append(layer_sizes)
    """
    return [], [], []
