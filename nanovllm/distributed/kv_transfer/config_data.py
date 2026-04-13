"""
PD 分离式推理的传输元数据与配置数据结构。

对应 vllm-ascend PR #950 中的 config_data.py，定义了 KV Cache 分布式传输所需的
全部数据结构，包括存储 key、请求元数据、加载规格、请求追踪器和连接器元数据。
"""

from dataclasses import dataclass, field

import torch


# ============================================================================
# 存储 Key 相关
# ============================================================================

@dataclass
class KeyMetadata:
    """
    KV Cache 存储 key 中的元信息部分。

    字段:
        model_name: str - LLM 模型名称（如 "Qwen3-8B"）
        tp_rank: int - 当前 tensor parallel rank
        pp_rank: int - 当前 pipeline parallel rank（nano-vllm 中暂为 0）
    """
    model_name: str = ""
    tp_rank: int = 0
    pp_rank: int = 0


@dataclass
class PoolKey:
    """
    KV Cache 在分布式存储中的唯一标识 key。

    由模型元信息和 block 内容哈希共同决定，确保相同输入在不同节点上
    可以通过相同 key 查找到对方已计算的 KV Cache。

    字段:
        key_metadata: KeyMetadata - 模型和并行度元信息
        chunk_hash: str - 该 block 对应 token 序列的哈希值
    """
    key_metadata: KeyMetadata = field(default_factory=KeyMetadata)
    chunk_hash: str = ""

    def to_string(self) -> str:
        """
        将 PoolKey 序列化为字符串，用作存储系统的 key。

        输出:
            str - 格式为 "{model_name}@tp_rank:{tp_rank}@pp_rank:{pp_rank}@{chunk_hash}"

        执行逻辑:
            # 拼接 key_metadata 各字段和 chunk_hash
            # 返回一个唯一标识字符串
        """
        return (
            f"{self.key_metadata.model_name}"
            f"@tp_rank:{self.key_metadata.tp_rank}"
            f"@pp_rank:{self.key_metadata.pp_rank}"
            f"@{self.chunk_hash}"
        )

    def split_layers(self, num_layers: int) -> list["LayerPoolKey"]:
        """
        将一个 block 级别的 key 拆分为多个 layer 级别的 key（用于逐层传输模式）。

        输入:
            num_layers: int - 模型的层数

        输出:
            list[LayerPoolKey] - 每一层对应一个 LayerPoolKey

        执行逻辑:
            # return [LayerPoolKey(self.key_metadata, self.chunk_hash, layer_id)
            #         for layer_id in range(num_layers)]
        """
        return [
            LayerPoolKey(
                key_metadata=self.key_metadata,
                chunk_hash=self.chunk_hash,
                layer_id=i,
            )
            for i in range(num_layers)
        ]


@dataclass
class LayerPoolKey(PoolKey):
    """
    层级别的 KV Cache 存储 key，继承 PoolKey 并增加 layer_id。

    用于 layerwise（逐层）传输模式，每一层的 KV Cache 有独立的 key。

    字段:
        layer_id: int - 模型中的层编号
    """
    layer_id: int = 0

    def to_string(self) -> str:
        """
        输出:
            str - 格式为 "{model_name}@tp_rank:{tp_rank}@{chunk_hash}@layer:{layer_id}"
        """
        return f"{super().to_string()}@layer:{self.layer_id}"


# ============================================================================
# 加载与请求追踪
# ============================================================================

@dataclass
class LoadSpec:
    """
    描述一个请求从远端 KV Pool 加载 KV Cache 的规格。

    在 Scheduler 端创建，当检测到远端命中后，记录可加载的 token 数量，
    传递给 Worker 端执行实际加载。

    字段:
        vllm_cached_tokens: int - 本地 vLLM 已缓存的 token 数
        kvpool_cached_tokens: int - 远端 KV Pool 中命中的 token 数
        can_load: bool - Scheduler 是否已批准加载（需要 block 分配成功后才为 True）
    """
    vllm_cached_tokens: int = 0
    kvpool_cached_tokens: int = 0
    can_load: bool = False


@dataclass
class RequestTracker:
    """
    追踪一个请求在 PD 分离传输中的状态。

    在 Scheduler 端维护，记录该请求已分配的 block、已保存的 token 数等信息，
    用于构建传输元数据。

    字段:
        req_id: str - 请求 ID
        token_len: int - 当前已调度的 token 总长度
        allocated_block_ids: list[int] - 已分配的 KV Cache block ID 列表
        num_saved_tokens: int - 已成功保存到远端的 token 数
        token_ids: list[int] | None - 已调度的 token ID 序列（用于生成 KV event）
    """
    req_id: str = ""
    token_len: int = 0
    allocated_block_ids: list[int] = field(default_factory=list)
    num_saved_tokens: int = 0
    token_ids: list[int] | None = None

    def update(self, new_block_ids: list[int]):
        """
        当请求在后续调度轮次中获得新的 block 分配时，更新追踪器。

        输入:
            new_block_ids: list[int] - 新分配的 block ID 列表

        执行逻辑:
            # self.allocated_block_ids.extend(new_block_ids)
        """
        self.allocated_block_ids.extend(new_block_ids)


# ============================================================================
# 请求传输元数据
# ============================================================================

@dataclass
class ReqMeta:
    """
    单个请求在一次调度步骤中的传输元数据。

    由 Scheduler 端的 build_connector_meta 构建，传递给 Worker 端，
    Worker 根据此信息决定对哪些 block 执行 save（写入远端）或 load（从远端加载）。

    字段:
        req_id: str - 请求 ID
        token_len_chunk: int - 本次 chunk 中需要保存的 token 数
        block_ids: list[int] - 相关的 KV Cache block ID
        block_hashes: list[str] - 对应 block 的哈希值（用于生成存储 key）
        can_save: bool | None - 是否允许将 KV 保存到远端
        load_spec: LoadSpec | None - 加载规格（为 None 表示不需要加载）
        is_last_chunk: bool | None - 是否为该请求的最后一个 chunk
        current_event: torch.cuda.Event | None - 用于计算-传输同步的 CUDA/NPU event
    """
    req_id: str = ""
    token_len_chunk: int = 0
    block_ids: list[int] = field(default_factory=list)
    block_hashes: list[str] = field(default_factory=list)
    can_save: bool | None = None
    load_spec: LoadSpec | None = None
    is_last_chunk: bool | None = None
    current_event: torch.cuda.Event | None = None

    @staticmethod
    def from_request_tracker(
        tracker: "RequestTracker",
        block_size: int,
        load_spec: "LoadSpec | None" = None,
        skip_save: bool = False,
        block_hashes: list[str] | None = None,
        is_last_chunk: bool | None = None,
    ) -> "ReqMeta | None":
        """
        从 RequestTracker 构建 ReqMeta。

        输入:
            tracker: RequestTracker - 请求追踪器
            block_size: int - KV Cache block 大小（token 数）
            load_spec: LoadSpec | None - 加载规格
            skip_save: bool - 是否跳过保存（如 consumer 角色不需要 save）
            block_hashes: list[str] | None - block 哈希列表
            is_last_chunk: bool | None - 是否最后一个 chunk

        输出:
            ReqMeta | None - 构建的元数据，如果既不需要 save 也不需要 load 则返回 None

        执行逻辑:
            # 1. 计算本次 chunk 需要保存的 token 数（按 block_size 对齐）
            #    num_tokens_to_save = tracker.token_len // block_size * block_size
            # 2. 判断是否需要跳过保存：
            #    - 如果 skip_save 为 True，或 num_tokens_to_save < 上一次保存的边界 → 跳过
            # 3. 如果既跳过保存又没有 load_spec → 返回 None（无需传输）
            # 4. 如果需要保存 → 更新 tracker.num_saved_tokens
            # 5. 如果 load_spec 存在且 can_load → 保留，否则置 None
            # 6. 构建并返回 ReqMeta
        """
        if block_hashes is None:
            block_hashes = []

        num_tokens_to_save = tracker.token_len // block_size * block_size
        skip_save = skip_save or num_tokens_to_save <= tracker.num_saved_tokens

        if skip_save and load_spec is None:
            return None

        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save

        if load_spec is not None and not load_spec.can_load:
            load_spec = None

        return ReqMeta(
            req_id=tracker.req_id,
            token_len_chunk=num_tokens_to_save,
            block_ids=tracker.allocated_block_ids,
            can_save=not skip_save,
            load_spec=load_spec,
            block_hashes=block_hashes,
            is_last_chunk=is_last_chunk,
        )


# ============================================================================
# 连接器元数据（Scheduler → Worker 传递）
# ============================================================================

class ConnectorMetadata:
    """
    一次调度步骤的完整 KV 传输元数据。

    由 Scheduler 端的 build_connector_meta() 构建，通过 bind_connector_metadata()
    传递给 Worker 端，Worker 根据其中的 ReqMeta 列表执行 save/load 操作。

    属性:
        requests: list[ReqMeta] - 本轮需要执行传输的所有请求元数据
        unfinished_request_ids: set[str] - 尚未完成的请求 ID 集合
        preempted_req_ids: set[str] - 被抢占的请求 ID 集合
    """

    def __init__(self, unfinished_request_ids: set[str] | None = None,
                 preempted_req_ids: set[str] | None = None):
        self.requests: list[ReqMeta] = []
        self.unfinished_request_ids = unfinished_request_ids or set()
        self.preempted_req_ids = preempted_req_ids or set()

    def add_request(self, req_meta: ReqMeta):
        """
        添加一个请求的传输元数据。

        输入:
            req_meta: ReqMeta - 请求传输元数据
        """
        self.requests.append(req_meta)
