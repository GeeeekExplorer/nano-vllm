from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.distributed.kv_transfer.connector import PDDisaggConnector
from nanovllm.distributed.kv_transfer.config_data import ConnectorMetadata


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.pd_disagg_enabled = config.pd_disagg_enabled
        self.pd_connector: PDDisaggConnector | None = None

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool, ConnectorMetadata | None]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1

            # --- PD 分离: 在 allocate 前查询远端 KV Pool ---
            num_external = self.maybe_query_external_kv(seq)

            self.block_manager.allocate(seq)

            # --- PD 分离: 在 allocate 后更新加载状态 ---
            self.maybe_update_after_alloc(seq, num_external)

            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            meta = self.maybe_build_connector_meta(scheduled_seqs, True)
            return scheduled_seqs, True, meta

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        meta = self.maybe_build_connector_meta(scheduled_seqs, False)
        return scheduled_seqs, False, meta

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                # --- PD 分离: 请求完成时判断是否延迟释放 ---
                self.maybe_handle_request_finished(seq)
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

    # ================================================================
    # PD 分离式推理调度钩子
    # ================================================================

    def maybe_query_external_kv(self, seq: Sequence) -> int:
        """
        在 block 分配前查询远端 KV Pool 中可加载的 token 数。

        对应 vllm-ascend PR #950 中 Scheduler 在 allocate 前调用
        connector.get_num_new_matched_tokens()，判断远端已有多少 KV 可以直接加载
        而不需要重新计算。

        输入:
            seq: Sequence - 待调度的请求序列

        输出:
            int - 可从远端加载的 token 数（0 表示无命中或未启用 PD 分离）

        执行逻辑:
            # if not self.pd_disagg_enabled or self.pd_connector is None:
            #     return 0
            # num_computed = seq.num_cached_tokens  # 本地已缓存的 token 数
            # num_external, is_async = self.pd_connector.get_num_new_matched_tokens(
            #     seq, num_computed
            # )
            # return num_external
            #
            # 这一步的意义:
            #   如果远端命中了 N 个 token，那么 BlockManager.allocate 时可以跳过
            #   这 N 个 token 对应 block 的计算，直接从远端加载 KV Cache。
            #   这避免了 Prefill 节点和 Decode 节点的重复计算。
        """
        return 0

    def maybe_update_after_alloc(self, seq: Sequence, num_external_tokens: int):
        """
        在 block 分配后更新连接器的加载状态。

        对应 vllm-ascend PR #950 中 Scheduler 在 allocate 后调用
        connector.update_state_after_alloc()，确认 block 分配成功后才允许加载。

        输入:
            seq: Sequence - 已分配 block 的请求序列
            num_external_tokens: int - maybe_query_external_kv 返回的外部 token 数

        执行逻辑:
            # if not self.pd_disagg_enabled or self.pd_connector is None:
            #     return
            # if num_external_tokens == 0:
            #     return
            # self.pd_connector.update_state_after_alloc(
            #     seq, seq.block_table, num_external_tokens
            # )
        """
        pass

    def maybe_build_connector_meta(
        self,
        scheduled_seqs: list[Sequence],
        is_prefill: bool,
    ) -> ConnectorMetadata | None:
        """
        构建本轮调度步骤的传输元数据。

        对应 vllm-ascend PR #950 中 Scheduler 在 schedule() 完成后调用
        connector.build_connector_meta()，将所有需要传输的请求信息打包为
        ConnectorMetadata，传递给 ModelRunner/Worker 执行实际传输。

        输入:
            scheduled_seqs: list[Sequence] - 本轮调度的序列列表
            is_prefill: bool - 是否为 prefill 阶段

        输出:
            ConnectorMetadata | None - 传输元数据，None 表示未启用 PD 分离

        执行逻辑:
            # if not self.pd_disagg_enabled or self.pd_connector is None:
            #     return None
            # return self.pd_connector.build_connector_meta(
            #     scheduled_seqs, is_prefill
            # )
        """
        return None

    def maybe_handle_request_finished(self, seq: Sequence):
        """
        请求完成时的 PD 分离处理。

        对应 vllm-ascend PR #950 中请求完成后调用 connector.request_finished()，
        判断是否需要延迟释放 block（因为 KV Cache 可能还在异步保存中）。

        输入:
            seq: Sequence - 已完成的请求序列

        执行逻辑:
            # if not self.pd_disagg_enabled or self.pd_connector is None:
            #     return
            # delay_free, _ = self.pd_connector.request_finished(
            #     seq, seq.block_table
            # )
            # if delay_free:
            #     # 不立即释放 block，等异步保存完成后再释放
            #     # 将 block_ids 加入延迟释放队列
            #     pass
        """
        pass
