from collections import deque # 引入了双端队列 deque 用于管理序列队列

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager

"""
主要负责决定哪些序列应该进入“预填充（Prefill）”阶段，哪些应该进入“解码（Decode）”阶段，并管理 KV Cache 的显存分配与抢占
"""
class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs # 最大并发序列数
        self.max_num_batched_tokens = config.max_num_batched_tokens # 最大批处理 token 数
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size) # 初始化块管理器，负责管理 KV Cache 的物理块
        self.waiting: deque[Sequence] = deque() # 将序列分层等待和运行两部分
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence): # 将新请求的序列加入 waiting 队列尾部
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # 先看能不能prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq): # 循环检查队首序列，如果加入该序列会导致超过 max_num_batched_tokens 或者 KV Cache 块不够 (can_allocate 返回 False)，则停止添加
                break
            num_seqs += 1
            self.block_manager.allocate(seq) # 为这个seq分配block
            num_batched_tokens += len(seq) - seq.num_cached_tokens # 需计算的是增量部分（即那些没有命中缓存、真正需要消耗算力的 Token）
            seq.status = SequenceStatus.RUNNING # 将序列状态改为 RUNNING，并从 waiting 移到 running
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True # 返回 True 表示这是 Prefill 阶段，如果有序列被调度进行 Prefill，直接返回，不会在同一轮中混合进行 Decode

        # decode：如果不需要 Prefill，则进入 Decode 阶段
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq): # 检查是否能为生成的新 token 分配空间
                if self.running:
                    self.preempt(self.running.pop()) # 显存不足，需要抢占
                else:
                    self.preempt(seq) # 如果只剩当前这一个序列还不够内存，只能抢占自己
                    break
            else: # 如果显存足够或者通过抢占腾出了空间
                num_seqs += 1
                self.block_manager.may_append(seq) # 预留空间
                scheduled_seqs.append(seq)
        assert scheduled_seqs # 断言列表是否为非空
        self.running.extendleft(reversed(scheduled_seqs)) # 将本轮调度的序列放回 running 队列头部，保持顺序，如果队列很长，排在前面的序列会一直占据调度权，直到它们结束或被抢占，后面的序列才能得到执行机会
        return scheduled_seqs, False # 返回 False 表示这是 Decode 阶段

    """
    抢占机制：
    如果空间不足，必须暂停某些序列以释放显存。
    优先抢占 running 队列末尾的序列（牺牲别人保全当前序列）。
    如果队列空了还不够，只能抢占当前序列自己。
    """
    def preempt(self, seq: Sequence): # 将序列的 KV Cache 全部释放（意味着下次调度该序列时需要重新计算/Prefill，或者如果有 Swap 机制则换出，这里代码简化为直接释放）
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]: # 模型推理完一步后调用，其中两个list是代表每个seq decode 了每个token
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id) # 将新生成的 token 加入序列
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens: # 如果序列生成了结束符或达到长度限制，标记为 FINISHED 并释放资源
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
