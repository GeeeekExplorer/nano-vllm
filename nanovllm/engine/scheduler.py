from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler: # 调度器，职责是“每次 step 给 engine 返回一批 seqs，告诉这是 prefill 还是 decode”。

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs # 512, 一个 batch 最多容纳多少条序列（decode 时就是 batch size 上限）
        self.max_num_batched_tokens = config.max_num_batched_tokens # 16384, prefill 时“总 token 预算”上限（因为 prefill 计算量与 token 数强相关）。
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size) # KV cache 的“物理块池”管理器：分配、释放、追加 block、prefix cache 复用
        # 相比V0少了个swapping队列
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        # todo：应该是seq==waiting再append，防止seq太长
        self.waiting.append(seq)

    # todo：头阻塞（HOL blocking）会出现吗？
    #   - yw：感觉资源分配应该计算过，不可能让一个seq num打爆
    #   - yw：优先prefill，prefill和decode分离
    def schedule(self) -> tuple[list[Sequence], bool]: # 本轮要跑的序列列表；本轮是 prefill（True）还是 decode（False）
        # prefill
        scheduled_seqs = []
        num_seqs = 0 # 本轮已经选了多少条 seq
        num_batched_tokens = 0 # prefill token 预算计数器
        while self.waiting and num_seqs < self.max_num_seqs: # 先尝试进行prefill
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq): # 如果物理 KV block 不够分配这条 seq 的所有 block，停止扩 batch。
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq) # 正在推理的seq在队尾
            scheduled_seqs.append(seq) # 加入本轮 batch
        # todo：和v0的scheduler对比，我怎么记得v0是先做decode（或隔一段时间）？
        if scheduled_seqs: # 只要本轮拉进来任何 waiting 请求，就直接返回 prefill batch
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            # todo：是上次后seq.len + 1了吗？
            while not self.block_manager.can_append(seq): # 如果这次append会跨进新 block（ len(seq)%block_size==1 ），就需要至少1个空闲block；否则不需要
                if self.running:
                    self.preempt(self.running.pop()) # 也不是完全的LRU，waiting就是放在尾部
                else:
                    # todo: free_block是和running中的绑定的吗？
                    # decode每生成 1 token就要往KV cache里写入一条新K/V，该seq decode生成的token太长
                    self.preempt(seq) # 如果已经没有别的 running，只能抢占当前 seq 自己，然后 break
                    break
            # todo：while结束之后到这吗？break会到这吗
            else:
                num_seqs += 1
                self.block_manager.may_append(seq) # 如果跨进新 block，就分配一个新物理 block；如果刚好填满 block，就计算 hash 写入 prefix cache map
                scheduled_seqs.append(seq)
        assert scheduled_seqs # 保证 decode 分支一定能安排到至少一条，（否则 engine 的 step 无法推进）
        # me：感觉是FCFS先来先服务
        # - deque.extendleft(iterable) 的语义是： 把 iterable 逐个元素从左侧插入 。这会导致插入顺序被翻转。
        # - reversed 是为了抵消 extendleft 自带的反转效果，保持队列顺序稳定。
        self.running.extendleft(reversed(scheduled_seqs)) # 把本轮挑出来 decode 的 seq 再放回 running 的队首，保持它们“仍然在运行中”，并维持一个近似的轮转顺序。
        return scheduled_seqs, False

    # 走recompute
    def preempt(self, seq: Sequence):
        # ！！！只是释放KV cache，但 seq 的 token_ids 不会减少 （已经生成的 token 还在），所以它的“逻辑长度”不会因为 preempt 变短。
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq) # preempt的seq在waiting队头

    # 一轮LLMEngine.step后的后处理
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        # todo：prefill返回的token_id是啥
        for seq, token_id in zip(seqs, token_ids): # token_id是最后一个token
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq) # 从 running 队列移除（注意这里是线性查找，O(n)，但文件很短，属于简化）
