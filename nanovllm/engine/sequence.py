from copy import copy # 用于浅拷贝列表，防止外部修改影响内部状态
from enum import Enum, auto # 用来定义序列的状态
from itertools import count # 一个无限迭代器，用于生成唯一的序列 ID

from nanovllm.sampling_params import SamplingParams

"""
框架中block的key体系:

1. seq内块 ID
范围：`0` 到 `seq_len // block_size`
作用域：Sequence 内部
含义：从 Sequence 的视角看，“第 0 块数据”、“第 1 块数据”...是连续的
对应代码：`Sequence.block_table` 的列表下标

2. 逻辑块 ID
作用域：全局 (BlockManager)
含义：即blockid，程序初始化时分配物理地址并固定，所以也和物理地址差不多意思
"""

class SequenceStatus(Enum):  # 序列的状态
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

"""
管理单个请求的生成，封装单个请求给scheduler和modelrunner
"""
class Sequence:
    block_size = 256 # 块大小，与 PagedAttention 对应，其实是因为flashattn目前支持的块大小为256
    counter = count() # 每创建一个新的 Sequence 实例，它就会递增，用来生成唯一的 seq_id

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids) # 保存输入的 token ID List，使用 copy 避免引用问题
        self.last_token = token_ids[-1] # 记录最后一个 token，用于后续生成时的输入
        self.num_tokens = len(self.token_ids) # 当前序列的总长度
        self.num_prompt_tokens = len(token_ids) # Prompt的长度，固定的，用于分开prompt和输出token id list
        self.num_cached_tokens = 0 # 已缓存 KV 的 token 数量，用于decode
        self.block_table = [] # 物理块索引表，用于映射逻辑块到物理内存块
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self): # len(sequence) 获取当前序列长度
        return self.num_tokens

    def __getitem__(self, key): # sequence[i] 访问特定的 token
        return self.token_ids[key]

    @property # 让你能够用属性访问的语法来调用一个方法，同时隐藏了方法内部的逻辑，多用于封装
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self): # 计算已生成的 token 数量
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self): # 获取 Prompt 部分的 token ID List
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self): # 获取生成部分的 token ID List
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self): # 计算已完全缓存的页数量，不包括未满的
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self): # 计算当前序列需要的逻辑块总数
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self): # 计算最后一个块中实际包含的 token 数量
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size] # 切片获取第 i 个逻辑块中的所有 token ID List，如果切片的结束索引超过了列表的实际长度，Python 不会报错，而是会自动截断到列表末尾

    def append_token(self, token_id: int): # 添加新 Token，在sampler后面
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    """
    多进程间传输seq时的一个有损传输优化：
    根据序列当前的状态（Prefill 还是 Decode），智能地决定只传输必要的数据，而不是整个对象的所有数据
    """
    def __getstate__(self): # 当系统需要把 Sequence 对象发送给另一个进程时调用
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state): # 当另一个进程接收到数据并重建 Sequence 对象时调用
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1] # Decode 阶段：只恢复 last_token
