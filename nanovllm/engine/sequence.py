# 推理调度系统的核心数据结构 Sequence类
from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams

# 定义序列的状态（等待、运行、完成）
class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()
    # 传入初始token序列（即prompt）和控制生成行为的参数
    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter) # 设置当前序列的id号
        self.status = SequenceStatus.WAITING # 设置序列初始状态为WAITING
        self.token_ids = copy(token_ids)     # 当前token序列的内容list[]
        self.last_token = token_ids[-1]      # 最近生成的token，初始化为prompt的最后一个token
        self.num_tokens = len(self.token_ids) # 当前该序列的token数（prompt+生成的token）
        self.num_prompt_tokens = len(token_ids) # prompt的token数，即长度
        self.num_cached_tokens = 0              # 已经缓存到kv cache的token数
        self.block_table = []                   # 存储该序列占用的kvcache block索引（由block manager管理）
        # 控制采样与停止条件
        self.temperature = sampling_params.temperature 
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    # 返回当前序列的token总数（含生成）
    def __len__(self):
        return self.num_tokens
    # 
    def __getitem__(self, key):
        return self.token_ids[key]

    # 加上@property，是属性装饰器，这样调用时就不用加上括号，看起来像访问成员变量
    # 判断当前序列是否已经推理完成
    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    # 返回当前生成的token数（不含prompt）
    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    # 返回该序列prompt的内容
    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    # 返回该序列生成的内容
    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    # 返回当前序列缓存了多少了kvcache block了
    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    # ? 当前序列总共需要多少个block
    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    # 最后一个block实际缓存的token数（通常不是满的）
    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    # 返回第i个block对应的token序列
    # 虽然还没有进行kvcache block的分配，但是这一步已经告诉了block manager这个序列需要多少个kvcache block了
    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    # 模型每生成一个新的token后
    def append_token(self, token_id: int):
        self.token_ids.append(token_id) # 将其追加到token_ids中
        self.last_token = token_id      # 更新last token
        self.num_tokens += 1            # 跟新总token数


    # 这两个函数用于序列化与反序列化（暂时忽略）
    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
