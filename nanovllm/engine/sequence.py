from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

## python：__func__成员方法作用-由 Python 运行时在特定场景自动调用 ，不是业务代码手动调用为主，如
    # - len(seq) 会触发 seq.__len__()
    # - seq[i] 会触发 seq.__getitem__(i)
    # - pickle.dumps(seq) / 多进程传参会触发 __getstate__/__setstate__
## python：@property 是 Python 里一个很实用的装饰器，它的核心作用是：
    # 👉 把“方法”伪装成“属性”来用
    # 也就是：用 obj.xxx 的方式访问一个函数的结果，而不是 obj.xxx()

# 类变量：定义在 class 作用域里、而不是 __init__ 里挂到 self 上的变量 ，叫类变量（也叫 class attribute）
# - 所有实例默认“共享/读取同一份”类变量；但如果你给某个实例写同名属性，会在实例上生成一个新的实例属性，遮蔽类变量

class Sequence:
    # todo：kv cache block的总数这个项目有统计吗？怎么统计的？
    block_size = 256 # 这样就是类变量，每个 KV cache block 对应连续的 256 个 token 位置 。用于 paged KV cache / prefix cache 的 block 管理。它要和 Config.kvcache_block_size 对齐（默认也是 256）。
    counter = count() # 生成一个递增计数器，用来分配全局唯一的 seq_id

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING # 新请求默认在 waiting 队列里。
        # 这个token ids是在add_request()中的tokenizer进行计算
        # ** 同一个 token（tokenizer 输出的最小单元）在同一个 tokenizer/vocab 下，id 永远一样。
        # 同一段文本 → token ids 的结果是由 tokenizer 决定的；同一个 tokenizer 下同一个 token 的 id 固定，但同一个“词”不一定映射到同一个 token 。
        self.token_ids = copy(token_ids) # 浅拷贝 token 列表，隔离外部引用 
        self.last_token = token_ids[-1] # 记录最后一个 token（decode 阶段每步只喂“最后 token”）。注意这里用的是入参 token_ids ，但因为 L21 是浅拷贝，等价；更严谨会用 self.token_ids[-1] 。
        self.num_tokens = len(self.token_ids)  # 当前总 token 数（prompt + 已生成）
        # 我们需要区分 prompt 和 completion ，因为 prompt 是用户输入的初始文本，completion 是模型生成的文本。
        self.num_prompt_tokens = len(token_ids) # prompt token 数，用来切分 prompt vs completion。
        # todo：什么时候更新
        self.num_cached_tokens = 0 # prefix cache 命中时，会把“已缓存的前缀 token 数”记在这里，prefill 时只跑未缓存部分（见 ModelRunner.prepare_prefill ）

        # 逻辑id->物理id，prefill 被调度进 batch 的那一刻才会填起来
        self.block_table = [] # 核心：paged KV cache 的“页表”。每个元素是一个 block_id，表示这条序列的第 i 个 token block 映射到全局 KV cache 的哪个 block。
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self): # LLMEngine.step() 用它作为输出。
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self): # prefix cache 命中多少个完整 block
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self): # prefill时判断，当前序列总共占用多少个 block（向上取整）。KV 分配、slot_mapping 都依赖它。
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self): # 最后一个 block 里有多少 token（用于定位 decode 写 KV 的最后位置）
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i): # 返回第 i 个 block 对应的 token_ids 切片。 BlockManager.allocate() 用它算 hash、做前缀复用。
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    # 外部通信时使用-序列化：Tensor Parallel（TP）多进程时，rank0 需要把 seqs 通过 shared memory 发给其他 rank 。pickle.dumps([method_name, *args])
    # 其他卡可以通过 id → embedding 向量
    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token) # prefill返回prompt，decode返回最后一个token

    # 反序列化，TP 多进程下，rank>0 收到 rank0 发来的 seqs。pickle.loads(...)
    def __setstate__(self, state): 
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
