from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence

# python：
# - _name ：弱私有（约定），这个函数是给类内部用的，不建议外部直接调用。Python 不会阻止你调用它，但团队协作里应当尊重这个边界。
# - __name ：名称改写（name mangling，防子类意外覆盖）
# - __name__ ：魔术方法（语言协议）

# - @classmethod 的第一个参数是 cls （类本身），可以访问类变量/构造子类实例，常用于工厂方法。
# - @staticmethod 没有 self/cls ，只是“放在类命名空间里的普通函数”。
# - 你可以这样记：
#   - 需要类信息 -> classmethod
#   - 完全不需要类/实例 -> staticmethod
# 两者都可以用 类名 或 实例 调用，但“隐式传入的第一个参数”不同。


class Block: # 代表 KV cache 池里的一个物理 block（block_id 唯一）

    def __init__(self, block_id):
        self.block_id = block_id # 物理块编号
        self.ref_count = 0 # 引用计数；支持多条 seq 共享同一物理 block（prefix cache 命中时
        self.hash = -1 # 这个物理块对应 token 内容的 hash（用于 prefix cache 索引）
        # todo：hash 一样不代表 token 一定一样，为啥？
        self.token_ids = [] # 这个 block 对应的 token 内容，用于 hash 碰撞校验（hash 一样不代表 token 一定一样）

    def update(self, hash: int, token_ids: list[int]): # 写入该物理 block 的 hash 和 token 内容快照。
        self.hash = hash 
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1 # 新分配出来默认被 1 条 seq 引用
        # 清空 hash/token_ids：表示这个块现在还没形成可复用的“完整块哈希”（或者重新开始写）
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)] # 初始化物理块数组
        self.hash_to_block_id: dict[int, int] = dict() # prefix cache 索引：hash → 物理 block_id
        self.free_block_ids: deque[int] = deque(range(num_blocks)) # 输入是迭代器，初始化长度为num_blocks的空闲deque
        self.used_block_ids: set[int] = set() # 已分配的 block_id 集合（用于判断某个 hash 对应的 block 当前是否仍在使用）

    # pyhton：类方法，不依赖实例状态（处理过程没有使用成员变量和方法），第一个参数cls=class
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1): # 这里做的是“链式哈希”：当前 block 的 hash 依赖于前一个 block 的 hash（prefix）
        # todo：vllmv1中是怎么哈希的
        # todo：了解一下这个hash对象的用法，xxhash.xxh64()
        h = xxhash.xxh64() # 创建哈希对象
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little")) # 把 prefix 的 8 字节写进哈希，保证前缀不同hash不同
        h.update(np.array(token_ids).tobytes())
        return h.intdigest() # 输出 64-bit 整数哈希

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id] # 取出 block 元数据对象
        assert block.ref_count == 0 # 只有完全空闲块才能分配
        block.reset() # ref_count=1，清空 hash/token_ids
        self.free_block_ids.remove(block_id) # 从空闲队列移除（注意： remove 是线性复杂度）。
        self.used_block_ids.add(block_id) # 标记为已使用
        return self.blocks[block_id] # todo：return block，python中这是引用还是拷贝？

    # 不会修改hash_to_block_id
    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool: # 用于判断prefill阶段，是否可以分配blocks
        return len(self.free_block_ids) >= seq.num_blocks

    # 只有prefill才触发？
    def allocate(self, seq: Sequence): # 为一个 seq 做“首次分配 + prefix cache 命中”
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            # 和前一个计算的hash值以及当前的token_ids做hash
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1 # 只有当这个 block 是“完整块”（长度==block_size）才计算 hash
            # todo：第二个参数作用是什么？
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss: # hash未命中分配一个新的物理块
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids: 
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # todop: 什么时候从hash_to_block_id中删除？
                    # 如果 hash_to_block_id 指向的 block 此刻在 free（说明之前被回收过，但 hash_to_block_id 仍保留着映射），就重新 allocate 这个 block（让它“复活”）。
                    block = self._allocate_block(block_id)
            # todo：没满连token_ids都没有吗
            if h != -1: # 该物理块已经满，更新hash和block_ids
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id) # seq管理自己的block_table

    def deallocate(self, seq: Sequence): # 释放 seq 占用的物理块
        for block_id in reversed(seq.block_table): # ！！！把最前的prefix放在最后，因为block的使用是LRU（先用deque的front）
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool: # 判断decode时block是否可以追加下一个token
        # todo：！！！推理时每次只生成一个token id，对应的kvcache什么时候存呢
        # decode后seq长度+1，在scheduler的postprocess中，token_id append到seq中
        # 如果下一 token 会进入一个新 block（也就是当前 token 数刚好是 block_size 的整数倍，append 后余数变 1），则需要 1 个空闲物理块。
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence): # decode时，真正调整 block_table / 写入 prefix cache
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1: # 分配新块（跨 block）
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0: # 形成完整块时，写入 hash（用于未来 prefix cache）
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1 # 取前一个物理块的 hash（如果存在）
            # todo：为什么不会检查当前的block是不是prefix cache，是的化直接复用
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids) # 计算新 hash，写入 last_block
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
