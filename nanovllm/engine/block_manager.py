from collections import deque # 双端队列，用于高效地管理空闲块列表（O(1) 的头部/尾部操作）
import xxhash # 一个极快的哈希算法库，用于计算 Token 序列的哈希值，以便进行前缀匹配和去重
import numpy as np

from nanovllm.engine.sequence import Sequence

"""
PagedAttention 机制的核心组件，用于高效管理显存中的 KV Cache 块。它支持块的分配、释放以及基于内容哈希的块共享（Prefix Caching）
"""
class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0 # 多个seq都有这个块（例如 System Prompt），作reference_count++
        self.hash = -1 # 每个block hash唯一，用于查找是否已存在相同内容的块
        self.token_ids = [] # 该块当前存储的 token ID List

    def update(self, hash: int, token_ids: list[int]): # 当块填满或内容确定时，更新其哈希值和内容记录
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict() # 哈希表：映射 block_hash -> block_id，用于实现前缀缓存（Prefix Caching）查找
        self.free_block_ids: deque[int] = deque(range(num_blocks)) # 双向队列作空块的给出
        self.used_block_ids: set[int] = set() # set存用着的block

    @classmethod # 属于类而非对象的方法，不需要创建一个 BlockManager 对象就能调用它
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little")) # 如果有前缀块的哈希值，先 update 进去，实现链式哈希，保证了只有在上下文完全一致的情况下，显存块才会被复用
        h.update(np.array(token_ids).tobytes()) # 相比于手动遍历 list 或用 struct 打包，Numpy 在处理批量数字转字节流时效率非常高，是 C 语言级别的速度
        return h.intdigest()# 返回计算出的最终哈希值（一个整数）

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0 # 只有引用计数为 0 时才真正回收
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool: # 检查剩余空闲块是否足够分配给一个新的序列
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence): # 针对于共享前缀或被抢占恢复的序列，通常在 Prefill 阶段调用，它会尝试重用已有的块（Prefix Caching）
        assert not seq.block_table # 确保序列还没有分配过块
        h = -1
        cache_miss = False # 标记是否发生缓存未命中
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1 # 只有填满的块才计算哈希并尝试匹配缓存
            block_id = self.hash_to_block_id.get(h, -1) # 不在dict里面返回-1
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids: # or后面是找到了但内容不匹配（哈希碰撞防御）
                cache_miss = True
            """
            当一个块被 deallocate 释放时，它会回到 free_block_ids 队列，但它在 hash_to_block_id 中的记录并没有被删除（这是一种惰性删除策略，或者叫“僵尸块”策略）
            所以需要将“决定用哪个块”和“把块标记为已用”解耦：
            调用者 (allocate)：负责决定用哪个块（是随便拿一个，还是复用旧的）
            执行者 (_allocate_block)：只负责执行分配动作（从空闲列表移除、加入已用集合、重置状态）
            """
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids: # 如果块正在被使用，增加引用计数（共享块）
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id) # 僵尸块重分配
            if h != -1: # 如果计算了哈希（即块是满的），更新块信息和全局哈希表
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id) # 将物理块 ID 填进此seq的页表

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id) # 如果引用计数归零，说明没有序列在使用该块，回收它
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:      # 在decode阶段使用的，decode阶段一个一个token生成，生成一个就判断一次can_append来判断要不要加新块
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence): # Decode 阶段 维护显存块状态：每当生成一个新的 Token 后，检查并更新当前序列的物理块状态，包括分配新块、计算满块哈希、以及维护哈希表；块填满和计算块哈希是分开的，检测到块填满才算哈希，和decode解耦
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1: # 刚刚生成了一个 token，且这个 token 是新的一块的第 1 个 token
            assert last_block.hash != -1 # 既然上一个块已经满了，那么上一个块必须已经计算过哈希并封存
            block_id = self.free_block_ids[0] # 从空闲队列头部获取一个空闲块 ID
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0: # 刚刚生成的那个 token 刚好把当前块填满了
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1) # 获取当前这个刚刚填满的块的所有 Token ID
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1 # 获取前一个块的哈希值作为前缀，如果这是序列的第一个块，前缀就是 -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids) # 更新物理块对象，记录它的哈希和内容
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
