from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence

# kvcache 块的数据结构
class Block:

    def __init__(self, block_id):
        self.block_id = block_id  # 块的唯一标识
        self.ref_count = 0        # 该block被引用的计数（管理有多少个序列在使用这个块）
        self.hash = -1            # 块内容的hash值，用于判断是否缓存命中
        self.token_ids = []       # 该块中存储token的数据列表

    # 更新块的内存和hash值
    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids
    
    # 重置块，清空所有数据
    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []

# 管理一组block的分配、释放和缓存逻辑
class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size   # 每个块的大小
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)] # 所有kvcache block的列表（每个元素都是一个block）
        self.hash_to_block_id: dict[int, int] = dict() # 哈希值到块id的映射，用于快速缓存查找
        self.free_block_ids: deque[int] = deque(range(num_blocks)) # 空闲的块ID队列
        self.used_block_ids: set[int] = set()  # 已经使用的块的集合（set）

    # hash计算
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        # prefix是该block前一个block的哈希值，-1代表没有前缀
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little")) # 将prefix转为8字节，并进行hash计算
        h.update(np.array(token_ids).tobytes()) 
        return h.intdigest()

    # 从free队列中拿一个块，进行初始化并加入used集合
    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)  # 从空闲的块队列中删除
        self.used_block_ids.add(block_id)     # 加入到used集合中
        return self.blocks[block_id]

    # 将块释放回free队列
    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0 # 确保没有序列引用这个block了
        self.used_block_ids.remove(block_id)   # 从used集合中删除
        self.free_block_ids.append(block_id)   # 加入free队列中

    # 判断当前空闲块是否足够分给一个序列所需的所有block
    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    # 给序列分配kvcache block的核心逻辑
    def allocate(self, seq: Sequence):
        assert not seq.block_table  # 确保这个序列还没有被分配过block（block table为空）
        h = -1  # 前一个block的hash值，用于链式哈希
        cache_miss = False  # 标记是否命中缓存

        for i in range(seq.num_blocks):
            token_ids = seq.block(i)  # 拿到该序列分配给第i个block的内容
            # 若block是完整大小，就计算该block中内容的链式哈希；若不是，hash设为-1不参与缓存匹配
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            
            block_id = self.hash_to_block_id.get(h, -1) # 拿到这个hash对应的block id
            # 若hash没找到活token不一样，判断为缓存未命中
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            # 若缓存未命中，从free队列中最左端取一个，分配给当前block
            if cache_miss:
                block_id = self.free_block_ids[0]  
                block = self._allocate_block(block_id)  # 从free中拿出一个初始化并加入used
            # 若缓存命中
            else:
                seq.num_cached_tokens += self.block_size # 更新该序列已经缓存的token数量
                # 若block正在被使用，直接取该block并增加该block的引用次数
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                # 若block没在使用，
                else:
                    block = self._allocate_block(block_id)  # 从free中拿出一个初始化并加入used
            if h != -1:
                block.update(h, token_ids)   # 若hash有效，更新block的hash和token内容
                self.hash_to_block_id[h] = block_id # 更新block和hash的映射
            seq.block_table.append(block_id) 
            # 把这个block id加入该序列的block table，方便序列知道自己的token被存在哪些block中

    # 释放某序列所占用的block
    def deallocate(self, seq: Sequence):
        # 从后往前遍历该序列的block table
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id] # 获取实际的block对象
            block.ref_count -= 1          # 将该block对象的引用数-1
            # 若引用数减为0，则说明没有任何序列需要这个block,则释放这个block的所有资源
            if block.ref_count == 0:      
                self._deallocate_block(block_id)
        
        seq.num_cached_tokens = 0  # 清空该序列已经缓存的token数
        seq.block_table.clear()    # 将block也清空

    # 判断当前系统是否有足够的空闲free来追加token
    def can_append(self, seq: Sequence) -> bool:
        # 当序列长度对block size取模=1时，意味着下一个生成的token会开启一个新的block
        # 检查空闲的block的个数，大于的话就还有空闲block可供分配
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    # 处理序列追加token后 block的更新
    def may_append(self, seq: Sequence):
        block_table = seq.block_table  # 拿到该序列的block table
        last_block = self.blocks[block_table[-1]] # 拿到该序列的最后一个block
        
        # 当序列长度对block size取模=1，意味着该序列的所有block都满了，新生成的token需要新的block来缓存
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1  # 确保最后一个block时有hash值的
            block_id = self.free_block_ids[0] # 取空闲块队列中最左边的block
            self._allocate_block(block_id)    # 从free去除加入used
            block_table.append(block_id)      # 将这个新block加入该序列的block table
        # 当序列长度对block size取模=0时，意味着加入新生成的token后，正好让所有block都满了（不需要申请新的block）
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1  # 确保最后一个block还没有被计算hash
            token_ids = seq.block(seq.num_blocks-1) # 取出该序列最后一个block对应的token内容
            # 若该table中不止一个block，则取倒数第二个block，拿到其hash值，作为prefix
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix) # 根据上一个block的hash和当前block的数据计算hash
            last_block.update(h, token_ids) # 更新最后一个block的hash和token内容（因为新增了一个token）
            self.hash_to_block_id[h] = last_block.block_id # 将这个hash 映射到block id
        else:
            assert last_block.hash == -1
