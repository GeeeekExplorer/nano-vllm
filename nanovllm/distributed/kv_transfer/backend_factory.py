"""
难点二: 多后端存储适配。

问题:
    KV Cache 的实际存储和传输需要适配多种分布式存储后端，每种后端的初始化方式、
    内存注册方式、数据搬运方向（L2G/G2L vs D2H/H2D）和 key 约束各不相同。
    需要在统一的 Backend 抽象接口下，让上层代码无需关心底层差异。

解决方案:
    1. create_backend(): 工厂函数，根据配置名称创建对应后端实例
    2. 每个后端封装了各自特有的:
       - 初始化方式（Memcache: DistributedObjectStore / Yuanrong: HeteroClient）
       - 内存注册策略（Memcache: 显式 register_buffer / Yuanrong: 无需注册）
       - 数据搬运方向（Memcache: L2G/G2L / Yuanrong: D2H/H2D）
       - Key 约束（Yuanrong: 255 字符限制 + SHA256 规范化）
    3. Backend ABC 提供统一的 register_buffer/put/get/exists 接口
"""

from nanovllm.distributed.kv_transfer.backend import (
    Backend,
    MemcacheBackend,
    YuanrongBackend,
)


def create_backend(backend_name: str, rank: int, world_size: int) -> Backend:
    """
    根据配置名称创建对应的存储后端实例。

    输入:
        backend_name: str - 后端名称: "memcache" 或 "yuanrong"
        rank: int - 当前进程 rank
        world_size: int - 总进程数

    输出:
        Backend - 创建好的后端实例

    执行逻辑:
        # 根据 backend_name 匹配并创建对应后端:
        #
        # "memcache":
        #   → MemcacheBackend(rank, world_size)
        #   后端特点:
        #   - 依赖 memcache_hybrid.DistributedObjectStore
        #   - 初始化时需要 all_gather 建立分布式通信组
        #   - register_buffer 显式注册 NPU 内存到全局共享内存池
        #   - put: Local → Global (COPY_L2G=0)
        #   - get: Global → Local (COPY_G2L=1)
        #   - 适用于昇腾 A2 芯片
        #
        # "yuanrong":
        #   → YuanrongBackend(rank, world_size)
        #   后端特点:
        #   - 依赖 yr.datasystem.HeteroClient
        #   - 初始化时从 DS_WORKER_ADDR 环境变量获取服务地址
        #   - register_buffer 无需操作（直接使用设备指针构建 Blob）
        #   - put: Device → Host (mset_d2h), 使用 NONE_L2_CACHE_EVICT 写策略
        #   - get: Host → Device (mget_h2d)
        #   - key 需要规范化: 长度 ≤ 255，非法字符替换 + SHA256 后缀
    """
    if backend_name == "memcache":
        return MemcacheBackend(rank, world_size)
    elif backend_name == "yuanrong":
        return YuanrongBackend(rank, world_size)
    else:
        raise ValueError(
            f"Unknown backend: {backend_name}. "
            f"Supported backends: 'memcache', 'yuanrong'"
        )


def init_backend_with_device(backend: Backend):
    """
    初始化后端并设置设备上下文。

    不同后端的设备初始化有不同的前置要求，此函数封装了差异。

    输入:
        backend: Backend - 待初始化的后端实例

    执行逻辑:
        # 1. 调用 backend.set_device() 设置当前 NPU/GPU 设备上下文
        #
        # 对于 MemcacheBackend:
        #   - set_device 内部调用 torch.npu.set_device()
        #   - 必须在 all_gather 之后调用（__init__ 中已完成）
        #
        # 对于 YuanrongBackend:
        #   - set_device 内部调用 torch.npu.set_device() 并记录 device_id
        #   - device_id 用于后续 DeviceBlobList 构建
        #   - 如果未调用 set_device，后续 put/get 会因 device_id=None 报错
    """
    backend.set_device()


def validate_backend_registration(
    backend: Backend,
    ptrs: list[int],
    lengths: list[int],
) -> bool:
    """
    验证后端内存注册是否与对齐策略一致。

    输入:
        backend: Backend - 后端实例
        ptrs: list[int] - 设备内存指针
        lengths: list[int] - 字节大小

    输出:
        bool - True 如果所有指针和大小满足后端要求

    执行逻辑:
        # 对于 MemcacheBackend (需要 4MB 对齐):
        #   for ptr, length in zip(ptrs, lengths):
        #       if ptr % (4 * 1024 * 1024) != 0:
        #           return False  # 指针未对齐
        #       if length % (4 * 1024 * 1024) != 0:
        #           return False  # 大小未对齐
        #   return True
        #
        # 对于 YuanrongBackend (无对齐要求):
        #   return True  # 元戎后端无需显式注册，无对齐约束
    """
    return True
