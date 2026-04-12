from nanovllm.layers.ascend.npu_ops import (
    npu_scatter_kv_blocks,
    npu_gather_kv_blocks,
    npu_burst_copy_to_cpu,
    npu_burst_copy_to_npu,
    npu_kv_rmsnorm_rope_cache,
)
from nanovllm.layers.ascend.hccl_utils import (
    hccl_init_comm_group,
    hccl_all_gather,
    hccl_reduce_scatter,
)
