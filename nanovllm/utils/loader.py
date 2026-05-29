import os
import time
from glob import glob
import torch
import torch.distributed as dist
from torch import nn
from safetensors import safe_open

from nanovllm.layers import materialize_full


def default_weight_loader(param: nn.Parameter, loaded_weight):
    # loaded_weight 可能是 safetensors 的 slice 句柄，这里整张读出来再拷贝。
    param.data.copy_(materialize_full(loaded_weight))


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    is_main = (not dist.is_initialized()) or dist.get_rank() == 0
    files = sorted(glob(os.path.join(path, "*.safetensors")))
    t_start = time.time()
    for fi, file in enumerate(files):
        t_file = time.time()
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # 传 slice 句柄而非整张 tensor：各 weight_loader 只会按需读取
                # 当前 rank 的分片，避免每个 rank 读全量后再丢弃，多卡加载显著加速。
                loaded = f.get_slice(weight_name)
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        param.weight_loader(param, loaded, shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded)
        if is_main:
            print(
                f"[load_model] {fi + 1}/{len(files)} {os.path.basename(file)} "
                f"+{time.time() - t_file:.1f}s (total {time.time() - t_start:.1f}s)",
                flush=True,
            )
