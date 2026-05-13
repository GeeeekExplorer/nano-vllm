import inspect
import os
from glob import glob
import torch
import torch.distributed as dist
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def sharded_weight_loader(shard_axis: int):
    """
    Return a weight loader that shards loaded_weight along shard_axis by TP rank
    (vLLM-style). Use for A_log, dt_bias when tensor_parallel_size > 1.
    """

    def loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        tp_rank = dist.get_rank()
        shard_size = param.data.shape[shard_axis]
        start_idx = tp_rank * shard_size
        loaded_shard = loaded_weight.narrow(shard_axis, start_idx, shard_size)
        param.data.copy_(loaded_shard)

    return loader


def load_model(model: nn.Module, path: str, name_mapping=None):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # Skip mtp.* (e.g. speculative / MTP adapter), same as vLLM
                if weight_name.startswith("mtp."):
                    continue
                target_name = weight_name
                if name_mapping is not None:
                    target_name = name_mapping(target_name)
                    if target_name is None:
                        continue

                for k, (v, shard_id) in packed_modules_mapping.items():
                    if k in weight_name:
                        param_name = target_name
                        if k in param_name:
                            param_name = param_name.replace(k, v)
                        elif k == "gate_proj" and "gate_up_proj" in param_name:
                            param_name = param_name.replace("gate_up_proj", v)
                        elif k == "up_proj" and "gate_up_proj" in param_name:
                            param_name = param_name.replace("gate_up_proj", v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        tensor = f.get_tensor(weight_name)
                        if tensor.dtype != param.dtype:
                            tensor = tensor.to(param.dtype)
                        if shard_id is not None:
                            weight_loader(param, tensor, shard_id)
                        else:
                            weight_loader(param, tensor)
                        break
                else:
                    try:
                        param = model.get_parameter(target_name)
                    except AttributeError as e:
                        raise AttributeError(
                            f"Failed to locate parameter '{target_name}' "
                            f"mapped from '{weight_name}'"
                        ) from e
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    tensor = f.get_tensor(weight_name)
                    if tensor.dtype != param.dtype:
                        tensor = tensor.to(param.dtype)
                    sig = inspect.signature(weight_loader)
                    if len(sig.parameters) >= 3 and "loaded_shard_id" in sig.parameters:
                        module = getattr(weight_loader, "__self__", None)
                        if module is not None and hasattr(module, "output_sizes"):
                            # Single merged tensor (e.g. gate_up_proj): split and load each shard
                            output_sizes = module.output_sizes
                            offset = 0
                            for shard_id, size in enumerate(output_sizes):
                                part = tensor.narrow(0, offset, size)
                                weight_loader(param, part, shard_id)
                                offset += size
                        else:
                            weight_loader(param, tensor, 0)
                    else:
                        weight_loader(param, tensor)
