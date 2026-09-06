import os
from glob import glob
import json
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    files = sorted(
        file
        for file in glob(os.path.join(path, "*.safetensors"))
        if os.path.isfile(file)
    )
    if not files:
        raise FileNotFoundError(f"no safetensors files found in model path: {path}")
    available = {os.path.basename(file) for file in files}
    for index_file in glob(os.path.join(path, "*.safetensors.index.json")):
        with open(index_file, encoding="utf-8") as stream:
            index = json.load(stream)
        missing = sorted(set(index["weight_map"].values()) - available)
        if missing:
            raise FileNotFoundError(
                f"missing safetensors shards referenced by {index_file}: "
                + ", ".join(missing)
            )
    for file in files:
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
