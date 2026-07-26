import os
from glob import glob
import torch
import importlib
from torch import nn
from safetensors import safe_open
from nanovllm.models.registry import NANO_VLLM_MODELS

def load_model_arch_from_config(hf_config):
    arch = hf_config.architectures[0]
    if arch not in NANO_VLLM_MODELS:
        raise ValueError(f"Unsupported architecture: {arch}")
    module_name, class_name = NANO_VLLM_MODELS[arch]
    module = importlib.import_module(f"nanovllm.models.{module_name}")
    model_cls = getattr(module, class_name)
    return model_cls


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
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
