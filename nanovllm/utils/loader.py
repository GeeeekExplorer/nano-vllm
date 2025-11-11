import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str, name_mapping=None):
    """Load safetensors weights into the model.

    Args:
        model: Target torch module whose parameters will be filled.
        path: Directory containing *.safetensors files.
        name_mapping: Optional callable that maps a weight name to the
            corresponding parameter name. Returning ``None`` skips the weight.
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
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
                        # Align dtype with the target parameter to avoid
                        # mismatches when loading mixed precision weights.
                        if tensor.dtype != param.dtype:
                            tensor = tensor.to(param.dtype)
                        weight_loader(param, tensor, shard_id)
                        break
                else:
                    try:
                        param = model.get_parameter(target_name)
                    except AttributeError as e:
                        raise AttributeError(f"Failed to locate parameter '{target_name}' mapped from '{weight_name}'") from e
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    tensor = f.get_tensor(weight_name)
                    if tensor.dtype != param.dtype:
                        tensor = tensor.to(param.dtype)
                    weight_loader(param, tensor)
