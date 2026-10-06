import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

import nanovllm.layers.linear as _linear_module


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def quantize_weight_int8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel symmetric INT8 quantization.

    Args:
        tensor: Float weight tensor of shape [out_features, in_features].

    Returns:
        int8_weight : torch.int8 tensor, same shape as input.
        scale       : float32 tensor of shape [out_features, 1].
                      Multiply by dequantized weight to recover float values.
    """
    # Per-output-row absolute max, kept in float32 for precision.
    scale = tensor.float().abs().amax(dim=-1, keepdim=True) / 127.0
    # Guard against all-zero rows.
    scale = scale.clamp(min=1e-8)
    int8_weight = torch.clamp(torch.round(tensor.float() / scale), -128, 127).to(torch.int8)
    return int8_weight, scale.float()


def _is_linear_weight(param_name: str, model: nn.Module) -> bool:
    """Return True if param_name is the `.weight` of a LinearBase subclass.

    Skips embedding weights, layer-norm weights, biases, and anything else
    that should stay in full precision.
    """
    if not param_name.endswith(".weight"):
        return False
    module_name = param_name[: -len(".weight")]
    try:
        module = model.get_submodule(module_name)
    except AttributeError:
        return False
    from nanovllm.layers.linear import LinearBase
    return isinstance(module, LinearBase)


def _load_scale(
    model: nn.Module,
    param_name: str,
    scale: torch.Tensor,
    shard_id=None,
):
    """Store a quantization scale into the parent LinearBase's weight_scale buffer.

    For sharded (packed) weights the scale is sliced the same way as the
    weight so each TP rank ends up with only its own rows.

    Args:
        model      : The full model.
        param_name : Fully-qualified parameter name, e.g. "model.layers.0.self_attn.qkv_proj.weight".
        scale      : Float32 scale tensor of shape [out_features_full, 1].
        shard_id   : int (MergedColumnParallelLinear) or str (QKVParallelLinear) or None.
    """
    module_name = param_name[: -len(".weight")]
    try:
        module = model.get_submodule(module_name)
    except AttributeError:
        return

    if not hasattr(module, "weight_scale"):
        return  # Not an int8 layer — skip silently.

    if shard_id is None:
        # Simple non-merged weight: copy the whole scale.
        module.weight_scale.copy_(scale)
        return

    tp_rank = module.tp_rank
    tp_size = module.tp_size

    if isinstance(shard_id, int):
        # MergedColumnParallelLinear (gate_proj shard 0 / up_proj shard 1)
        output_sizes = module.output_sizes
        shard_offset = sum(output_sizes[:shard_id]) // tp_size
        shard_size   = output_sizes[shard_id] // tp_size
    else:
        # QKVParallelLinear: shard_id in {"q", "k", "v"}
        num_heads    = module.num_heads
        num_kv_heads = module.num_kv_heads
        head_size    = module.head_size
        if shard_id == "q":
            shard_offset = 0
            shard_size   = num_heads * head_size
        elif shard_id == "k":
            shard_offset = num_heads * head_size
            shard_size   = num_kv_heads * head_size
        else:  # "v"
            shard_offset = (num_heads + num_kv_heads) * head_size
            shard_size   = num_kv_heads * head_size

    # The scale before entering here has shape [out_full, 1].
    # chunk() along dim=0 gives each rank its slice.
    scale_chunk = scale.chunk(tp_size, dim=0)[tp_rank]
    module.weight_scale[shard_offset: shard_offset + shard_size].copy_(scale_chunk)


def load_model(model: nn.Module, path: str, quantization: str | None = None):
    """Load safetensors weights into the model.

    Args:
        model        : The model instance (weights are modified in-place).
        path         : Directory containing *.safetensors checkpoint files.
        quantization : Pass "int8" to quantize all LinearBase weights to INT8
                       on-the-fly during loading.  None (default) keeps the
                       original full-precision behaviour.
    """
    # Flip the global flag so LinearBase.__init__ allocates int8 buffers.
    # Note: the model is already constructed at this point; the flag is used
    # in forward() to select the dequant path.
    if quantization == "int8":
        _linear_module._INT8_QUANTIZED = True

    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        loaded_weight = f.get_tensor(weight_name)

                        if quantization == "int8" and _is_linear_weight(param_name, model):
                            loaded_weight, scale = quantize_weight_int8(loaded_weight)
                            weight_loader = getattr(param, "weight_loader")
                            weight_loader(param, loaded_weight, shard_id)
                            _load_scale(model, param_name, scale, shard_id)
                        else:
                            weight_loader = getattr(param, "weight_loader")
                            weight_loader(param, loaded_weight, shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    loaded_weight = f.get_tensor(weight_name)

                    if quantization == "int8" and _is_linear_weight(weight_name, model):
                        loaded_weight, scale = quantize_weight_int8(loaded_weight)
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight)
                        _load_scale(model, weight_name, scale)
                    else:
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight)
