import os
from glob import glob  # 用于文件路径模式匹配
import torch
from torch import nn
from safetensors import safe_open  # 从 safetensors 库导入 safe_open，用于安全高效地读取权重文件


# 定义默认的权重加载函数
def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)  # 将加载的权重数据 (loaded_weight) 复制到模型参数 (param) 中


# 定义加载模型的主函数
def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {}) # 这个映射用于处理如 QKV 合并的权重（例如将 q_proj, k_proj, v_proj 合并存储的情况）
    for file in glob(os.path.join(path, "*.safetensors")): # 遍历指定路径下所有的 .safetensors 文件
        with safe_open(file, "pt", "cpu") as f: # 使用 safe_open 打开 safetensors 文件，指定框架为 pt (PyTorch)，设备为 cpu
            for weight_name in f.keys(): # 遍历文件中的每一个权重键名 (key)，是按字典序遍历的
                # 检查当前权重名是否在 packed_modules_mapping 的键中（即是否需要特殊处理的合并权重）
                for k in packed_modules_mapping:
                    if k in weight_name:  # 如果权重名包含映射键（例如 "qkv_proj"）
                        v, shard_id = packed_modules_mapping[k]  # 获取对应的真实参数名后缀 (v) 和分片 ID (shard_id)
                        param_name = weight_name.replace(k, v)   # 构造模型中实际的参数名称
                        param = model.get_parameter(param_name)  # 从模型中获取该参数对象
                        
                        # 获取该参数特定的加载器，通常是处理切片的加载逻辑
                        weight_loader = getattr(param, "weight_loader")
                        # 调用特定加载器，传入参数、加载的张量和分片ID
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break  # 匹配成功，跳出内部循环，处理下一个权重
                else:
                    # 如果内部循环正常结束（即没有 break，说明不是 packed module），则直接按名称获取模型参数
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader) # 获取参数的加载器，如果没有定义特定加载器，则使用 default_weight_loader
                    weight_loader(param, f.get_tensor(weight_name))
