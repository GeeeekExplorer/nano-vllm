import sys

if sys.platform == "win32":
    from nanovllm.layers.attention_windows import Attention
else:
    from nanovllm.layers.attention_fast import Attention
