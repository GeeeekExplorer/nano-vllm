"""Target and draft runners on one DeviceRuntime; both weight sets load before either KV cache."""

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.runtime import DeviceRuntime


class DualModelRunner:

    def __init__(self, target_config: Config, draft_config: Config):
        if target_config is draft_config:
            raise ValueError("target and draft need separate Config objects")
        shared = ("kvcache_block_size", "max_model_len", "max_num_seqs", "max_num_batched_tokens", "enforce_eager")
        target_values = (target_config.kvcache_block_size, target_config.max_model_len, target_config.max_num_seqs,
                         target_config.max_num_batched_tokens, target_config.enforce_eager)
        draft_values = (draft_config.kvcache_block_size, draft_config.max_model_len, draft_config.max_num_seqs,
                        draft_config.max_num_batched_tokens, draft_config.enforce_eager)
        if target_values != draft_values:
            raise ValueError(f"target and draft must agree on {shared}")
        self.runtime = DeviceRuntime(0, 1)
        self.target = ModelRunner(target_config, 0, [], runtime=self.runtime, defer_cache=True, kv_role="target")
        self.draft = ModelRunner(draft_config, 0, [], runtime=self.runtime, defer_cache=True, kv_role="draft")
        self.target.initialize_cache()
        self.draft.initialize_cache()

    def exit(self):
        self.draft.exit()
        self.target.exit()
        self.runtime.close()
