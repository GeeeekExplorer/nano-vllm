import json

import pytest

from nanovllm.config import Config

TINY_QWEN3 = {
    "model_type": "qwen3", "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 128,
    "max_position_embeddings": 1024, "torch_dtype": "bfloat16", "tie_word_embeddings": False,
}


@pytest.fixture
def model_dir(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(TINY_QWEN3))
    return str(tmp_path)


def test_defaults_have_no_draft(model_dir):
    config = Config(model_dir)
    assert config.draft_model is None and config.num_speculative_tokens == 0


def test_draft_requires_k_prefix_cache_off_and_budgets(model_dir):
    kwargs = dict(draft_model=model_dir, num_speculative_tokens=3, enable_prefix_cache=False,
                  kv_cache_memory_bytes=1 << 20, draft_kv_cache_memory_bytes=1 << 20)
    Config(model_dir, **kwargs)
    with pytest.raises(ValueError):
        Config(model_dir, **{**kwargs, "num_speculative_tokens": 0})
    with pytest.raises(ValueError):
        Config(model_dir, **{**kwargs, "enable_prefix_cache": True})
    with pytest.raises(ValueError):
        Config(model_dir, **{**kwargs, "draft_kv_cache_memory_bytes": None})
    with pytest.raises(ValueError):
        Config(model_dir, **{**kwargs, "tensor_parallel_size": 2})


def test_speculative_tokens_without_draft_is_rejected(model_dir):
    with pytest.raises(ValueError):
        Config(model_dir, num_speculative_tokens=2)


def test_draft_rejects_batch_sizes_beyond_the_graph_table(model_dir):
    kwargs = dict(draft_model=model_dir, num_speculative_tokens=3, enable_prefix_cache=False,
                  kv_cache_memory_bytes=1 << 20, draft_kv_cache_memory_bytes=1 << 20)
    Config(model_dir, max_num_seqs=512, **kwargs)
    with pytest.raises(ValueError):
        Config(model_dir, max_num_seqs=513, **kwargs)


@pytest.fixture
def dflash_dir(tmp_path):
    return block_draft_dir(tmp_path, "dflash")


def block_draft_dir(tmp_path, name: str, **fields) -> str:
    directory = tmp_path / name
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({**TINY_QWEN3, "block_size": 7, "mask_token_id": 127,
                                                       "target_layer_ids": [0], **fields}))
    return str(directory)


@pytest.fixture
def dspark_dir(tmp_path):
    return block_draft_dir(tmp_path, "dspark", markov_rank=8, markov_head_type="vanilla")


def test_dspark_takes_k_from_its_block_and_needs_no_draft_budget(model_dir, dspark_dir):
    config = Config(model_dir, draft_model=dspark_dir, draft_type="dspark", enable_prefix_cache=False,
                    kv_cache_memory_bytes=1 << 20, enforce_eager=True)
    assert config.num_speculative_tokens == 7 and config.draft_kv_cache_memory_bytes is None


def test_the_draft_type_must_match_the_markov_head(model_dir, dflash_dir, dspark_dir, tmp_path):
    kwargs = dict(enable_prefix_cache=False, kv_cache_memory_bytes=1 << 20, enforce_eager=True)
    with pytest.raises(ValueError):
        Config(model_dir, draft_model=dspark_dir, draft_type="dflash", **kwargs)
    with pytest.raises(ValueError):
        Config(model_dir, draft_model=dflash_dir, draft_type="dspark", **kwargs)
    gated = block_draft_dir(tmp_path, "gated", markov_rank=8, markov_head_type="gated")
    with pytest.raises(ValueError):
        Config(model_dir, draft_model=gated, draft_type="dspark", **kwargs)


def test_dflash_takes_k_from_its_block_and_needs_no_draft_budget(model_dir, dflash_dir):
    config = Config(model_dir, draft_model=dflash_dir, draft_type="dflash", enable_prefix_cache=False,
                    kv_cache_memory_bytes=1 << 20, enforce_eager=True)
    assert config.num_speculative_tokens == 7 and config.draft_kv_cache_memory_bytes is None


def test_dflash_rejects_another_k_and_unknown_draft_types(model_dir, dflash_dir):
    kwargs = dict(draft_model=dflash_dir, enable_prefix_cache=False, kv_cache_memory_bytes=1 << 20, enforce_eager=True)
    with pytest.raises(ValueError):
        Config(model_dir, draft_type="dflash", num_speculative_tokens=3, **kwargs)
    with pytest.raises(ValueError):
        Config(model_dir, draft_type="DFlash", **kwargs)
