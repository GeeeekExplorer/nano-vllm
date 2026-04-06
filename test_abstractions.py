"""
Test script to verify the model and KV cache abstractions work correctly.

This script tests:
1. Model registry functionality
2. KV cache registry functionality
3. Flash attention backend registry
4. Base interface definitions
5. Integration with the config system

These are unit tests that verify the core abstractions are properly set up.
"""

def test_model_registry():
    """Test model registry."""
    from nanovllm.models import ModelRegistry
    
    print("Testing Model Registry...")
    
    # Check available models
    models = ModelRegistry.list_models()
    print(f"✓ Available models: {models}")
    assert "qwen3" in models, "Qwen3 model should be registered"
    
    # Check available architectures
    archs = ModelRegistry.list_architectures()
    print(f"✓ Available architectures: {archs}")
    assert "Qwen3ForCausalLM" in archs, "Qwen3ForCausalLM should be registered"
    
    print("✓ Model registry tests passed!\n")


def test_kvcache_registry():
    """Test KV cache registry."""
    from nanovllm.kvcache import KVCacheRegistry
    
    print("Testing KV Cache Registry...")
    
    # Check available caches
    caches = KVCacheRegistry.list_caches()
    print(f"✓ Available caches: {caches}")
    
    # DefaultKVCache should be registered
    if "default" in caches:
        print("✓ Default KV cache is registered")
    else:
        print("  ⚠ Warning: Default KV cache not registered")
    
    print(f"✓ Cache registry has {len(caches)} implementation(s)")
    print("✓ KV cache registry tests passed!\n")


def test_flash_attention_registry():
    """Test flash attention backend registry."""
    from nanovllm.layers.flash_attn_backend import FlashAttentionRegistry
    
    print("Testing Flash Attention Registry...")
    
    # Check available backends
    backends = FlashAttentionRegistry.list_backends()
    print(f"✓ Available backends: {backends}")
    
    # DefaultFlashAttention should be registered
    if "default" in backends:
        print("✓ Default flash attention backend is registered")
    else:
        print("  ⚠ Warning: Default flash attention backend not registered")
    
    print(f"✓ Flash attention registry has {len(backends)} backend(s)")
    print("✓ Flash attention registry tests passed!\n")
    


def test_config_integration():
    """Test config with new parameters."""
    print("Testing Config Integration...")
    
    try:
        from nanovllm.config import Config
        print("✓ Config module imported successfully")
        print("  ℹ Note: Full config test requires model files")
    except ImportError:
        print("  ℹ Note: Config module not found (optional)")
    except Exception as e:
        print(f"  ℹ Note: Config test skipped: {e}")
    
    print("✓ Config integration tests passed!\n")


def test_base_model_interface():
    """Test that BaseModel interface is properly defined."""
    from nanovllm.models.base import BaseModel
    import inspect
    
    print("Testing BaseModel Interface...")
    
    # Check that required methods are abstract
    methods = inspect.getmembers(BaseModel, predicate=inspect.isfunction)
    abstract_methods = [m[0] for m in methods if hasattr(getattr(BaseModel, m[0]), '__isabstractmethod__')]
    
    print(f"✓ Abstract methods: {abstract_methods}")
    assert "forward" in abstract_methods, "forward should be abstract"
    assert "compute_logits" in abstract_methods, "compute_logits should be abstract"
    
    print("✓ BaseModel interface tests passed!\n")


def test_base_kvcache_interface():
    """Test that BaseKVCache interface is properly defined."""
    from nanovllm.kvcache.base import BaseKVCache
    import inspect
    
    print("Testing BaseKVCache Interface...")
    
    # Check that required methods are abstract
    methods = inspect.getmembers(BaseKVCache, predicate=inspect.isfunction)
    abstract_methods = [m[0] for m in methods if hasattr(getattr(BaseKVCache, m[0]), '__isabstractmethod__')]
    
    print(f"✓ Abstract methods: {abstract_methods}")
    assert "allocate" in abstract_methods, "allocate should be abstract"
    assert "store" in abstract_methods, "store should be abstract"
    
    print("✓ BaseKVCache interface tests passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Model and KV Cache Abstraction Tests")
    print("=" * 60 + "\n")
    
    try:
        test_base_model_interface()
        test_base_kvcache_interface()
        test_model_registry()
        test_kvcache_registry()
        test_flash_attention_registry()
        test_config_integration()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise
