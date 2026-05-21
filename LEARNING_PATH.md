# Learning Path for Nano-vLLM

Nano-vLLM is most valuable as a small, readable implementation of a modern LLM serving engine. The best way to learn it is not to read files alphabetically, but to follow the actual runtime path first and then descend into the optimizations that make the engine fast.

## Core Mental Model

Think of the project in four layers:

```text
API layer
  example.py -> llm_engine.py

Serving control layer
  sequence.py -> scheduler.py -> block_manager.py

Execution layer
  model_runner.py

Model and kernel layer
  qwen3.py -> layers/*
```

If you study the repo in that order, each new file answers a question created by the previous one.

## Recommended Learning Sequence

### 1. Start from the user-facing path

Read:

1. `example.py`
2. `nanovllm/llm.py`
3. `nanovllm/engine/llm_engine.py`

Focus on:

- what the public API looks like
- what happens when `generate()` is called
- how requests flow through `add_request() -> step() -> scheduler -> model_runner`
- the distinction between **prefill** and **decode**

At this stage, you are building the skeleton of the whole system in your head.

### 2. Understand request state and scheduling

Read:

4. `nanovllm/engine/sequence.py`
5. `nanovllm/engine/scheduler.py`

Focus on:

- `SequenceStatus`
- `waiting` vs `running`
- how prefill batches are formed
- how decode batches are formed
- why decode schedules one token per sequence
- what preemption means in this engine

If you understand `Scheduler.schedule()`, you understand a large part of how vLLM-style serving works.

### 3. Learn KV-cache management

Read:

6. `nanovllm/engine/block_manager.py`

Focus on:

- block allocation and deallocation
- `block_table`
- reference counting
- prefix caching through hashed blocks
- why KV cache is managed in fixed-size blocks instead of as one large contiguous region per request

The key relationship to understand is:

- the scheduler decides **which sequences should run**
- the block manager decides **whether they have enough KV-cache memory to run**

### 4. Study the bridge from scheduling to GPU execution

Read:

7. `nanovllm/engine/model_runner.py`

Recommended order inside the file:

1. `__init__`
2. `warmup_model`
3. `allocate_kv_cache`
4. `prepare_prefill`
5. `prepare_decode`
6. `run_model`
7. `run`
8. `capture_cudagraph`

Focus on:

- how logical sequence state becomes actual GPU tensors
- what `slot_mapping`, `block_tables`, and `context_lens` mean
- how prefill inputs differ from decode inputs
- why CUDA graph capture is used for decode
- how tensor parallel workers are coordinated

This is the most important file in the repo. Once it clicks, the rest of the project becomes much easier to place.

### 5. Move into the model internals

Read:

8. `nanovllm/models/qwen3.py`

Then:

9. `nanovllm/layers/linear.py`
10. `nanovllm/layers/embed_head.py`
11. `nanovllm/layers/layernorm.py`
12. `nanovllm/layers/activation.py`
13. `nanovllm/layers/rotary_embedding.py`

Focus on:

- how Q, K, and V are packed and split
- how tensor parallel linear layers work
- how vocab parallel embedding and LM head work
- why `compute_logits()` is separated from the model forward pass
- why only the last token logits matter during prefill

You already know standard Transformer structure; here, the interesting part is how that structure is adapted for efficient serving.

### 6. Finish with the performance-specific pieces

Read:

14. `nanovllm/layers/attention.py`
15. `nanovllm/layers/sampler.py`
16. `nanovllm/utils/context.py`
17. `nanovllm/utils/loader.py`
18. `nanovllm/config.py`

Focus on:

- the Triton kernel that stores K/V into cache
- FlashAttention in prefill
- FlashAttention with KV cache in decode
- the tiny but useful `@torch.compile` example in the sampler
- how shared runtime context is passed into lower-level modules
- how model weights are loaded into packed and sharded parameters

These files are much easier to understand after you already know what the engine needs from them.

## Fast Path: One-Afternoon Overview

If you want the shortest route to the big picture, read only:

1. `example.py`
2. `nanovllm/engine/llm_engine.py`
3. `nanovllm/engine/scheduler.py`
4. `nanovllm/engine/block_manager.py`
5. `nanovllm/engine/model_runner.py`
6. `nanovllm/layers/attention.py`
7. `nanovllm/models/qwen3.py`

That gives you the main serving ideas without getting lost in every implementation detail.

## Deep Understanding Checklist

Use these exercises while reading:

### Exercise 1: Trace the first generated token

Follow one prompt from:

```text
LLM.generate()
-> add_request()
-> Scheduler.schedule()
-> ModelRunner.run()
-> model forward
-> sampler
-> Scheduler.postprocess()
```

Your goal is to explain what happens before the first completion token appears.

### Exercise 2: Compare prefill and decode

Trace the same request from its second generated token onward.

Be able to explain:

- why decode only feeds one new token per sequence
- why cached K/V makes that possible
- how the input tensors differ from prefill

### Exercise 3: Explain the three serving tensors

Write down, in your own words:

- `block_table`
- `slot_mapping`
- `context_lens`

If you can explain all three clearly, you understand the core data plumbing of the engine.

### Exercise 4: Follow prefix caching

Take two prompts with the same prefix and trace how:

- full blocks are hashed
- a second sequence reuses cached blocks
- reference counts prevent premature reuse or deletion

This is the cleanest way to understand why the block manager exists.

### Exercise 5: Reason about tensor parallelism

Mentally set `tensor_parallel_size=2` and explain how:

- `QKVParallelLinear`
- `RowParallelLinear`
- `VocabParallelEmbedding`

split work across GPUs and where communication is needed.

If you can do this without reading the code line by line, you have absorbed the design rather than just the implementation.

## Suggested Study Order in One Line

```text
LLMEngine
-> Scheduler
-> BlockManager
-> ModelRunner
-> Attention
-> Qwen3
-> Tensor-parallel layers
```

That order teaches the serving system first and the Transformer implementation second, which is the right priority for this codebase.
