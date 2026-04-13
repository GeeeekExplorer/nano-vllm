"""
测试 KV Cache 和 Prefix Cache 命中情况。

测试策略：
1. 用一段很长的 system prompt 作为公共前缀
2. 第一轮请求：冷启动，无 cache，prefill 耗时正常
3. 第二轮请求：相同前缀，应命中 prefix cache，prefill computed token 数接近 0，速度极快
4. 打印每一步 prefill 的耗时和 cached token 数量来验证
"""
import os
from time import perf_counter
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


class InstrumentedLLM(LLM):
    """覆盖 step()，打印每次 prefill 的 cached/computed token 数及耗时。"""

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        if is_prefill:
            total_tokens = sum(len(seq) for seq in seqs)
            cached_tokens = sum(seq.num_cached_tokens for seq in seqs)
            computed_tokens = total_tokens - cached_tokens
            print(
                f"  [Prefill] total={total_tokens} tokens | "
                f"cached={cached_tokens} | computed={computed_tokens}"
            )
        t = perf_counter()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        elapsed = perf_counter() - t
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        if is_prefill:
            total_tokens = sum(len(seq) for seq in seqs)
            cached_tokens = sum(seq.num_cached_tokens for seq in seqs)
            computed_tokens = total_tokens - cached_tokens
            tps = computed_tokens / elapsed if elapsed > 0 else float("inf")
            print(f"  [Prefill] elapsed={elapsed*1000:.1f}ms | throughput={tps:.0f} tok/s (computed only)")
        return outputs, (sum(len(seq) for seq in seqs) if is_prefill else -len(seqs))

    def generate_timed(self, prompts, sampling_params):
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        t0 = perf_counter()
        while not self.is_finished():
            output, _ = self.step()
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
        total_elapsed = perf_counter() - t0
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        return [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in outputs
        ], total_elapsed


def build_prompt(tokenizer, system: str, user: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def main():
    path = os.path.expanduser("~/huggingface/Llama-3.2-1B-Instruct/")
    tokenizer = AutoTokenizer.from_pretrained(path)

    # 公共前缀需超过一个 block（256 tokens）才能触发 prefix cache，重复 8 次约 400+ tokens
    long_system = (
        "You are a helpful, harmless, and honest AI assistant. "
        "Your goal is to provide accurate, thoughtful responses. "
        "Always be concise and clear in your explanations. "
        "When answering questions, consider multiple perspectives. "
        "Be respectful and professional at all times. "
    ) * 8

    prompt1 = build_prompt(tokenizer, long_system, "What is 1 + 1?")
    prompt2 = build_prompt(tokenizer, long_system, "What is 2 + 2?")

    n1 = len(tokenizer.encode(prompt1))
    n2 = len(tokenizer.encode(prompt2))
    print(f"Prompt 1: {n1} tokens")
    print(f"Prompt 2: {n2} tokens")
    print(f"Block size: 256 tokens — prefix cache triggers on full 256-token blocks")
    print()

    llm = InstrumentedLLM(path, enforce_eager=True, tensor_parallel_size=1)
    sp = SamplingParams(temperature=0.6, max_tokens=32)

    print("=" * 60)
    print("Round 1 — Cold start (no cache expected)")
    print("=" * 60)
    outputs1, t1 = llm.generate_timed([prompt1], sp)
    print(f"  Total time: {t1*1000:.1f}ms")
    print(f"  Output: {outputs1[0]['text']!r}")
    print()

    print("=" * 60)
    print("Round 2 — Same prefix (prefix cache hit expected)")
    print("=" * 60)
    outputs2, t2 = llm.generate_timed([prompt2], sp)
    print(f"  Total time: {t2*1000:.1f}ms")
    print(f"  Output: {outputs2[0]['text']!r}")
    print()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    speedup = t1 / t2 if t2 > 0 else float("inf")
    print(f"  Round 1 (cold): {t1*1000:.1f}ms")
    print(f"  Round 2 (warm): {t2*1000:.1f}ms")
    print(f"  Speedup:        {speedup:.2f}x")
    if speedup > 1.5:
        print("  -> Prefix cache is working!")
    else:
        print("  -> Cache may not have been hit (check if shared prefix > 256 tokens)")


if __name__ == "__main__":
    main()
