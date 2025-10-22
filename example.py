import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
from nanovllm.utils.secure import set_security_config


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    # 启用加密方案：CPU 作为 TEE，GPU 作为不可信加速器
    set_security_config(
        enable_softmax_encrypt=True,   # Q/K 正交加密（softmax 不变）
        enable_linear_noise=True,      # QKV 线性层输入侧加噪 + rW 预计算
        encrypt_on_cpu=True,           # 在 CPU(TEE) 执行加密
        decrypt_on_cpu=True,           # 在 CPU(TEE) 执行解密补偿（更贴合 TEE）
        noise_pool_size=16,
        noise_scale=0.05,
        seed=1234,
    )
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "介绍自己",
        "计算1到100的数字求和",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
