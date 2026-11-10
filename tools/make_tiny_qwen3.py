import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config

from nanovllm.models.qwen3 import Qwen3ForCausalLM


def create_model(output: Path):
    output.mkdir(parents=True, exist_ok=True)

    config = Qwen3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        max_position_embeddings=512,
        torch_dtype="float16",
        eos_token_id=3,
    )
    config.save_pretrained(output)
    torch.manual_seed(42)
    model = Qwen3ForCausalLM(config)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "norm" in name and name.endswith("weight"):
                param.fill_(1)
            else:
                param.normal_(0, 0.02)
    save_file(
        {name: tensor.detach().contiguous() for name, tensor in model.state_dict().items()},
        output / "model.safetensors",
    )

    vocab = {"[UNK]": 0, "[PAD]": 1, "hello": 2, "[EOS]": 3}
    vocab.update({f"t{i}": i for i in range(4, 128)})
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    ).save_pretrained(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    create_model(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
