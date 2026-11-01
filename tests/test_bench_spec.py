import hashlib
import json
from types import SimpleNamespace

import pytest

from bench_spec import (ROOT, RESULT_KEYS, assert_gpu, build_result, engine_kwargs, gpu_state, installed_packages,
                        kv_budget_bytes, load_panel, run_closed_loop)

TARGET_BLOCK, DRAFT_BLOCK = 37_748_736, 29_360_128


def write_panel(tmp_path, count):
    path = tmp_path / "panel.json"
    path.write_text(json.dumps([{"id": f"p{index}", "category": "NIAH", "prompt_token_ids": [index]}
                                for index in range(count)]))
    return path


def test_load_panel_full_cycles(tmp_path):
    path = write_panel(tmp_path, 48)
    assert [len(load_panel(path, concurrency, "full_cycles")) for concurrency in (1, 13, 32, 64)] == [48, 96, 144, 288]
    assert load_panel(path, 32, "full_cycles")[48]["id"] == "p0"


def test_load_panel_prefix(tmp_path):
    path = write_panel(tmp_path, 128)
    assert [entry["id"] for entry in load_panel(path, 8, "prefix")] == [f"p{index}" for index in range(48)]
    assert [entry["id"] for entry in load_panel(path, 64, "prefix")] == [f"p{index}" for index in range(128)] * 2


@pytest.mark.parametrize("longest, concurrency, k, block_bytes, expected", [
    (2262, 64, 3, TARGET_BLOCK, 26_575_110_144),
    (2262, 64, 3, DRAFT_BLOCK, 20_669_530_112),
    (2816, 64, 3, TARGET_BLOCK, 33_822_867_456),
    (1000, 1, 0, TARGET_BLOCK, 226_492_416),
])
def test_kv_budget_bytes(longest, concurrency, k, block_bytes, expected):
    panel = [{"id": "long", "category": "NIAH", "prompt_token_ids": [0] * longest},
             {"id": "short", "category": "NIAH", "prompt_token_ids": [0] * 10}]
    assert kv_budget_bytes(panel, concurrency, 512, k, block_bytes) == expected


class FakeSequence:
    def __init__(self, target_len):
        self.target_len = target_len
        self.num_completion_tokens = 0
        self.is_finished = False

    @property
    def completion_token_ids(self):
        return list(range(self.num_completion_tokens))


class FakeEngine:
    """Prefill steps give every waiting request one token; decode steps give every running one tokens_per_step."""

    def __init__(self, tokens_per_step=1):
        self.tokens_per_step = tokens_per_step
        self.waiting, self.running = [], []
        self.num_steps = 0
        self.max_in_flight = 0

    def add_request(self, prompt_token_ids, params):
        sequence = FakeSequence(prompt_token_ids[0])
        self.waiting.append(sequence)
        return sequence

    def step(self):
        self.max_in_flight = max(self.max_in_flight, len(self.waiting) + len(self.running))
        if self.waiting:
            for sequence in self.waiting:
                sequence.num_completion_tokens += 1
            self.running += self.waiting
            self.waiting = []
        else:
            for sequence in self.running:
                sequence.num_completion_tokens += min(self.tokens_per_step,
                                                      sequence.target_len - sequence.num_completion_tokens)
        for sequence in self.running:
            sequence.is_finished = sequence.num_completion_tokens >= sequence.target_len
        self.running = [sequence for sequence in self.running if not sequence.is_finished]
        self.num_steps += 1


def panel_of(lengths):
    return [{"id": f"r{index}", "category": "NIAH", "prompt_token_ids": [length]} for index, length in enumerate(lengths)]


def trace(record):
    return {request["id"]: (request["submit_step"], request["deliveries"], request["finish_step"])
            for request in record["requests"]}


def test_closed_loop_step_by_step():
    record = run_closed_loop(FakeEngine(), panel_of([3, 1, 2]), None, concurrency=2)
    assert trace(record) == {"r0": (0, [[0, 1], [2, 1], [3, 1]], 3),
                             "r1": (0, [[0, 1]], 0),
                             "r2": (1, [[1, 1], [2, 1]], 2)}


def test_closed_loop_several_tokens_per_delivery():
    record = run_closed_loop(FakeEngine(tokens_per_step=3), panel_of([7]), None, concurrency=1)
    assert trace(record) == {"r0": (0, [[0, 1], [1, 3], [2, 3]], 2)}


def test_closed_loop_never_exceeds_concurrency():
    engine = FakeEngine()
    record = run_closed_loop(engine, panel_of([2] * 5), None, concurrency=2)
    submits = [request["submit_step"] for request in record["requests"]]
    assert engine.max_in_flight <= 2 and submits == sorted(submits)
    assert [request["id"] for request in record["requests"]] == [f"r{index}" for index in range(5)]


def test_closed_loop_refills_two_finished_in_one_step():
    trace_by_id = trace(run_closed_loop(FakeEngine(), panel_of([1, 1, 2, 2]), None, concurrency=2))
    assert trace_by_id["r2"] == trace_by_id["r3"] == (1, [[1, 1], [2, 1]], 2)


def test_gpu_state():
    def run(args, **kwargs):
        if args[0] == "nvidia-smi":
            return SimpleNamespace(stdout="NVIDIA RTX PRO 6000 Blackwell Server Edition, 600.00, 2617, 14001, Enabled, 41, "
                                          "0x0000000000000000\n")
        return SimpleNamespace(stdout="gpu-pro6000-7\n")
    assert gpu_state(run) == {"name": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "power.limit": "600.00",
                              "clocks.max.sm": "2617", "clocks.max.memory": "14001", "ecc.mode.current": "Enabled",
                              "temperature.gpu": "41", "clocks_event_reasons.active": "0x0000000000000000",
                              "node": "gpu-pro6000-7"}


def test_assert_gpu_rejects_another_card():
    with pytest.raises(SystemExit) as exit_info:
        assert_gpu({"name": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"},
                   "NVIDIA RTX PRO 6000 Blackwell Server Edition")
    assert exit_info.value.code == 3


def test_installed_packages():
    distributions = lambda: [SimpleNamespace(metadata={"Name": "a"}, version="1")]    # noqa: E731
    assert installed_packages(distributions) == ("a==1\n", hashlib.sha256(b"a==1\n").hexdigest())


def test_build_result_has_the_schema_keys():
    result = build_result(**{key: None for key in RESULT_KEYS[1:]})
    assert set(result) == set(RESULT_KEYS) and result["schema"] == "bench-v2"


def kwargs_of_every_line(protocol_file: str, tmp_path) -> dict[str, dict]:
    protocol = json.loads((ROOT / "benchmarks" / protocol_file).read_text())
    for directory in [protocol["models"]["target"], *protocol["models"]["drafts"].values()]:
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "config.json").write_text(
            json.dumps({"num_hidden_layers": 2, "num_key_value_heads": 1, "head_dim": 2}))
    panel = [{"prompt_token_ids": [0] * 100}]
    return {name: engine_kwargs(protocol, line, panel, 8, 17011, str(tmp_path))[2]
            for name, line in protocol["lines"].items()}


def test_engine_kwargs_follow_the_protocol_lines(tmp_path):
    kwargs = kwargs_of_every_line("protocol-v2.json", tmp_path)
    assert len({value["kv_cache_memory_bytes"] for value in kwargs.values()}) == 1    # one target budget for every line
    assert "draft_model" not in kwargs["k0"]
    assert kwargs["model_k3"]["num_speculative_tokens"] == 3 and kwargs["model_k3"]["draft_kv_cache_memory_bytes"] > 0
    assert "draft_type" not in kwargs["model_k3"]
    assert kwargs["dflash_k7"]["draft_type"] == "dflash" and kwargs["dflash_k7"]["num_speculative_tokens"] == 7
    assert "draft_kv_cache_memory_bytes" not in kwargs["dflash_k7"]


def test_engine_kwargs_give_dspark_its_draft_type(tmp_path):
    kwargs = kwargs_of_every_line("protocol-v2.1.json", tmp_path)
    assert kwargs["dspark_k7"]["draft_type"] == "dspark" and kwargs["dspark_k7"]["num_speculative_tokens"] == 7
    assert "draft_kv_cache_memory_bytes" not in kwargs["dspark_k7"]
