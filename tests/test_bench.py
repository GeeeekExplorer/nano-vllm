import contextlib
import io
import json
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import bench


class Params:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class BenchTests(unittest.TestCase):
    def test_help_and_invalid_options(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as result:
                bench.parse_args(["--help"])
        self.assertEqual(result.exception.code, 0)
        self.assertIn("--output", output.getvalue())
        for args in (["--num-seqs", "0"], ["--runs", "-1"],
                     ["--min-input-len", "11", "--max-input-len", "10"],
                     ["--min-output-len", "11", "--max-output-len", "10"],
                     ["--max-input-len", "4000", "--max-output-len", "100"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    bench.parse_args(args)

    def test_workload_deterministic_and_fresh(self):
        args = bench.parse_args(["--num-seqs", "2", "--min-input-len", "2", "--max-input-len", "4",
                                 "--min-output-len", "2", "--max-output-len", "4"])
        rng = random.Random(args.seed)
        first, first_params = bench.make_workload(args, rng, Params)
        second, _ = bench.make_workload(args, rng, Params)
        repeat, repeat_params = bench.make_workload(args, random.Random(args.seed), Params)
        self.assertEqual(first, repeat)
        self.assertEqual([p.max_tokens for p in first_params], [p.max_tokens for p in repeat_params])
        self.assertNotEqual(first, second)

    def test_report_counts_actual_tokens_and_excludes_warmup(self):
        class Engine:
            calls = []

            def __init__(self, *args, **kwargs):
                pass

            def generate(self, prompts, params, **kwargs):
                self.calls.append(prompts)
                if len(self.calls) == 1:
                    return [{"token_ids": list(range(99))}]
                return [{"token_ids": [1, 2]}, {"token_ids": [3]}]

        fake = types.ModuleType("nanovllm")
        fake.LLM, fake.SamplingParams = Engine, Params
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            with mock.patch.dict(sys.modules, {"nanovllm": fake}), \
                 mock.patch.object(bench.time, "perf_counter", side_effect=[10, 12, 20, 24]), \
                 mock.patch.object(bench, "metadata", return_value={}), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                report = bench.main(["--num-seqs", "2", "--runs", "2", "--output", str(path)])
            self.assertEqual(json.loads(path.read_text()), report)
        self.assertEqual(len(Engine.calls), 3)
        self.assertEqual(report["runs"], [{"seconds": 2, "generated_tokens": 3},
                                          {"seconds": 4, "generated_tokens": 3}])
        self.assertEqual(report["summary"]["median_batch_seconds"], 3)
        self.assertEqual(report["summary"]["p95_batch_seconds"], 4)
        self.assertEqual(report["summary"]["throughput_tokens_per_second"], 1)
        self.assertIn("Run 2: Total: 3tok", output.getvalue())


if __name__ == "__main__":
    unittest.main()
