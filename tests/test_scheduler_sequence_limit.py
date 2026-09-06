from collections import deque
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch


def load_scheduler():
    config = ModuleType("nanovllm.config")
    config.Config = object
    sequence = ModuleType("nanovllm.engine.sequence")
    sequence.Sequence = object
    sequence.SequenceStatus = SimpleNamespace(RUNNING=object())
    block_manager = ModuleType("nanovllm.engine.block_manager")
    block_manager.BlockManager = object
    with patch.dict(sys.modules, {
        "nanovllm.config": config,
        "nanovllm.engine.sequence": sequence,
        "nanovllm.engine.block_manager": block_manager,
    }):
        path = Path(__file__).parents[1] / "nanovllm/engine/scheduler.py"
        spec = spec_from_file_location("scheduler_sequence_limit", path)
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load scheduler module")
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
    return module.Scheduler


Scheduler = load_scheduler()


def make_sequence(seq_id, *, running=False):
    return SimpleNamespace(
        seq_id=seq_id,
        block_table=[] if not running else [0],
        num_tokens=4,
        num_cached_tokens=4 if running else 0,
        num_scheduled_tokens=0,
        is_prefill=not running,
        status=None,
    )


def make_scheduler(*, running=(), waiting=(), max_num_batched_tokens=64):
    scheduler = object.__new__(Scheduler)
    scheduler.max_num_seqs = 2
    scheduler.max_num_batched_tokens = max_num_batched_tokens
    scheduler.block_size = 4
    scheduler.running = deque(running)
    scheduler.waiting = deque(waiting)
    scheduler.block_manager = Mock()
    scheduler.block_manager.can_allocate.return_value = 0
    scheduler.block_manager.can_append.return_value = True
    return scheduler


class SchedulerSequenceLimitTest(TestCase):
    def test_running_sequences_count_toward_limit(self):
        running = [make_sequence(1, running=True), make_sequence(2, running=True)]
        waiting = make_sequence(3)
        scheduler = make_scheduler(running=running, waiting=[waiting])

        scheduled, is_prefill = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, running)
        self.assertEqual(list(scheduler.waiting), [waiting])
        scheduler.block_manager.can_allocate.assert_not_called()

    def test_waiting_sequence_uses_remaining_slot(self):
        running = make_sequence(1, running=True)
        waiting = make_sequence(2)
        scheduler = make_scheduler(running=[running], waiting=[waiting])

        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [waiting])
        self.assertEqual(list(scheduler.running), [running, waiting])
        self.assertFalse(scheduler.waiting)

    def test_chunked_prefill_consumes_the_remaining_sequence_slot(self):
        running = make_sequence(1, running=True)
        first = make_sequence(2)
        second = make_sequence(3)
        scheduler = make_scheduler(
            running=[running],
            waiting=[first, second],
            max_num_batched_tokens=2,
        )

        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [first])
        self.assertEqual(first.num_scheduled_tokens, 2)
        self.assertEqual(list(scheduler.waiting), [first, second])
        self.assertEqual(list(scheduler.running), [running])
