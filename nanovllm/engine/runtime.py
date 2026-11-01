"""One process-group owner; model runners only borrow it in the TP=1 pair."""

import torch
import torch.distributed as dist


class DeviceRuntime:
    def __init__(self, rank: int, world_size: int):
        if dist.is_initialized():
            raise RuntimeError("An existing process group needs its own explicit owner")
        self.rank = rank
        self.world_size = world_size
        self._clients = set()
        self.closed = False
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", "tcp://localhost:2333",
                                world_size=world_size, rank=rank)

    def attach(self, runner):
        if self.closed:
            raise RuntimeError("Runtime is closed")
        if runner.rank != self.rank or runner.world_size != self.world_size:
            raise ValueError("Runner and runtime must use the same rank/world size")
        self._clients.add(runner)

    def detach(self, runner):
        self._clients.discard(runner)

    def close(self):
        if self.closed:
            return
        if self._clients:
            raise RuntimeError("Release all model runners before closing the runtime")
        dist.destroy_process_group()
        self.closed = True
