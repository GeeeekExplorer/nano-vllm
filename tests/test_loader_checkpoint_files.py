from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
from tempfile import TemporaryDirectory
from unittest import TestCase, main

from torch import nn
import torch
from safetensors.torch import save_file


ROOT = Path(__file__).parents[1]
SPEC = spec_from_file_location(
    "loader_checkpoint_files",
    ROOT / "nanovllm" / "utils" / "loader.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load checkpoint loader")
LOADER = module_from_spec(SPEC)
SPEC.loader.exec_module(LOADER)


def write_sharded_checkpoint(root: Path, *, complete: bool) -> None:
    names = [f"model-{index:05d}-of-00002.safetensors" for index in (1, 2)]
    tensors = (
        {"0.weight": torch.ones(2, 2)},
        {"1.weight": torch.full((2, 2), 2.0)},
    )
    save_file(tensors[0], root / names[0])
    if complete:
        save_file(tensors[1], root / names[1])
    weight_map = {
        weight: names[index]
        for index, shard in enumerate(tensors)
        for weight in shard
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )


class LoaderCheckpointFilesTest(TestCase):
    def test_empty_model_directory_is_rejected(self):
        with TemporaryDirectory() as directory, self.assertRaisesRegex(
            FileNotFoundError, "no safetensors files found"
        ):
            LOADER.load_model(nn.Linear(2, 2, bias=False), directory)

    def test_safetensors_directory_is_not_treated_as_checkpoint(self):
        with TemporaryDirectory() as directory:
            (Path(directory) / "accidental.safetensors").mkdir()
            with self.assertRaisesRegex(
                FileNotFoundError, "no safetensors files found"
            ):
                LOADER.load_model(nn.Linear(2, 2, bias=False), directory)

    def test_missing_indexed_shard_is_rejected_before_partial_loading(self):
        with TemporaryDirectory() as directory:
            write_sharded_checkpoint(Path(directory), complete=False)
            model = nn.Sequential(
                *(nn.Linear(2, 2, bias=False) for _ in range(2))
            )
            untouched = model[0].weight.detach().clone()
            with self.assertRaisesRegex(
                FileNotFoundError, "model-00002-of-00002.safetensors"
            ):
                LOADER.load_model(model, directory)
            torch.testing.assert_close(model[0].weight, untouched)

    def test_complete_indexed_checkpoint_loads_all_shards(self):
        with TemporaryDirectory() as directory:
            write_sharded_checkpoint(Path(directory), complete=True)
            model = nn.Sequential(
                *(nn.Linear(2, 2, bias=False) for _ in range(2))
            )
            LOADER.load_model(model, directory)
            torch.testing.assert_close(model[0].weight, torch.ones(2, 2))
            torch.testing.assert_close(model[1].weight, torch.full((2, 2), 2.0))


if __name__ == "__main__":
    main()
