import numpy as np
import pytest
import torch
from PIL import Image

from hcfg.config import load_config
from hcfg.models.sdxl import load_sdxl


@pytest.fixture
def tiny_config(tmp_path):
    return load_config("configs/tiny_cpu.yaml", [f"output_dir={tmp_path.as_posix()}/runs"])


@pytest.fixture
def tiny_model(tiny_config):
    return load_sdxl(tiny_config.model)


@pytest.fixture
def image_dir(tmp_path):
    """Ten 64x32 landscape images (Cityscapes aspect ratio 2:1)."""
    directory = tmp_path / "images"
    directory.mkdir()
    rng = np.random.default_rng(0)
    for i in range(10):
        array = rng.integers(0, 255, size=(32, 64, 3), dtype=np.uint8)
        Image.fromarray(array).save(directory / f"img_{i:02d}.png")
    return directory


@pytest.fixture
def heldout_data(tmp_path):
    """14 images: the 10 listed in ``train10.txt`` are "training", 4 are held-out candidates."""
    directory = tmp_path / "cityscapes"
    directory.mkdir()
    rng = np.random.default_rng(1)
    for i in range(14):
        array = rng.integers(0, 255, size=(32, 64, 3), dtype=np.uint8)
        Image.fromarray(array).save(directory / f"img_{i:02d}.png")
    image_list = tmp_path / "train10.txt"
    image_list.write_text("".join(f"img_{i:02d}.png\n" for i in range(10)), encoding="utf-8")
    return directory, image_list


def randomize_lora(model, seed=0):
    """Give every LoRA a non-zero B so adapters actually change outputs."""
    from hcfg.models.lora import iter_lora

    g = torch.Generator().manual_seed(seed)
    for _, m in iter_lora(model):
        m.lora_B.data.copy_(torch.randn(m.lora_B.shape, generator=g) * 0.1)
