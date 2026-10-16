"""The GPU-check harness itself: CPU-testable parts, plus real-GPU tests opt-in via HCFG_GPU_TESTS=1."""

import os
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from hcfg.gpu_check import Report, classify_kernels, finetuner_config, run_gpu_check, synthetic_images

needs_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and os.environ.get("HCFG_GPU_TESTS") == "1"),
    reason="real-SDXL GPU test: set HCFG_GPU_TESTS=1 on the GPU server",
)


def _evt(key, us):
    return SimpleNamespace(key=key, self_device_time_total=us)


def test_classify_kernels():
    events = [
        _evt("ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_tn", 800.0),
        _evt("sm80_xmma_fprop_implicit_gemm_bf16bf16_bf16f32", 100.0),
        _evt("pytorch_flash::flash_fwd_kernel", 50.0),
        _evt("ampere_sgemm_128x64_nn", 50.0),  # fp32 SIMT: no tensor cores
        _evt("void at::native::elementwise_kernel", 999.0),  # not a math kernel: ignored
    ]
    stats = classify_kernels(events)
    assert stats["tensor_core_share"] == pytest.approx(0.95)
    assert stats["top_non_tensor_core"][0][0].startswith("ampere_sgemm")


def test_synthetic_images(tmp_path):
    directory = synthetic_images(tmp_path / "s", count=3, size=(128, 64))
    images = sorted(directory.glob("*.png"))
    assert len(images) == 3 and Image.open(images[0]).size == (128, 64)


def test_finetuner_configs_for_check(tmp_path):
    for name in ("vanilla_lora", "ca_lora_style", "ca_lora_style_dropout", "dreambooth_lora", "textual_inversion"):
        config = finetuner_config(name, tmp_path, tmp_path, "train.max_steps=5")
        assert config.train.max_steps == 5 and config.name == name
    assert finetuner_config("ca_lora_style", tmp_path, tmp_path).finetuner.selection.endswith("ref_style_t81.json")


def test_report_status():
    report = Report()
    report.add("s", "a", True)
    report.add("s", "b", False, warn=True)
    report.add("s", "c", None)
    assert [c.status for c in report.checks] == ["PASS", "WARN", "SKIP"] and not report.failed
    report.add("s", "d", False)
    assert report.failed


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-GPU path")
def test_run_on_cpu_aborts_cleanly(tmp_path):
    report = run_gpu_check(out_dir=tmp_path, stages=["env"])
    assert report.failed and (tmp_path / "report.json").exists()


@needs_gpu
def test_gpu_exact_and_tensorcore(tmp_path):
    report = run_gpu_check(out_dir=tmp_path, stages=["env", "exact", "tensorcore"])
    assert not report.failed, [c for c in report.checks if c.status == "FAIL"]
