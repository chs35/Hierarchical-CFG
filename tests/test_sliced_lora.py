"""Sliced LoRA (CA-LoRA type A / type B): equivalence, locality, exact on/off, merge."""

import pytest
import torch
from torch import nn

from hcfg.models.lora import LoRALinear, head_slice_index


def _layer(in_f=12, out_f=12, seed=0):
    torch.manual_seed(seed)
    return nn.Linear(in_f, out_f)


def _rand_lora(m):
    torch.manual_seed(1)
    m.lora_A.data.normal_()
    m.lora_B.data.normal_()


def test_head_slice_index():
    assert head_slice_index([0, 2], 3).tolist() == [0, 1, 2, 6, 7, 8]


@pytest.mark.parametrize("slice_dim", ["row", "col"])
def test_all_heads_equals_full_lora(slice_dim):
    base = _layer()
    sliced = LoRALinear(base, rank=4, alpha=8, heads=[0, 1, 2], num_heads=3, head_dim=4, slice_dim=slice_dim)
    assert not sliced.sliced  # all heads -> plain LoRA
    _rand_lora(sliced)
    x = torch.randn(2, 5, 12)
    expected = base(x) + (x @ sliced.lora_A.T @ sliced.lora_B.T) * sliced.scaling
    torch.testing.assert_close(sliced(x), expected)


def test_type_b_writes_only_selected_rows():
    base = _layer()
    m = LoRALinear(base, rank=4, alpha=4, heads=[1], num_heads=3, head_dim=4, slice_dim="row")
    assert m.lora_type == "B" and m.lora_A.shape == (4, 12) and m.lora_B.shape == (4, 4)
    _rand_lora(m)
    x = torch.randn(3, 12)
    diff = m(x) - base(x)
    assert torch.all(diff[:, :4] == 0) and torch.all(diff[:, 8:] == 0)
    assert diff[:, 4:8].abs().sum() > 0
    dense = m.full_delta_weight()
    assert torch.all(dense[:4] == 0) and torch.all(dense[8:] == 0)
    torch.testing.assert_close(diff, x @ dense.T)


def test_type_a_reads_only_selected_cols():
    base = _layer()
    m = LoRALinear(base, rank=4, alpha=4, heads=[0, 2], num_heads=3, head_dim=4, slice_dim="col")
    assert m.lora_type == "A" and m.lora_A.shape == (4, 8) and m.lora_B.shape == (12, 4)
    _rand_lora(m)
    x = torch.randn(3, 12)
    x2 = x.clone()
    x2[:, 4:8] = torch.randn(3, 4)  # unselected head's input must not matter for the delta
    torch.testing.assert_close(m(x) - base(x), m(x2) - base(x2))
    dense = m.full_delta_weight()
    assert torch.all(dense[:, 4:8] == 0)
    torch.testing.assert_close(m(x) - base(x), x @ dense.T)


@pytest.mark.parametrize("slice_dim", ["row", "col"])
def test_trainable_params_are_exactly_the_slice(slice_dim):
    m = LoRALinear(_layer(), rank=4, alpha=4, heads=[2], num_heads=3, head_dim=4, slice_dim=slice_dim)
    x = torch.randn(3, 12)
    m(x).sum().backward()
    assert m.base_layer.weight.grad is None or not m.base_layer.weight.requires_grad
    assert m.lora_A.numel() + m.lora_B.numel() == 4 * 12 + 4 * 4


@pytest.mark.parametrize("slice_dim", ["row", "col"])
def test_scale_zero_is_bit_exact_base(slice_dim):
    base = _layer()
    m = LoRALinear(base, rank=4, alpha=4, heads=[1], num_heads=3, head_dim=4, slice_dim=slice_dim)
    _rand_lora(m)
    m.scale = 0.0
    x = torch.randn(3, 12)
    assert torch.equal(m(x), base(x))


@pytest.mark.parametrize("slice_dim", ["row", "col"])
def test_merge_matches_unmerged(slice_dim):
    base = _layer()
    m = LoRALinear(base, rank=4, alpha=4, heads=[0, 2], num_heads=3, head_dim=4, slice_dim=slice_dim)
    _rand_lora(m)
    x = torch.randn(3, 12)
    before = m(x)
    w0 = base.weight.detach().clone()
    m.merge()
    torch.testing.assert_close(m(x), before, rtol=1e-5, atol=1e-5)
    m.unmerge()
    torch.testing.assert_close(base.weight, w0, rtol=1e-6, atol=1e-6)


def test_bf16_base_with_fp32_lora():
    base = _layer().to(torch.bfloat16)
    m = LoRALinear(base, rank=4, alpha=4, heads=[1], num_heads=3, head_dim=4, slice_dim="row")
    assert m.lora_A.dtype == torch.float32
    out = m(torch.randn(2, 12, dtype=torch.bfloat16))
    assert out.dtype == torch.bfloat16
