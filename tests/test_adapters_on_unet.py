"""Adapters on the (tiny) diffusers UNet — Implementation detail §8 tests 2 and 3."""

import pytest
import torch
from conftest import randomize_lora

from hcfg.config import load_config
from hcfg.models.finetuners import attach_finetuner, expand_ti_prompt, load_finetuner, save_finetuner
from hcfg.models.heads import enumerate_projections, plans_from_units
from hcfg.models.lora import adapter_scale, disable_adapter, iter_lora, merge_lora, unmerge_lora
from hcfg.models.sdxl import load_sdxl
from hcfg.utils.io import save_json


def _eps(model, cond, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(cond.batch_size, 4, model.latent_size, model.latent_size, generator=g)
    t = torch.tensor([500, 81][: cond.batch_size])
    with torch.no_grad():
        return model.predict(z, t, cond)


def _selection(tmp_path, model):
    projections = enumerate_projections(model.unet)
    names = [
        f"{projections[0].name}@h1",
        f"{projections[0].name}@h3",
        f"{projections[3].name}@h0",
        f"{projections[5].name}@h2",
    ]
    path = tmp_path / "sel.json"
    save_json({"granularity": "head", "names": names}, path)
    return path, names


def _config(kind, tmp_path, model):
    overrides = [f"output_dir={tmp_path.as_posix()}/runs", f"finetuner.kind={kind}"]
    if kind == "ca_lora":
        path, _ = _selection(tmp_path, model)
        overrides.append(f"finetuner.selection={path.as_posix()}")
    if kind == "dreambooth_lora":
        overrides.append("prompts.set=dreambooth")
    if kind == "textual_inversion":
        overrides.append("prompts.set=textual_inversion")
    return load_config("configs/tiny_cpu.yaml", overrides)


@pytest.mark.parametrize("kind", ["lora", "ca_lora", "dreambooth_lora"])
def test_lora_scale_zero_equals_base(tiny_model, tmp_path, kind):
    """§8-3: LoRA scale 0 -> ε(FT, ·) == ε(Base, ·) exactly, for c and ∅."""
    model = tiny_model
    config = _config(kind, tmp_path, model)
    cond = model.encode_prompts(["a street", "a street at night"])
    null = model.null_cond(2)
    base_c, base_null = _eps(model, cond), _eps(model, null)
    attach_finetuner(model, config)
    randomize_lora(model.unet)
    ft_c = _eps(model, cond)
    assert not torch.equal(ft_c, base_c)  # the adapter does something
    with disable_adapter(model.unet):
        assert torch.equal(_eps(model, cond), base_c)
        assert torch.equal(_eps(model, null), base_null)
    with adapter_scale(model.unet, 0.0):
        pass
    assert torch.equal(_eps(model, cond), ft_c)  # scale restored


def test_ca_lora_only_selected_heads(tiny_model, tmp_path):
    model = tiny_model
    config = _config("ca_lora", tmp_path, model)
    projections = enumerate_projections(model.unet)
    params = attach_finetuner(model, config)
    modules = dict(iter_lora(model.unet))
    assert len(modules) == 3
    q = modules[projections[0].name]
    assert q.heads == (1, 3) and q.lora_type == "B" and q.lora_B.shape[0] == 2 * q.head_dim
    out = modules[projections[3].name]
    assert out.heads == (0,) and out.lora_type == "A" and out.lora_A.shape[1] == out.head_dim
    # nothing but the LoRA factors is trainable
    trainable = {id(p) for p in model.unet.parameters() if p.requires_grad}
    assert trainable == {id(p) for p in params}


def test_plans_all_heads_becomes_full(tiny_model):
    projections = enumerate_projections(tiny_model.unet)
    p = projections[0]
    plans = plans_from_units(projections, [f"{p.name}@h{h}" for h in range(p.num_heads)])
    assert plans[0].heads is None


def test_reference_selection_parses():
    """The reference repo's selection file uses the same unit naming."""
    from hcfg.models.finetuners import load_selection_units
    from hcfg.models.heads import parse_unit_name

    units = load_selection_units("configs/selections/ref_style_t81.json")
    assert len(units) == 208
    module, head = parse_unit_name(units[0])
    assert module.endswith("attn2.to_k") and head is not None


def test_merge_equals_unmerged_on_unet(tiny_model, tmp_path):
    model = tiny_model
    attach_finetuner(model, _config("ca_lora", tmp_path, model))
    randomize_lora(model.unet)
    cond = model.encode_prompts(["a street", "x"])
    before = _eps(model, cond)
    merge_lora(model.unet)
    torch.testing.assert_close(_eps(model, cond), before, rtol=1e-4, atol=1e-5)
    unmerge_lora(model.unet)
    torch.testing.assert_close(_eps(model, cond), before, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("uncond", ["zeros", "empty"])
def test_textual_inversion_diff_null_is_zero(tmp_path, uncond):
    """§8-2: Textual Inversion -> Diff_Null == 0 exactly (∅ never contains S*)."""
    config = load_config(
        "configs/tiny_cpu.yaml",
        [
            f"output_dir={tmp_path.as_posix()}/runs",
            "finetuner.kind=textual_inversion",
            "prompts.set=textual_inversion",
            f"model.uncond={uncond}",
        ],
    )
    model = load_sdxl(config.model)
    null_base = model.null_cond(2)
    eps_base_null = _eps(model, null_base)
    plain = "Photorealistic first-person urban street view"
    plain_base = model.encode_prompts([plain])
    params = attach_finetuner(model, config)
    with torch.no_grad():
        for p in params:
            p.add_(torch.randn_like(p))  # "trained" vectors
    null_ft = model.null_cond(2)
    assert torch.equal(null_ft.prompt_embeds, null_base.prompt_embeds)
    assert torch.equal(null_ft.pooled, null_base.pooled)
    assert torch.equal(_eps(model, null_ft), eps_base_null)  # Diff_Null == 0
    assert torch.equal(model.encode_prompts([plain]).prompt_embeds, plain_base.prompt_embeds)
    s_prompt = expand_ti_prompt(plain + " with S* style", config)
    assert "<s*>" in s_prompt
    token_id = model.tokenizers[0].convert_tokens_to_ids("<s*>")
    assert token_id in model.tokenizers[0](s_prompt).input_ids
    assert not torch.equal(model.encode_prompts([s_prompt]).prompt_embeds, plain_base.prompt_embeds)


def test_ti_gradient_reaches_only_new_vectors(tmp_path):
    config = load_config(
        "configs/tiny_cpu.yaml",
        [f"output_dir={tmp_path.as_posix()}/runs", "finetuner.kind=textual_inversion", "prompts.set=textual_inversion"],
    )
    model = load_sdxl(config.model)
    params = attach_finetuner(model, config)
    prompt = expand_ti_prompt("street with S* style", config)
    cond = model.encode_prompts([prompt], grad=True)
    (cond.prompt_embeds.sum() + cond.pooled.sum()).backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in params)
    others = [p for e in model.text_encoders for p in e.parameters() if p.requires_grad and all(p is not q for q in params)]
    assert others == []


@pytest.mark.parametrize("kind", ["lora", "ca_lora", "textual_inversion"])
def test_checkpoint_roundtrip(tiny_config, tmp_path, kind):
    config = _config(kind, tmp_path, load_sdxl(tiny_config.model))
    model = load_sdxl(config.model)
    params = attach_finetuner(model, config)
    with torch.no_grad():
        for p in params:
            p.add_(0.1 * torch.randn_like(p))
    prompt = "Photorealistic first-person urban street view with S* style"
    if kind == "textual_inversion":
        prompt = expand_ti_prompt(prompt, config)
    expected = _eps(model, model.encode_prompts([prompt, prompt]))
    path = tmp_path / "ck.safetensors"
    save_finetuner(model, config, path)

    fresh = load_sdxl(config.model)
    meta = load_finetuner(fresh, path)
    assert meta["finetuner"] == kind
    assert torch.equal(_eps(fresh, fresh.encode_prompts([prompt, prompt])), expected)
