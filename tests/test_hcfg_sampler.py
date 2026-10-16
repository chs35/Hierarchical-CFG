"""Hierarchical-CFG sampler and the (w1, w2) grid on the tiny model (Implementation detail §3, §8-1)."""

import json
import math
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import randomize_lora
from PIL import Image

from hcfg.config import load_config
from hcfg.data.prompts import load_prompt_set
from hcfg.diagnostics import storage
from hcfg.generation import sweep as sweep_mod
from hcfg.generation.generate import generate_images, generate_latents, image_seed
from hcfg.generation.hierarchical import (
    HCFGConds,
    WeightSchedule,
    combine,
    combine_split,
    hcfg_sample,
    split_diff_null,
    split_par_default,
)
from hcfg.models.finetuners import attach_finetuner, save_finetuner
from hcfg.models.lora import adapter_scale
from hcfg.models.sdxl import load_sdxl

DiskUsage = namedtuple("DiskUsage", "total used free")
SEEDS = [image_seed(0, 0), image_seed(0, 1)]


@pytest.fixture(autouse=True)
def persistent_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv(storage.ENV_PERSISTENT_ROOT, str(tmp_path))


@pytest.fixture
def lora_model(tiny_config):
    model = load_sdxl(tiny_config.model)
    attach_finetuner(model, tiny_config)
    randomize_lora(model.unet)
    return model


def _conds(model, base=False):
    c, cr = "a street", "a street, in snowy weather"
    extra = dict(base_domain=model.encode_prompts([c]), base_refine=model.encode_prompts([cr])) if base else {}
    return HCFGConds(null=model.null_cond(1), domain=model.encode_prompts([c]), refine=model.encode_prompts([cr]), **extra)


def _sample(model, conds, variant, w1, w2, **kw):
    return hcfg_sample(model, conds, SEEDS, variant=variant, w1=WeightSchedule(w1), w2=WeightSchedule(w2),
                       num_inference_steps=kw.pop("steps", 3), **kw)


# --------------------------------------------------------------------------------------
# Formulas and schedules
# --------------------------------------------------------------------------------------


def test_combine_formulas():
    g = torch.Generator().manual_seed(0)
    e = {k: torch.randn(2, 4, 8, 8, generator=g) for k in ("ft_null", "ft_domain", "ft_refine", "base_null", "base_domain", "base_refine")}
    f0, fc, fr, b0, bc, br = (e[k] for k in ("ft_null", "ft_domain", "ft_refine", "base_null", "base_domain", "base_refine"))
    torch.testing.assert_close(combine("basic", 3.0, 3.0, e), f0 + 3 * (fr - f0))  # w1 = w2: standard CFG on c_refine
    torch.testing.assert_close(combine("basic", 2.0, 5.0, e), f0 + 2 * (fr - fc) + 5 * (fc - f0))
    torch.testing.assert_close(combine("alt_domain", 2.0, 5.0, e), b0 + 2 * (fr - fc) + 5 * (fc - b0))
    torch.testing.assert_close(combine("alt_domain", 4.0, 4.0, e), b0 + 4 * (fr - b0))  # Unconditional Priors Matter
    torch.testing.assert_close(combine("alt_refine", 2.0, 5.0, e), f0 + 2 * (br - bc) + 5 * (fc - f0))
    # G_domain,hybrid = G_domain + Diff_Null
    torch.testing.assert_close(combine("alt_domain", 2.0, 5.0, e) - combine("basic", 2.0, 5.0, e), (1 - 5) * (b0 - f0))
    with pytest.raises(ValueError):
        combine("other", 1.0, 1.0, e)


def test_weight_schedules():
    assert WeightSchedule(7.0)(500, 961) == 7.0
    lin, quad = WeightSchedule(7.0, "linear", 1.0), WeightSchedule(7.0, "quadratic", 1.0)
    assert lin(961, 961) == quad(961, 961) == 7.0  # grid value at the first step
    assert lin(0, 961) == quad(0, 961) == 1.0
    assert math.isclose(lin(480.5, 961), 4.0) and math.isclose(quad(480.5, 961), 1.0 + 6.0 * 0.25)
    with pytest.raises(ValueError):
        WeightSchedule(1.0, "cosine")


# --------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("w", [1.0, 4.0])
def test_basic_equal_weights_is_standard_cfg(lora_model, w):
    """§8-1: w1 = w2 = w -> the latents of standard CFG with c_refine at the same seeds."""
    model = lora_model
    conds = _conds(model)
    expected = generate_latents(model, conds.refine, conds.null, SEEDS, num_inference_steps=3, guidance_scale=w)
    assert torch.equal(_sample(model, conds, "basic", w, w), expected)  # reduced path == the pipeline, bit for bit
    full = _sample(model, conds, "basic", w, w, reduce_equal_weights=False)  # 3-branch formula, same maths
    torch.testing.assert_close(full, expected, rtol=1e-4, atol=1e-4)


def test_weights_change_the_image(lora_model):
    conds = _conds(lora_model)
    a = _sample(lora_model, conds, "basic", 1.0, 7.0)
    b = _sample(lora_model, conds, "basic", 7.0, 1.0)
    assert not torch.allclose(a, b)


@pytest.mark.parametrize("w1,w2", [(2.0, 5.0), (4.0, 4.0)])
def test_alternatives_equal_basic_without_adapter(lora_model, w1, w2):
    """With the adapter off, FT == Base, so both alternatives reduce to the basic form."""
    model = lora_model
    conds = _conds(model, base=True)
    with adapter_scale(model.unet, 0.0):
        basic = _sample(model, conds, "basic", w1, w2, reduce_equal_weights=False)
        for variant in ("alt_domain", "alt_refine"):
            torch.testing.assert_close(_sample(model, conds, variant, w1, w2), basic, rtol=1e-4, atol=1e-4)
    # with the adapter on they differ (Diff_Null != 0, G_refine != G_refine^Base)
    assert not torch.allclose(_sample(model, conds, "alt_domain", w1, w2), _sample(model, conds, "basic", w1, w2))


def test_alt_refine_needs_base_conds(lora_model):
    with pytest.raises(ValueError, match="Base-prompt"):
        _sample(lora_model, _conds(lora_model), "alt_refine", 2.0, 5.0)


def test_schedule_sampling_runs(lora_model):
    conds = _conds(lora_model)
    out = hcfg_sample(lora_model, conds, SEEDS, variant="basic", w1=WeightSchedule(7.0, "linear"),
                      w2=WeightSchedule(2.0, "quadratic"), num_inference_steps=3)
    assert out.shape == (2, 4, lora_model.latent_size, lora_model.latent_size) and torch.isfinite(out).all()


# --------------------------------------------------------------------------------------
# Grid runner
# --------------------------------------------------------------------------------------


def _run(tmp_path, kind="lora", *overrides):
    extra = {"lora": ["finetuner.kind=lora"],
             "textual_inversion": ["finetuner.kind=textual_inversion", "prompts.set=textual_inversion"]}[kind]
    config = load_config("configs/tiny_cpu.yaml", [f"output_dir={tmp_path.as_posix()}/runs", "name=ft", *extra, *overrides])
    model = load_sdxl(config.model)
    params = attach_finetuner(model, config)
    with torch.no_grad():
        for p in params:
            p.add_(0.1 * torch.randn(p.shape, generator=torch.Generator().manual_seed(0)))
    ckpt = config.run_dir / "finetuner-final.safetensors"
    save_finetuner(model, config, ckpt)
    return config, ckpt


def _pixels(path):
    return np.asarray(Image.open(path), dtype=np.int16)


def test_sweep_layout_manifest_and_diagonal(tmp_path):
    config, ckpt = _run(tmp_path)
    root = sweep_mod.run_sweep(config, checkpoint=ckpt)
    assert root == config.run_dir / "hcfg" / "basic"
    cells = sorted(p.name for p in root.iterdir() if p.is_dir())
    assert cells == ["w1_1_w2_1", "w1_1_w2_4", "w1_4_w2_1", "w1_4_w2_4"]
    assert len(list(root.rglob("*.png"))) == 4 * 1 * 1 * 2
    assert not list(root.rglob("*.partial"))
    manifest = json.loads((root / "w1_1_w2_4" / "manifest.json").read_text(encoding="utf-8"))
    assert (manifest["w1"], manifest["w2"], manifest["variant"]) == (1.0, 4.0, "basic")
    entry = manifest["prompts"]["snowy"]
    assert entry["prompt"] == "Photorealistic first-person urban street view, in snowy weather"
    assert [r["path"] for r in entry["records"]] == ["snowy/seed0/0000.png", "snowy/seed0/0001.png"]
    assert json.loads((root / "sweep.json").read_text(encoding="utf-8"))["images_total"] == 8

    # the w1 = w2 diagonal is standard CFG: same pixels as generate_images at that w
    model = load_sdxl(config.model)
    from hcfg.models.finetuners import load_finetuner

    load_finetuner(model, ckpt)
    ref = generate_images(model, entry["prompt"], num_images=2, seed=0, out_dir=tmp_path / "ref",
                          num_inference_steps=config.generate.num_inference_steps, guidance_scale=4.0, batch_size=2)
    for r in ref:
        assert np.array_equal(_pixels(r["path"]), _pixels(root / "w1_4_w2_4" / "snowy" / "seed0" / f"{r['index']:04d}.png"))

    # a cell directory is a samples directory for the evaluator
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_pipeline_e2e import FakeDetector, FakeEmbedder

    from hcfg.eval.evaluate import evaluate_samples

    result = evaluate_samples(config, root / "w1_1_w2_4", embedder=FakeEmbedder(), detector=FakeDetector())
    assert result["per_prompt"]["snowy"]["num_images"] == 2
    assert (root / "eval_w1_1_w2_4.json").exists()


def test_sweep_resumes_and_refuses_changed_settings(tmp_path, monkeypatch):
    config, ckpt = _run(tmp_path)
    root = sweep_mod.run_sweep(config, checkpoint=ckpt)
    victim = root / "w1_4_w2_1" / "snowy" / "seed0" / "0001.png"
    before = {p: p.read_bytes() for p in root.rglob("*.png")}
    victim.unlink()
    calls = []
    original = sweep_mod.hcfg_sample
    monkeypatch.setattr(sweep_mod, "hcfg_sample", lambda *a, **k: calls.append(k) or original(*a, **k))
    sweep_mod.run_sweep(config, checkpoint=ckpt)
    assert len(calls) == 1 and (calls[0]["w1"].start, calls[0]["w2"].start) == (4.0, 1.0)
    assert {p: p.read_bytes() for p in root.rglob("*.png")} == before  # same bytes, whole batch regenerated
    calls.clear()
    sweep_mod.run_sweep(config, checkpoint=ckpt)
    assert calls == []  # complete: nothing to do (the model is not even loaded)

    changed = load_config("configs/tiny_cpu.yaml", [f"output_dir={tmp_path.as_posix()}/runs", "name=ft",
                                                     "finetuner.kind=lora", "hcfg.images_per_prompt=3"])
    with pytest.raises(ValueError, match="different settings"):
        sweep_mod.run_sweep(changed, checkpoint=ckpt)


def test_sweep_checks_disk_and_persistence_before_loading(tmp_path, monkeypatch):
    config, ckpt = _run(tmp_path)

    def must_not_load(*a, **k):
        raise AssertionError("model loaded")

    monkeypatch.setattr(sweep_mod, "load_sdxl", must_not_load)
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda p: DiskUsage(0, 0, 10))
    with pytest.raises(storage.InsufficientDiskSpace):
        sweep_mod.run_sweep(config, checkpoint=ckpt)
    assert not (config.run_dir / "hcfg").exists()
    monkeypatch.setenv(storage.ENV_PERSISTENT_ROOT, str(tmp_path / "volume"))
    with pytest.raises(storage.NotPersistentError):
        sweep_mod.run_sweep(config, checkpoint=ckpt)


def test_sweep_estimate_for_the_real_grid():
    config = load_config("configs/vanilla_lora.yaml")
    images = len(sweep_mod.planned_images(config, sweep_mod.sweep_root(config)))
    assert images == 25 * 5 * 20  # 5x5 cells x 5 c_refine x 20 images x 1 seed
    est = sweep_mod.estimate_sweep_bytes(config, images)
    assert est["images"] == 2500 * 2 * 1024**2


def test_sweep_textual_inversion_prompts_and_alt_refine(tmp_path):
    config, ckpt = _run(tmp_path, "textual_inversion", "hcfg.variant=alt_refine", "hcfg.base_prompt_set=null",
                        "hcfg.w1=[2]", "hcfg.w2=[4]")
    with pytest.raises(ValueError, match="placeholder"):
        sweep_mod.run_sweep(config, checkpoint=ckpt)
    ok = load_config("configs/tiny_cpu.yaml", [f"output_dir={tmp_path.as_posix()}/runs", "name=ft",
                                                "finetuner.kind=textual_inversion", "prompts.set=textual_inversion",
                                                "hcfg.variant=alt_refine", "hcfg.base_prompt_set=default",
                                                "hcfg.w1=[2]", "hcfg.w2=[4]"])
    root = sweep_mod.run_sweep(ok, checkpoint=ckpt, out_name="alt_refine_ok")
    entry = json.loads((root / "w1_2_w2_4" / "manifest.json").read_text(encoding="utf-8"))["prompts"]["snowy"]
    assert "<s*>" in entry["model_prompt"] and "S*" in entry["prompt"]
    assert len(entry["records"]) == 2


def test_hcfg_config():
    assert load_config("configs/textual_inversion.yaml").hcfg.base_prompt_set == "default"
    h = load_config("configs/vanilla_lora.yaml").hcfg
    assert (h.variant, h.w1, h.w2, h.images_per_prompt) == ("basic", [1, 2, 4, 7, 10], [1, 2, 4, 7, 10], 20)
    with pytest.raises(ValueError, match="merge_lora"):
        load_config("configs/vanilla_lora.yaml", ["hcfg.merge_lora=true", "hcfg.variant=alt_domain"])
    with pytest.raises(ValueError, match="schedule"):
        load_config("configs/vanilla_lora.yaml", ["hcfg.schedule=cosine"])
    prompts = load_prompt_set(load_config("configs/vanilla_lora.yaml").prompts)
    assert all(prompts.refine(k).startswith(prompts.domain) for k in h.keys)


# --------------------------------------------------------------------------------------
# Diff_Null split (report §7)
# --------------------------------------------------------------------------------------


def _eps(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {k: torch.randn(3, 4, 8, 8, generator=g) for k in ("ft_null", "ft_domain", "ft_refine", "base_null", "base_domain")}


def test_split_diff_null_projection():
    e = _eps()
    diff, r = e["ft_null"] - e["base_null"], e["ft_domain"] - e["base_domain"]
    par, perp = split_diff_null(diff, r)
    torch.testing.assert_close(par + perp, diff)
    dims = (1, 2, 3)
    assert (perp * r).sum(dims).abs().max() < 1e-4 * (r * r).sum(dims).min()  # Diff_⊥ ⟂ r, per sample
    cos = torch.nn.functional.cosine_similarity(par.flatten(1), r.flatten(1))
    torch.testing.assert_close(cos.abs(), torch.ones(3))  # Diff_∥ ∥ r
    one_par, one_perp = split_diff_null(diff[1:2], r[1:2])  # one projection per sample, not per batch
    torch.testing.assert_close(one_par, par[1:2])
    torch.testing.assert_close(one_perp, perp[1:2])
    zero_par, zero_perp = split_diff_null(diff, torch.zeros_like(r))  # r = 0: nothing is parallel
    assert torch.equal(zero_par, torch.zeros_like(diff)) and torch.equal(zero_perp, diff)


def test_combine_split_generalises_basic_and_alt_domain():
    e = _eps()
    for variant, own in (("basic", 1.0), ("alt_domain", 5.0)):  # the variant's own a = λ: bit for bit
        assert torch.equal(combine_split(variant, 2.0, 5.0, own, own, e), combine(variant, 2.0, 5.0, e))
    # one family: the same (a, λ) from either anchor
    torch.testing.assert_close(combine_split("basic", 2.0, 5.0, 5.0, 5.0, e), combine("alt_domain", 2.0, 5.0, e))
    torch.testing.assert_close(combine_split("alt_domain", 2.0, 5.0, 1.0, 1.0, e), combine("basic", 2.0, 5.0, e))
    par, perp = split_diff_null(e["ft_null"] - e["base_null"], e["ft_domain"] - e["base_domain"])
    # basic with w3: λ = 1 − w3; alt_domain: w2·Diff_∥ + (w2 − w3)·Diff_⊥
    torch.testing.assert_close(combine_split("basic", 2.0, 5.0, 1.0, 0.0, e), combine("basic", 2.0, 5.0, e) - perp)
    torch.testing.assert_close(combine_split("alt_domain", 2.0, 5.0, 5.0, 1.0, e),
                               combine("alt_domain", 2.0, 5.0, e) - 4.0 * perp)
    assert split_par_default("basic", 4.0) == 1.0 and split_par_default("alt_domain", 4.0) == 4.0
    with pytest.raises(ValueError, match="basic / alt_domain"):
        split_par_default("alt_refine", 4.0)


@pytest.mark.parametrize("variant,w1,w2,lam", [("basic", 2.0, 5.0, 1.0), ("basic", 4.0, 4.0, 1.0),
                                               ("alt_domain", 2.0, 5.0, 5.0), ("alt_domain", 4.0, 4.0, 4.0)])
def test_split_with_the_variants_lambda_is_the_unsplit_sampler(lora_model, variant, w1, w2, lam):
    conds = _conds(lora_model, base=True)
    split = _sample(lora_model, conds, variant, w1, w2, split_lambda=lam)
    unsplit = _sample(lora_model, conds, variant, w1, w2, reduce_equal_weights=False)
    if variant == "basic":  # same FT batch layout and the same float32 formula: bit for bit
        assert torch.equal(split, unsplit)
    else:  # alt_domain's unsplit form batches 2 FT rows, not 3
        torch.testing.assert_close(split, unsplit, rtol=1e-4, atol=1e-4)


def test_split_lambda_and_par_change_the_image(lora_model):
    conds = _conds(lora_model, base=True)
    ref = _sample(lora_model, conds, "basic", 7.0, 4.0, split_lambda=1.0)
    for kw in ({"split_lambda": 0.0}, {"split_lambda": 5.0}, {"split_lambda": 1.0, "split_par": 4.0}):
        assert not torch.allclose(_sample(lora_model, conds, "basic", 7.0, 4.0, **kw), ref)
    # explicit a = the variant's own value is the default
    torch.testing.assert_close(_sample(lora_model, conds, "alt_domain", 7.0, 4.0, split_lambda=0.0, split_par=4.0),
                               _sample(lora_model, conds, "alt_domain", 7.0, 4.0, split_lambda=0.0), rtol=0, atol=0)


def test_split_without_adapter_ignores_lambda(lora_model):
    """Adapter off: Diff_Null = 0 (the Textual Inversion control), so λ and a change nothing."""
    conds = _conds(lora_model, base=True)
    with adapter_scale(lora_model.unet, 0.0):
        basic = _sample(lora_model, conds, "basic", 7.0, 4.0, reduce_equal_weights=False)
        for lam, par in ((0.0, None), (5.0, None), (-1.0, 3.0)):
            out = _sample(lora_model, conds, "alt_domain", 7.0, 4.0, split_lambda=lam, split_par=par)
            torch.testing.assert_close(out, basic, rtol=1e-4, atol=1e-4)


def test_split_schedule_and_rescale_run(lora_model):
    conds = _conds(lora_model, base=True)
    out = hcfg_sample(lora_model, conds, SEEDS, variant="alt_domain", w1=WeightSchedule(7.0, "linear"),
                      w2=WeightSchedule(4.0, "quadratic"), num_inference_steps=3, guidance_rescale=0.7, split_lambda=0.0)
    assert torch.isfinite(out).all()


def test_split_argument_errors(lora_model):
    with pytest.raises(ValueError, match="basic / alt_domain"):
        _sample(lora_model, _conds(lora_model, base=True), "alt_refine", 2.0, 5.0, split_lambda=0.0)
    with pytest.raises(ValueError, match="base_domain"):
        _sample(lora_model, _conds(lora_model), "basic", 2.0, 5.0, split_lambda=0.0)
    with pytest.raises(ValueError, match="split_par without"):
        _sample(lora_model, _conds(lora_model, base=True), "basic", 2.0, 5.0, split_par=1.0)


def test_split_config():
    cfg = lambda *o: load_config("configs/ca_lora_style.yaml", list(o))  # noqa: E731
    h = cfg().hcfg
    assert (h.split_lambda, h.split_par, h.cells, h.split) == (None, None, None, False)
    assert cfg("hcfg.split_lambda=[0, 1]").hcfg.split and cfg("hcfg.cells=[[7,4,0]]").hcfg.split
    assert not cfg("hcfg.cells=[[7,4]]").hcfg.split
    for overrides, match in [
        (["hcfg.cells=[[7,4,0]]", "hcfg.variant=alt_refine"], "basic or alt_domain"),
        (["hcfg.split_lambda=[0]", "hcfg.merge_lora=true"], "merge_lora"),
        (["hcfg.split_par=2"], "split_par needs"),
        (["hcfg.cells=[[7,4,0]]", "hcfg.split_lambda=[0]"], "either"),
        (["hcfg.cells=[[7,4,0],[7,4]]"], "all"),
        (["hcfg.cells=[]"], "non-empty"),
        (["hcfg.cells=[[7]]"], "all"),
        (["hcfg.split_lambda=[]"], "non-empty"),
    ]:
        with pytest.raises(ValueError, match=match):
            cfg(*overrides)


def test_split_cells_names_and_counts():
    cfg = lambda *o: load_config("configs/ca_lora_style.yaml", list(o))  # noqa: E731
    default = cfg()
    assert sweep_mod.sweep_name(default) == "basic" and sweep_mod.sweep_cells(default)[:2] == [(1, 1, None), (1, 2, None)]
    assert sweep_mod.cell_name(7, 4) == "w1_7_w2_4"  # unchanged names of the existing grids
    # report's experiment: basic (4 new cells) and alt_domain (5 new cells), 5 keys x 20 images each
    basic = cfg("hcfg.cells=[[7,2,0],[7,4,0],[7,4,-1],[7,4,5]]")
    alt = cfg("hcfg.variant=alt_domain", "hcfg.cells=[[7,2,0],[7,4,0],[7,2,1],[7,4,1],[7,4,8]]")
    assert sweep_mod.sweep_name(basic) == "basic_split" and sweep_mod.sweep_name(alt) == "alt_domain_split"
    assert [sweep_mod.cell_name(*c) for c in sweep_mod.sweep_cells(basic)] == [
        "w1_7_w2_2_lam_0", "w1_7_w2_4_lam_0", "w1_7_w2_4_lam_-1", "w1_7_w2_4_lam_5"]
    assert len(sweep_mod.planned_images(alt, sweep_mod.sweep_root(alt))) == 5 * 5 * 20
    grid = cfg("hcfg.w1=[7]", "hcfg.w2=[2,4]", "hcfg.split_lambda=[0,1]", "hcfg.split_par=1",
               "hcfg.schedule=linear", "hcfg.variant=alt_domain")
    assert sweep_mod.sweep_name(grid) == "alt_domain_split_a1_linear"
    assert sweep_mod.sweep_cells(grid) == [(7, 2, 0), (7, 2, 1), (7, 4, 0), (7, 4, 1)]
    pairs = cfg("hcfg.cells=[[7,2],[10,4]]")
    assert sweep_mod.sweep_name(pairs) == "basic" and sweep_mod.sweep_cells(pairs) == [(7, 2, None), (10, 4, None)]


def test_split_keeps_existing_sweep_settings(tmp_path):
    """Defaults add no key to sweep.json settings, so the existing sweeps still resume."""
    config = load_config("configs/vanilla_lora.yaml")
    h = sweep_mod.settings(config, tmp_path / "ckpt")["hcfg"]
    assert not set(sweep_mod.SPLIT_KEYS) & set(h)
    assert set(sweep_mod.settings(load_config("configs/vanilla_lora.yaml", ["hcfg.cells=[[7,4,0]]"]),
                                  tmp_path / "ckpt")["hcfg"]) >= {"cells"}
    from hcfg.config import PROJECT_ROOT

    found = sorted((PROJECT_ROOT / "runs").glob("*/hcfg/*/sweep.json"))
    if not found:
        pytest.skip("no existing sweeps in runs/")
    for path in found:
        stored = json.loads(path.read_text(encoding="utf-8"))
        run, variant = path.parts[-4], stored["settings"]["hcfg"]["variant"]
        if not (PROJECT_ROOT / "configs" / f"{run}.yaml").exists():
            continue
        config = load_config(f"configs/{run}.yaml", [f"hcfg.variant={variant}"])
        if sweep_mod.sweep_root(config) != path.parent:
            continue  # a sweep under another --out-name / with other overrides
        assert sweep_mod.settings(config, Path(stored["settings"]["checkpoint"])) == stored["settings"], path
        assert [sweep_mod.cell_name(a, b, lam) for a, b, lam in sweep_mod.sweep_cells(config)] == stored["cells"], path


def test_split_sweep(tmp_path):
    config, ckpt = _run(tmp_path, "lora", "hcfg.cells=[[4,1,0],[4,1,1],[4,4,-1]]")
    root = sweep_mod.run_sweep(config, checkpoint=ckpt)
    assert root == config.run_dir / "hcfg" / "basic_split"
    assert sorted(p.name for p in root.iterdir() if p.is_dir()) == ["w1_4_w2_1_lam_0", "w1_4_w2_1_lam_1", "w1_4_w2_4_lam_-1"]
    assert len(list(root.rglob("*.png"))) == 3 * 2 and not list(root.rglob("*.partial"))
    manifest = json.loads((root / "w1_4_w2_1_lam_0" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["split"] == {"lambda": 0, "par": None, "r": "diff_cond"} and (manifest["w1"], manifest["w2"]) == (4, 1)
    assert [r["path"] for r in manifest["prompts"]["snowy"]["records"]] == ["snowy/seed0/0000.png", "snowy/seed0/0001.png"]
    sweep = json.loads((root / "sweep.json").read_text(encoding="utf-8"))
    assert sweep["cells"] == ["w1_4_w2_1_lam_0", "w1_4_w2_1_lam_1", "w1_4_w2_4_lam_-1"]
    assert sweep["settings"]["hcfg"]["cells"] == [[4, 1, 0], [4, 1, 1], [4, 4, -1]]

    # λ = 1 is the basic cell of the plain grid (same z_T), up to float rounding of the formula
    plain = load_config("configs/tiny_cpu.yaml", [f"output_dir={tmp_path.as_posix()}/runs", "name=ft", "finetuner.kind=lora"])
    plain_root = sweep_mod.run_sweep(plain, checkpoint=ckpt)
    assert "split" not in json.loads((plain_root / "w1_4_w2_1" / "manifest.json").read_text(encoding="utf-8"))
    for i in range(2):
        same = _pixels(root / "w1_4_w2_1_lam_1" / "snowy" / "seed0" / f"{i:04d}.png")
        other = _pixels(plain_root / "w1_4_w2_1" / "snowy" / "seed0" / f"{i:04d}.png")
        assert np.abs(same - other).max() <= 1
    assert not np.array_equal(_pixels(root / "w1_4_w2_1_lam_0" / "snowy" / "seed0" / "0000.png"),
                              _pixels(root / "w1_4_w2_1_lam_1" / "snowy" / "seed0" / "0000.png"))
    # resume: nothing missing -> nothing regenerated; changed λ list -> refused
    assert sweep_mod.check_only([config])["images_to_generate"] == {"ft": 0}
    changed = load_config("configs/tiny_cpu.yaml", [f"output_dir={tmp_path.as_posix()}/runs", "name=ft",
                                                     "finetuner.kind=lora", "hcfg.cells=[[4,1,0]]"])
    with pytest.raises(ValueError, match="different settings"):
        sweep_mod.run_sweep(changed, checkpoint=ckpt)


def test_split_sweep_textual_inversion_needs_plain_base_prompts(tmp_path):
    config, ckpt = _run(tmp_path, "textual_inversion", "hcfg.base_prompt_set=null", "hcfg.cells=[[4,1,0]]")
    with pytest.raises(ValueError, match="placeholder"):
        sweep_mod.run_sweep(config, checkpoint=ckpt)
