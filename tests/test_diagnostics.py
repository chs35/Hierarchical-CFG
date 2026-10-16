"""Diagnostics (report §3; docs/Implementation detail.md §7) on the tiny model."""

import json
import math
import os
import tempfile
from collections import namedtuple
from pathlib import Path

import pytest
import torch
from conftest import randomize_lora

from hcfg.config import load_config
from hcfg.data.prompts import load_prompt_set
from hcfg.diagnostics import run as run_mod
from hcfg.diagnostics import storage
from hcfg.diagnostics.branches import NULL_KEY, encode_rows, predict_ft_base, predict_rows, row_texts
from hcfg.diagnostics.forward import select_heldout_images
from hcfg.diagnostics.metrics import cosine, diagnostic_tensors, mean_sq, sample_metrics
from hcfg.diagnostics.run import diagnose_run
from hcfg.diagnostics.summary import cross_sample_cos, summarize
from hcfg.diagnostics.trajectory import euler_cfg_sample, sampler_timesteps, two_row_eps_fn
from hcfg.generation.generate import generate_latents, image_seed
from hcfg.models.finetuners import attach_finetuner, save_finetuner
from hcfg.models.lora import adapter_scale
from hcfg.models.sdxl import load_sdxl

ALL_TENSORS = "diagnostics.save_tensors=[eps_ft_null, eps_base_null, g_domain, diff_null, g_refine, g_refine_base]"
DiskUsage = namedtuple("DiskUsage", "total used free")


@pytest.fixture(autouse=True)
def persistent_tmp(tmp_path, monkeypatch):
    """Hermetic persistence check: the test's tmp dir plays the persistent volume."""
    monkeypatch.setenv(storage.ENV_PERSISTENT_ROOT, str(tmp_path))
    return tmp_path


def _config(tmp_path, heldout_data, *overrides):
    image_dir, image_list = heldout_data
    return load_config(
        "configs/tiny_cpu.yaml",
        [
            f"output_dir={tmp_path.as_posix()}/runs",
            f"data.image_dir={image_dir.as_posix()}",
            f"data.image_list={image_list.as_posix()}",
            *overrides,
        ],
    )


KIND_OVERRIDES = {
    "lora": ["name=lora", "finetuner.kind=lora"],
    "textual_inversion": [
        "name=ti",
        "finetuner.kind=textual_inversion",
        "prompts.set=textual_inversion",
        "diagnostics.base_prompt_set=default",
    ],
}


def _checkpoint(tmp_path, heldout_data, kind="lora", *overrides):
    """A "trained" checkpoint (perturbed adapters) and the run's config."""
    config = _config(tmp_path, heldout_data, *KIND_OVERRIDES[kind], *overrides)
    model = load_sdxl(config.model)
    params = attach_finetuner(model, config)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in params:
            p.add_(0.1 * torch.randn(p.shape, generator=g))
    path = config.run_dir / "finetuner-final.safetensors"
    save_finetuner(model, config, path)
    return config, path


def _records(out_dir):
    return [json.loads(line) for line in (out_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()]


def _rows(model, texts=("a street", "a street, in snow")):
    return encode_rows(model, {"domain": texts[0], "snowy": texts[1]})


def _z(model, batch=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 4, model.latent_size, model.latent_size, generator=g)


# --------------------------------------------------------------------------------------
# Branches and metrics
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("max_rows", [1, 3, 100])
def test_predict_rows_matches_individual_predicts(tiny_model, max_rows):
    model = tiny_model
    rows = _rows(model)
    z, t = _z(model), torch.tensor([700, 81])
    eps = predict_rows(model, z, t, rows, max_rows=max_rows)
    assert list(eps) == [NULL_KEY, "domain", "snowy"]
    with torch.no_grad():
        for name, cond in rows.items():
            torch.testing.assert_close(eps[name], model.predict(z, t, cond.repeat(2)), rtol=1e-4, atol=1e-5)


def test_metric_identities():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(3, 4, 8, 8, generator=g)
    y = torch.randn(3, 4, 8, 8, generator=g)
    torch.testing.assert_close(cosine(x, x), torch.ones(3, dtype=torch.float64))
    torch.testing.assert_close(cosine(x, -x), -torch.ones(3, dtype=torch.float64))
    assert torch.isnan(cosine(x, torch.zeros_like(x))).all()
    torch.testing.assert_close(mean_sq(x), (x.double() ** 2).mean(dim=(1, 2, 3)))

    eps = lambda: {k: torch.randn(3, 4, 8, 8, generator=g) for k in (NULL_KEY, "domain", "snowy")}  # noqa: E731
    ft, base = eps(), eps()
    terms = diagnostic_tensors(ft, base, ["snowy"])
    torch.testing.assert_close(terms["g_domain_hybrid"], terms["g_domain"] + terms["diff_null"])
    rows = sample_metrics(terms, ["snowy"])
    for i, row in enumerate(rows):
        assert math.isclose(row["ms_g_domain_hybrid"], float(mean_sq(terms["g_domain"] + terms["diff_null"])[i]), rel_tol=1e-12)
        assert math.isclose(row["rel_diff_null"], math.sqrt(row["ms_diff_null"] / row["ms_g_domain"]), rel_tol=1e-12)
        l2 = float(terms["diff_null"][i].norm() / terms["g_domain"][i].norm())
        assert math.isclose(row["rel_diff_null"], l2, rel_tol=1e-9)  # ms ratio == L2 ratio

    # Diff_Null == 0 -> the cosine is undefined and serialises as null, not NaN
    rows = sample_metrics(diagnostic_tensors(ft, {**base, NULL_KEY: ft[NULL_KEY]}, ["snowy"]), ["snowy"])
    assert all(r["ms_diff_null"] == 0.0 and r["cos_diff_null_g_domain"] is None for r in rows)
    json.dumps(rows, allow_nan=False)


def test_scale_zero_gives_base(tiny_config):
    """§8-3 through the diagnostics path: LoRA scale 0 -> Diff_Null == 0, G_refine == G_refine^Base."""
    model = load_sdxl(tiny_config.model)
    attach_finetuner(model, tiny_config)
    randomize_lora(model.unet)
    rows = _rows(model)
    z, t = _z(model), torch.tensor([700, 81])
    with adapter_scale(model.unet, 0.0):
        ft, base = predict_ft_base(model, z, t, rows, rows, max_rows=4)
    metrics = sample_metrics(diagnostic_tensors(ft, base, ["snowy"]), ["snowy"])
    for m in metrics:
        assert m["ms_diff_null"] == 0.0
        assert m["refine"]["snowy"]["ms_g_refine"] == m["refine"]["snowy"]["ms_g_refine_base"]
        assert math.isclose(m["refine"]["snowy"]["cos_g_refine"], 1.0, abs_tol=1e-9)
    ft, base = predict_ft_base(model, z, t, rows, rows, max_rows=4)
    assert all(m["ms_diff_null"] > 0 for m in sample_metrics(diagnostic_tensors(ft, base, ["snowy"]), ["snowy"]))


def test_cross_sample_cos_matches_brute_force(tmp_path):
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(0)
    x = torch.randn(3, 2, 4, 4, 4, generator=g)  # [N, T, C, h, w]
    x[1, 1] = 0  # a zero vector is skipped
    save_file({"diff_null": x, "g_domain": x.clone()}, str(tmp_path / "u.safetensors"),
              metadata={"source": "forward", "t": "[500, 10]"})
    out = cross_sample_cos([tmp_path / "u.safetensors"], "forward")["all"]
    for j, t in enumerate(out["t"]):
        vectors = [v for v in x[:, [500, 10].index(t)].flatten(1).double() if v.norm() > 0]
        pairs = [float(torch.nn.functional.cosine_similarity(a, b, dim=0))
                 for i, a in enumerate(vectors) for b in vectors[i + 1:]]
        assert math.isclose(out["diff_null"][j], sum(pairs) / len(pairs), rel_tol=1e-9)


# --------------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------------


def test_trajectory_reproduces_generate_latents(tiny_config):
    """The hooked Euler loop is the real sampling trajectory: same latents as the pipeline."""
    model = load_sdxl(tiny_config.model)
    attach_finetuner(model, tiny_config)
    randomize_lora(model.unet)
    cond, null = model.encode_prompts(["a street"]), model.null_cond(1)
    seeds = [image_seed(0, 0), image_seed(0, 1)]
    kw = dict(num_inference_steps=3, guidance_scale=5.0)
    expected = generate_latents(model, cond, null, seeds, **kw)
    ours = euler_cfg_sample(model, seeds, two_row_eps_fn(model, null, cond), **kw)
    torch.testing.assert_close(ours, expected, rtol=1e-5, atol=1e-6)

    rows = {NULL_KEY: null, "domain": cond, "extra": model.encode_prompts(["x"])}

    def eps_fn(info, z, t):  # the diagnostics' bigger, chunked batch
        eps = predict_rows(model, z, t, rows, max_rows=4)
        return eps[NULL_KEY], eps["domain"]

    torch.testing.assert_close(euler_cfg_sample(model, seeds, eps_fn, **kw), expected, rtol=1e-4, atol=1e-5)


def test_timesteps_are_actual(tmp_path, heldout_data):
    config, ckpt = _checkpoint(tmp_path, heldout_data)
    model = load_sdxl(config.model)
    assert sampler_timesteps(model, 2) == [501, 1]  # leading spacing + steps_offset=1, not step indices
    out = diagnose_run(config, checkpoint=ckpt)
    records = _records(out)
    assert {r["t"] for r in records} == {501, 1}
    assert {r["step"] for r in records if r["source"] == "trajectory"} == {0, 1}

    config2, _ = _checkpoint(tmp_path, heldout_data, "lora", "diagnostics.timesteps=[999, 500, 81]", "diagnostics.sources=[forward]")
    out2 = diagnose_run(config2, checkpoint=ckpt, out_name="explicit_t")
    assert sorted({r["t"] for r in _records(out2)}) == [81, 500, 999]


def test_heldout_selection(tmp_path, heldout_data):
    config = _config(tmp_path, heldout_data)
    a = select_heldout_images(config.data, num=3, seed=0)
    train = {f"img_{i:02d}.png" for i in range(10)}
    assert len(a) == 3 and not {p.name for p in a} & train
    assert a == select_heldout_images(config.data, num=3, seed=0)
    assert {tuple(select_heldout_images(config.data, num=3, seed=s)) for s in range(6)} != {tuple(a)}
    with pytest.raises(ValueError, match="held-out candidates"):
        select_heldout_images(config.data, num=5, seed=0)
    with pytest.raises(ValueError, match="image_list"):
        select_heldout_images(_config(tmp_path, heldout_data, "data.image_list=null").data, num=2, seed=0)
    pinned = tmp_path / "pinned.txt"
    pinned.write_text("img_12.png\nimg_13.png\n", encoding="utf-8")
    assert [p.name for p in select_heldout_images(config.data, num=2, seed=0, pinned=str(pinned))] == ["img_12.png", "img_13.png"]
    pinned.write_text("img_00.png\n", encoding="utf-8")
    with pytest.raises(ValueError, match="training images"):
        select_heldout_images(config.data, num=1, seed=0, pinned=str(pinned))


# --------------------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------------------


def test_diagnose_run_end_to_end_and_deterministic(tmp_path, heldout_data):
    config, ckpt = _checkpoint(tmp_path, heldout_data, "lora", ALL_TENSORS)
    flags = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark)
    out = diagnose_run(config, checkpoint=ckpt)
    assert (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark) == flags

    for name in ("config.yaml", "heldout_images.txt", "progress.json", "records.jsonl", "summary.json",
                 "diff_null.png", "g_refine_norm.png", "cos_g_refine.png", "consistency.png"):
        assert (out / name).exists(), name
    heldout = (out / "heldout_images.txt").read_text(encoding="utf-8").split()
    assert len(heldout) == 2 and not set(heldout) & {f"img_{i:02d}.png" for i in range(10)}
    records = _records(out)
    d = config.diagnostics
    assert len([r for r in records if r["source"] == "forward"]) == 2 * 2  # images x t
    assert len([r for r in records if r["source"] == "trajectory"]) == len(d.trajectory_prompts) * len(d.seeds) * 2
    assert all(r["ms_diff_null"] > 0 and set(r["refine"]) == {"snowy", "foggy"} for r in records)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["sources"]) == {"forward", "trajectory"}
    traj = summary["sources"]["trajectory"]
    assert set(traj["groups"]) == {"all", "domain", "snowy", "control"}
    assert traj["groups"]["all"]["t"] == [501, 1]
    assert traj["seed_consistency"]["all"]["seeds"] == [0, 1]
    assert traj["cross_cos"]["all"]["diff_null"][0] is not None
    assert summary["meta"]["prompts"]["snowy"] == {
        "ft": "Photorealistic first-person urban street view, in snowy weather",
        "base": "Photorealistic first-person urban street view, in snowy weather",
    }
    assert summary["meta"]["drive_prompts"]["control"] == "A bowl of fruit on a wooden table"
    assert len(list((out / "trajectory_images").rglob("*.png"))) == len(d.trajectory_prompts) * len(d.seeds)

    from safetensors import safe_open

    with safe_open(str(out / "tensors" / "trajectory__snowy__seed1__0000.safetensors"), framework="pt") as f:
        assert f.get_tensor("diff_null").shape == (1, 2, 4, 16, 16)  # [N, T, C, h, w]
        assert json.loads(f.metadata()["t"]) == [501, 1]
        assert {"g_refine/snowy", "g_refine_base/foggy", "eps_ft_null"} <= set(f.keys())

    again = diagnose_run(config, checkpoint=ckpt, out_name="again")
    assert (again / "records.jsonl").read_bytes() == (out / "records.jsonl").read_bytes()

    small, _ = _checkpoint(tmp_path / "small", heldout_data, "lora", "diagnostics.batch_size=1", "diagnostics.max_rows=1")
    other = _records(diagnose_run(small, checkpoint=ckpt))
    key = lambda r: (r["source"], r.get("prompt_key"), r.get("image"), r["seed"], r.get("index"), r["t"])  # noqa: E731
    by_key = {key(r): r for r in records}
    assert len(other) == len(records)
    for r in other:
        assert math.isclose(r["ms_diff_null"], by_key[key(r)]["ms_diff_null"], rel_tol=1e-4)

    with pytest.raises(FileExistsError):  # never silently overwrite measurements
        diagnose_run(config, checkpoint=ckpt)


def test_textual_inversion_diff_null_exactly_zero(tmp_path, heldout_data):
    config, ckpt = _checkpoint(tmp_path, heldout_data, "textual_inversion")
    out = diagnose_run(config, checkpoint=ckpt)
    records = _records(out)
    assert records and all(r["ms_diff_null"] == 0.0 for r in records)
    assert any(r["refine"]["snowy"]["ms_g_refine"] != r["refine"]["snowy"]["ms_g_refine_base"] for r in records)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["meta"]["checks"]["ti_diff_null_exact_zero"] is True
    prompts = summary["meta"]["prompts"]
    assert all("<s*>" not in p["base"] and "S*" not in p["base"] for p in prompts.values())
    assert all("<s*>" in p["ft"] for p in prompts.values())


def test_textual_inversion_base_prompts_must_not_contain_placeholder(tmp_path, heldout_data, monkeypatch):
    config, ckpt = _checkpoint(tmp_path, heldout_data, "textual_inversion", "diagnostics.base_prompt_set=null")
    monkeypatch.setattr(run_mod, "load_sdxl", _must_not_load)
    with pytest.raises(ValueError, match="placeholder"):
        diagnose_run(config, checkpoint=ckpt)


def test_prompt_sets_for_base(tmp_path):
    ti = load_config("configs/textual_inversion.yaml")
    db = load_config("configs/dreambooth_lora.yaml")
    assert run_mod.base_prompt_set(ti).domain == "Photorealistic first-person urban street view"
    assert run_mod.base_prompt_set(db).domain.endswith("with sks style")  # DreamBooth keeps "sks"
    texts = row_texts(load_prompt_set(ti.prompts), ["snowy"], lambda p: p.replace("S*", "<s*>"))
    assert texts["snowy"].endswith("with <s*> style, in snowy weather")


def test_plot_summaries(tmp_path, heldout_data):
    pytest.importorskip("matplotlib")
    from hcfg.diagnostics.plot import plot_summaries, run_colors

    config, ckpt = _checkpoint(tmp_path, heldout_data)
    summary = json.loads((diagnose_run(config, checkpoint=ckpt) / "summary.json").read_text(encoding="utf-8"))
    other = json.loads(json.dumps(summary))
    other["meta"]["run"] = "textual_inversion"
    written = plot_summaries([summary, other], tmp_path / "figs")
    assert sorted(p.name for p in written) == ["consistency.png", "cos_g_refine.png", "diff_null.png", "g_refine_norm.png"]
    colors = run_colors(["vanilla_lora", "x", "y"])
    assert colors["vanilla_lora"] == "#2a78d6" and colors["x"] != colors["y"]


# --------------------------------------------------------------------------------------
# Storage: room for the measurements
# --------------------------------------------------------------------------------------


def _must_not_load(*args, **kwargs):
    raise AssertionError("the model must not be loaded")


def _du(root):
    return {p.relative_to(root): p.stat().st_size for p in root.rglob("*") if p.is_file()}


def test_storage_estimate_bounds_actual_output(tmp_path, heldout_data):
    config, ckpt = _checkpoint(tmp_path, heldout_data, "lora", ALL_TENSORS)
    estimate = storage.estimate_output_bytes(config)
    out = diagnose_run(config, checkpoint=ckpt)
    files = _du(out)
    records = files[Path("records.jsonl")]
    tensors = {k: v for k, v in files.items() if k.parts[0] == "tensors"}
    images = sum(v for k, v in files.items() if k.parts[0] == "trajectory_images")
    fixed = sum(v for k, v in files.items() if k.parts[0] not in ("tensors", "trajectory_images") and k.name != "records.jsonl")

    assert records <= estimate["records"] <= 3 * records
    assert images <= estimate["images"]
    assert fixed <= estimate["fixed"]
    assert sum(tensors.values()) <= estimate["tensors"]

    # the tensor payload is exact (fp16), only the header is an allowance
    from safetensors import safe_open

    payload = 0
    for name in tensors:
        with safe_open(str(out / name), framework="pt") as f:
            payload += sum(f.get_tensor(k).numel() * 2 for k in f.keys())
    units = sum(c["units"] for c in storage.plan_counts(config, config.diagnostics.sources).values())
    assert payload == estimate["tensors"] - units * storage.TENSOR_FILE_OVERHEAD
    assert sum(tensors.values()) - payload <= units * storage.TENSOR_FILE_OVERHEAD


def test_storage_estimate_scales_with_settings():
    base = load_config("configs/vanilla_lora.yaml", ["diagnostics.save_tensors=[eps_ft_null, eps_base_null, g_domain]"])
    est = storage.estimate_output_bytes(base)
    # SDXL defaults: (7 prompts x 4 seeds x 2 images + 20 held-out) x 25 t x 3 tensors x 4x128x128 fp16
    samples_steps = (7 * 4 * 2 + 20) * 25
    units = 7 * 4 + 5
    assert est["tensors"] == samples_steps * 3 * 4 * 128 * 128 * 2 + units * storage.TENSOR_FILE_OVERHEAD
    assert 0.6 < est["tensors"] / storage.GB < 0.8  # the plan's ~0.7 GB per model

    doubled = load_config("configs/vanilla_lora.yaml", [
        "diagnostics.save_tensors=[eps_ft_null, eps_base_null, g_domain]", "diagnostics.seeds=[0,1,2,3,4,5,6,7]",
        "diagnostics.sources=[trajectory]",
    ])
    traj = load_config("configs/vanilla_lora.yaml", [
        "diagnostics.save_tensors=[eps_ft_null, eps_base_null, g_domain]", "diagnostics.sources=[trajectory]",
    ])
    a, b = storage.estimate_output_bytes(traj), storage.estimate_output_bytes(doubled)
    assert b["records"] == 2 * a["records"] and b["images"] == 2 * a["images"] and b["tensors"] == 2 * a["tensors"]
    none = storage.estimate_output_bytes(load_config("configs/vanilla_lora.yaml", ["diagnostics.save_images=false"]))
    assert none["tensors"] == 0 and none["images"] == 0
    refine = storage.estimate_output_bytes(load_config("configs/vanilla_lora.yaml", ["diagnostics.save_tensors=[g_refine]"]))
    single = storage.estimate_output_bytes(load_config("configs/vanilla_lora.yaml", ["diagnostics.save_tensors=[g_domain]"]))
    assert refine["tensors"] - single["tensors"] == 4 * (single["tensors"] - units * storage.TENSOR_FILE_OVERHEAD)


def test_insufficient_disk_space_fails_before_loading(tmp_path, heldout_data, monkeypatch):
    config, ckpt = _checkpoint(tmp_path, heldout_data)
    monkeypatch.setattr(run_mod, "load_sdxl", _must_not_load)
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda p: DiskUsage(10**12, 10**12, 10**6))
    with pytest.raises(storage.InsufficientDiskSpace, match=r"Need .* GB .* free"):
        diagnose_run(config, checkpoint=ckpt)
    assert not (config.run_dir / "diagnostics").exists()


def test_disk_check_boundary_and_missing_dir(tmp_path, monkeypatch):
    cfg = load_config("configs/tiny_cpu.yaml", ["storage.min_free_gb=1", "storage.disk_margin=2"])
    estimate = {"records": 1000, "tensors": 0, "images": 0, "fixed": 24}
    required = 2 * 1024 + storage.GB
    seen = []

    def fake(path):
        seen.append(Path(path))
        return DiskUsage(0, 0, free)

    monkeypatch.setattr(storage.shutil, "disk_usage", fake)
    free = required
    missing = tmp_path / "not" / "yet" / "there"
    info = storage.check_disk_space(missing, estimate, cfg.storage)
    assert info["required"] == required and seen[-1] == tmp_path  # nearest existing parent
    free = required - 1
    with pytest.raises(storage.InsufficientDiskSpace):
        storage.check_disk_space(missing, estimate, cfg.storage)


def test_disk_check_uses_volume_quota_not_df(tmp_path, monkeypatch):
    """RunPod: df shows the whole cluster; free space is quota - usage of the volume."""
    cfg = load_config("configs/tiny_cpu.yaml", ["storage.volume_quota_gb=150", "storage.min_free_gb=0"])
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda p: DiskUsage(404 * 1024**4, 290 * 1024**4, 115 * 1024**4))
    monkeypatch.setattr(storage, "directory_usage_bytes", lambda root: 149 * storage.GB)
    estimate = {"records": 0, "tensors": 2 * storage.GB, "images": 0, "fixed": 0}
    with pytest.raises(storage.InsufficientDiskSpace, match="quota 150 GB"):
        storage.check_disk_space(tmp_path, estimate, cfg.storage)
    monkeypatch.setattr(storage, "directory_usage_bytes", lambda root: 100 * storage.GB)
    info = storage.check_disk_space(tmp_path, estimate, cfg.storage)
    assert info["free"] == 50 * storage.GB


def test_directory_usage_bytes(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.bin").write_bytes(b"0" * 5000)
    (tmp_path / "y.bin").write_bytes(b"0" * 3000)
    assert storage.directory_usage_bytes(tmp_path) >= 8000


# --------------------------------------------------------------------------------------
# Persistence (RunPod: only the network volume survives closing the pod)
# --------------------------------------------------------------------------------------


def test_output_outside_persistent_root_fails_before_loading(tmp_path, heldout_data, monkeypatch):
    config, ckpt = _checkpoint(tmp_path, heldout_data)
    monkeypatch.setenv(storage.ENV_PERSISTENT_ROOT, str(tmp_path / "volume"))
    monkeypatch.setattr(run_mod, "load_sdxl", _must_not_load)
    with pytest.raises(storage.NotPersistentError, match="lost when the pod"):
        diagnose_run(config, checkpoint=ckpt)
    assert not (config.run_dir / "diagnostics").exists()
    inside = tmp_path / "volume" / "runs" / "x"
    assert storage.ensure_persistent(inside, config.storage) == (tmp_path / "volume").resolve()


def test_persistent_root_resolution(monkeypatch, tmp_path):
    cfg = load_config("configs/tiny_cpu.yaml").storage
    monkeypatch.delenv(storage.ENV_PERSISTENT_ROOT, raising=False)
    monkeypatch.delenv("RUNPOD_VOLUME_ID", raising=False)
    assert storage.persistent_root(cfg) is None
    assert storage.ensure_persistent("/tmp/anything", cfg) is None  # local box: no check
    monkeypatch.setenv("RUNPOD_VOLUME_ID", "abc")
    assert storage.persistent_root(cfg) == Path("/workspace")
    with pytest.raises(storage.NotPersistentError):
        storage.ensure_persistent("/tmp/results", cfg)
    monkeypatch.setenv(storage.ENV_PERSISTENT_ROOT, str(tmp_path))
    assert storage.persistent_root(cfg) == tmp_path.resolve()
    monkeypatch.delenv(storage.ENV_PERSISTENT_ROOT)
    pinned = load_config("configs/tiny_cpu.yaml", [f"storage.persistent_root={tmp_path.as_posix()}"]).storage
    assert storage.persistent_root(pinned) == tmp_path.resolve()


def test_nothing_written_outside_output_dir(tmp_path, heldout_data, monkeypatch):
    pytest.importorskip("matplotlib.pyplot")  # its own cache dir is created on first import
    config, ckpt = _checkpoint(tmp_path, heldout_data)
    sentinel = tmp_path / "tempdir"
    sentinel.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(sentinel))
    before = set(_du(tmp_path))
    out = diagnose_run(config, checkpoint=ckpt)
    new = set(_du(tmp_path)) - before
    rel_out = out.relative_to(tmp_path)
    assert new and all(p.is_relative_to(rel_out) for p in new)
    assert list(sentinel.iterdir()) == []


# --------------------------------------------------------------------------------------
# Interruption and resume
# --------------------------------------------------------------------------------------


class Crash(RuntimeError):
    pass


def _crash_after(monkeypatch, target, name, calls):
    original = getattr(target, name)
    count = {"n": 0}

    def wrapper(*args, **kwargs):
        count["n"] += 1
        if count["n"] == calls:
            raise Crash(f"simulated pod shutdown at call {calls}")
        return original(*args, **kwargs)

    monkeypatch.setattr(target, name, wrapper)


def test_resume_after_crash_mid_unit(tmp_path, heldout_data, monkeypatch):
    config, ckpt = _checkpoint(tmp_path, heldout_data, "lora", ALL_TENSORS)
    reference = diagnose_run(config, checkpoint=ckpt, out_name="reference")

    # forward unit = 2 calls, each trajectory unit = 2 calls: call 5 is inside the 3rd unit
    with monkeypatch.context() as m:
        _crash_after(m, run_mod, "predict_ft_base", 5)
        with pytest.raises(Crash):
            diagnose_run(config, checkpoint=ckpt, out_name="crashed")
    out = config.run_dir / "crashed"
    progress = json.loads((out / "progress.json").read_text(encoding="utf-8"))
    assert progress["units"] == ["forward/seed0/0000", "trajectory/domain/seed0/0000"]
    lines = (out / "records.jsonl").read_text(encoding="utf-8").splitlines()
    assert {json.loads(line)["unit"] for line in lines} == set(progress["units"])  # every line parses
    assert not list(out.rglob("*.partial"))
    assert not (out / "summary.json").exists()

    (out / "tensors" / "junk.safetensors.partial").write_bytes(b"x")  # a write cut short
    diagnose_run(config, checkpoint=ckpt, out_name="crashed", resume=True)
    assert (out / "records.jsonl").read_bytes() == (reference / "records.jsonl").read_bytes()
    assert not list(out.rglob("*.partial"))
    ref_summary = json.loads((reference / "summary.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["sources"] == ref_summary["sources"]


def test_resume_drops_records_of_uncommitted_unit(tmp_path, heldout_data, monkeypatch):
    """Crash after a unit's records were appended but before progress.json was updated."""
    config, ckpt = _checkpoint(tmp_path, heldout_data)
    reference = diagnose_run(config, checkpoint=ckpt, out_name="reference")
    with monkeypatch.context() as m:
        _crash_after(m, run_mod.ProgressStore, "_save", 3)  # 1st call = creation, then one per unit
        with pytest.raises(Crash):
            diagnose_run(config, checkpoint=ckpt, out_name="crashed")
    out = config.run_dir / "crashed"
    units_in_records = {json.loads(line)["unit"] for line in (out / "records.jsonl").read_text(encoding="utf-8").splitlines()}
    assert len(units_in_records) == 2 and len(json.loads((out / "progress.json").read_text(encoding="utf-8"))["units"]) == 1
    diagnose_run(config, checkpoint=ckpt, out_name="crashed", resume=True)
    assert (out / "records.jsonl").read_bytes() == (reference / "records.jsonl").read_bytes()


def test_resume_refuses_changed_settings(tmp_path, heldout_data):
    config, ckpt = _checkpoint(tmp_path, heldout_data)
    diagnose_run(config, checkpoint=ckpt, sources=["forward"])
    changed, _ = _checkpoint(tmp_path / "c", heldout_data, "lora", "diagnostics.seeds=[5]")
    changed.output_dir = config.output_dir
    with pytest.raises(ValueError, match="settings changed"):
        diagnose_run(changed, checkpoint=ckpt, sources=["forward"], resume=True)
    with pytest.raises(FileNotFoundError):
        diagnose_run(config, checkpoint=ckpt, out_name="never_started", resume=True)


def test_summary_from_records_only():
    records = [
        {"source": "forward", "unit": "u", "image": f"i{i}", "seed": s, "t": t,
         **{k: v for k, v in zip(
             ("ms_eps_ft_null", "ms_eps_base_null", "ms_diff_null", "ms_g_domain", "ms_g_domain_base",
              "ms_g_domain_hybrid", "rel_diff_null", "cos_diff_null_g_domain", "cos_eps_null"),
             (1.0, 1.0, 0.25 * (s + 1), 1.0, 1.0, 1.0, 0.5, 0.1, 0.9))},
         "refine": {"snowy": {"ms_g_refine": 2.0, "ms_g_refine_base": 1.0, "cos_g_refine": None}}}
        for s in (0, 1) for i in range(2) for t in (900, 100)
    ]
    out = summarize(records)["sources"]["forward"]
    group = out["groups"]["all"]
    assert group["t"] == [900, 100] and group["n"] == [4, 4]
    assert group["pooled"]["g_refine_ratio"]["snowy"] == [2.0, 2.0]
    assert group["refine"]["snowy"]["cos_g_refine"]["mean"] == [None, None]
    assert math.isclose(group["pooled"]["rel_diff_null"][0], math.sqrt(0.375))
    ms = out["seed_consistency"]["all"]["metrics"]["ms_diff_null"]
    assert ms["seed_means"] == {"0": [0.25, 0.25], "1": [0.5, 0.5]}
    assert math.isclose(ms["cv"][0], 0.125 / 0.375)


def test_script_check_disk(tmp_path, monkeypatch, capsys):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import diagnose

    monkeypatch.setattr(sys, "argv", [
        "diagnose.py", "--config", "configs/tiny_cpu.yaml", "configs/tiny_cpu.yaml", "--check-disk",
        "--set", f"output_dir={tmp_path.as_posix()}/runs",
    ])
    diagnose.main()
    assert "OK: 2 run(s)" in capsys.readouterr().out
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda p: DiskUsage(0, 0, 1))
    with pytest.raises(storage.InsufficientDiskSpace):
        diagnose.main()
    assert not os.path.exists(tmp_path / "runs")
