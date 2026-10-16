import pytest
import torch

from hcfg.config import load_config
from hcfg.data.prompts import load_prompt_set
from hcfg.models.sdxl import Cond

C = "Photorealistic first-person urban street view"


@pytest.mark.parametrize(
    "path,kind",
    [
        ("configs/vanilla_lora.yaml", "lora"),
        ("configs/ca_lora_style.yaml", "ca_lora"),
        ("configs/ca_lora_style_dropout.yaml", "ca_lora"),
        ("configs/dreambooth_lora.yaml", "dreambooth_lora"),
        ("configs/textual_inversion.yaml", "textual_inversion"),
    ],
)
def test_finetuner_configs_load(path, kind):
    config = load_config(path)
    assert config.finetuner.kind == kind
    assert config.finetuner.lora.rank == 64
    # A100 tensor-core settings (docs/REPRODUCE.md, performance notes)
    assert config.perf.mixed_precision == "bf16" and config.model.weight_dtype == "bfloat16"
    assert config.perf.allow_tf32 and not config.perf.gradient_checkpointing
    assert config.model.uncond == "zeros"


def test_dropout_control_differs_only_in_dropout():
    a = load_config("configs/ca_lora_style.yaml").to_dict()
    b = load_config("configs/ca_lora_style_dropout.yaml").to_dict()
    assert b["train"]["cond_dropout"] == 0.1 and a["train"]["cond_dropout"] == 0.0
    a["train"]["cond_dropout"] = b["train"]["cond_dropout"]
    a["name"] = b["name"]
    assert a == b


def test_unknown_key_raises():
    with pytest.raises(KeyError):
        load_config("configs/vanilla_lora.yaml", ["train.max_stepz=3"])


def test_diagnostics_and_storage_sections():
    config = load_config("configs/vanilla_lora.yaml")
    d = config.diagnostics
    assert d.sources == ["forward", "trajectory"] and d.base_prompt_set is None
    assert d.refine_keys == ["snowy", "foggy", "night", "green_truck", "suv_motorcycle"]
    assert d.trajectory_prompts[-1] == "control" and d.num_heldout == 20 and d.save_tensors == []
    assert config.storage.volume_quota_gb == 200 and config.storage.persistent_root is None
    assert load_config("configs/textual_inversion.yaml").diagnostics.base_prompt_set == "default"
    assert load_config("configs/dreambooth_lora.yaml").diagnostics.base_prompt_set is None  # keeps "sks"
    with pytest.raises(KeyError):
        load_config("configs/vanilla_lora.yaml", ["diagnostics.seedz=[1]"])
    with pytest.raises(ValueError, match="save_tensors"):
        load_config("configs/vanilla_lora.yaml", ["diagnostics.save_tensors=[eps_everything]"])
    with pytest.raises(ValueError, match="sources"):
        load_config("configs/vanilla_lora.yaml", ["diagnostics.sources=[train]"])
    with pytest.raises(ValueError, match="timesteps"):
        load_config("configs/vanilla_lora.yaml", ["diagnostics.timesteps=[1000]"])


def test_control_prompt():
    for path in ("configs/vanilla_lora.yaml", "configs/dreambooth_lora.yaml", "configs/textual_inversion.yaml"):
        prompts = load_prompt_set(load_config(path).prompts)
        assert prompts.prompt("control") == "A bowl of fruit on a wooden table"
        assert "control" not in prompts.keys  # neither c nor c_refine


def test_scientific_notation_is_float():
    config = load_config("configs/vanilla_lora.yaml", ["train.learning_rate=3e-4"])
    assert config.train.learning_rate == pytest.approx(3e-4)


def test_yaml_bool_no_is_string():
    config = load_config("configs/vanilla_lora.yaml", ["perf.mixed_precision=no"])
    assert config.perf.mixed_precision == "no"


def test_prompts_match_the_report():
    default = load_prompt_set(load_config("configs/vanilla_lora.yaml").prompts)
    db = load_prompt_set(load_config("configs/dreambooth_lora.yaml").prompts)
    ti = load_prompt_set(load_config("configs/textual_inversion.yaml").prompts)
    assert default.domain == C
    assert db.domain == C + " with sks style" and db.prior == C
    assert ti.domain == C + " with S* style"
    assert default.refine("snowy") == C + ", in snowy weather"
    assert default.refine("night") == C + ", in dark night"
    assert default.refine("green_truck") == C + ", a green truck on the road"
    assert default.refine("suv_motorcycle") == C + ", a blue SUV and a red motorcycle on the road"
    for ps in (default, db, ti):
        for key in ps.refine_suffixes:
            assert ps.refine(key).startswith(ps.domain)  # c_refine contains c
    assert set(default.refine_suffixes) == {"snowy", "foggy", "night", "green_truck", "suv_motorcycle"}


def test_cond_where_selects_per_sample():
    a = Cond(torch.ones(3, 2, 4), torch.ones(3, 5))
    b = Cond(torch.zeros(3, 2, 4), torch.zeros(3, 5))
    out = Cond.where(torch.tensor([True, False, True]), a, b)
    assert out.prompt_embeds[:, 0, 0].tolist() == [1, 0, 1]
    assert out.pooled[:, 0].tolist() == [1, 0, 1]


def test_predict_rejects_batch_mismatch(tiny_model):
    z = torch.randn(2, 4, tiny_model.latent_size, tiny_model.latent_size)
    with pytest.raises(ValueError):
        tiny_model.predict(z, torch.tensor([1, 2]), tiny_model.null_cond(1))


def test_null_cond_zeros(tiny_model):
    null = tiny_model.null_cond(3)
    assert null.batch_size == 3
    assert torch.count_nonzero(null.prompt_embeds) == 0 and torch.count_nonzero(null.pooled) == 0
    c = tiny_model.encode_prompts(["x"])
    assert null.prompt_embeds.shape[1:] == c.prompt_embeds.shape[1:]
    assert null.pooled.shape[1:] == c.pooled.shape[1:]
