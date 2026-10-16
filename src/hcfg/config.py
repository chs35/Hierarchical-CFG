"""Typed experiment configuration.

A YAML file maps onto :class:`Config`. Files may ``inherit`` from another file
(path relative to the inheriting file); the child is deep-merged over the parent.
Unknown keys raise, so a typo in a config never silently falls back to a default.
Every script also accepts ``--set a.b=value`` overrides (value parsed as YAML).

Relative paths inside a config (``prompts.file``, ``finetuner.selection`` ...) are
resolved against the project root, never the working directory, so the same
config runs unchanged on the Windows CPU box and the Linux GPU server.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FINETUNER_KINDS = ("lora", "ca_lora", "dreambooth_lora", "textual_inversion")
UNCOND_MODES = ("zeros", "empty")
MIXED_PRECISIONS = ("no", "bf16", "fp16")
HCFG_VARIANTS = ("basic", "alt_domain", "alt_refine")
SCHEDULE_KINDS = ("constant", "linear", "quadratic")
DIAG_SOURCES = ("forward", "trajectory")
DIAG_TENSORS = ("eps_ft_null", "eps_base_null", "g_domain", "diff_null", "g_refine", "g_refine_base")


@dataclass
class ModelConfig:
    kind: str = "sdxl"
    """``sdxl`` (diffusers checkpoint) or ``tiny`` (random offline toy, CPU tests)."""
    pretrained: str = "stabilityai/stable-diffusion-xl-base-1.0"
    revision: str | None = None
    variant: str | None = None
    """``fp16`` downloads the half-precision weight files (smaller, same numerics
    after the cast to ``weight_dtype``)."""
    device: str = "auto"
    weight_dtype: str = "bfloat16"
    """Storage dtype of the *frozen* UNet / text encoders. Trainable parameters
    (LoRA A/B, TI vectors) are always fp32 master weights."""
    vae_dtype: str = "float32"
    """The SDXL VAE overflows in fp16; fp32 costs nothing since latents are cached."""
    resolution: int = 1024
    uncond: str = "zeros"
    """Definition of the null condition ∅. ``zeros``: zero ``prompt_embeds`` and
    zero pooled ``text_embeds`` (SDXL ``force_zeros_for_empty_prompt``, what the
    diffusers pipeline uses by default). ``empty``: the encoding of ``""``. One
    definition is used everywhere: condition dropout, generation, diagnostics."""


@dataclass
class PerfConfig:
    allow_tf32: bool = True
    mixed_precision: str = "bf16"
    """Autocast dtype for UNet forward/backward (tensor cores). Ignored on CPU."""
    channels_last: bool = True
    gradient_checkpointing: bool = False
    torch_compile: bool = False
    fused_optimizer: bool = True
    cudnn_benchmark: bool = True


@dataclass
class DataConfig:
    image_dir: str | None = None
    """Directory holding the fixed training images (set on the GPU server)."""
    image_list: str | None = None
    """Optional text file of image paths (relative to ``image_dir``), one per line.
    Pins the exact 10 images; otherwise every image in ``image_dir`` is used."""
    num_images: int = 10
    """Expected image count; a mismatch raises so every fine-tuner sees the same set."""
    crop: str = "random"
    """``random``/``center`` square crop, done in *latent* space on cached latents."""
    hflip: bool = True
    """Random horizontal flip. Both orientations are VAE-encoded once and cached
    (``VAE(flip(x)) != flip(VAE(x))``)."""


@dataclass
class PromptConfig:
    file: str = "configs/prompts.yaml"
    set: str = "default"
    """Prompt set in ``file``: ``default`` / ``dreambooth`` / ``textual_inversion``."""


@dataclass
class LoRAConfig:
    rank: int = 64
    alpha: float = 64.0
    dropout: float = 0.0
    init: str = "gaussian"
    kinds: list[str] = field(default_factory=lambda: ["q", "k", "v", "out"])
    attn: list[int] = field(default_factory=lambda: [1, 2])
    """Which attention projections a full (non-sliced) LoRA targets."""


@dataclass
class CAConfig:
    """CA measurement / selection (style CA-LoRA only)."""

    axis: str = "style"
    timestep: int = 81
    fraction: float = 0.02
    num_samples: int = 256
    augs_per_sample: int = 4
    augmentation_file: str = "configs/ca_augmentations.json"
    seed: int = 0


@dataclass
class DreamBoothConfig:
    prior_loss_weight: float = 1.0
    num_class_images: int = 100
    class_data_dir: str | None = None
    """Where the prior images live; generated with the *base* model if missing."""
    class_guidance_scale: float = 5.0
    class_num_inference_steps: int = 25
    class_batch_size: int = 4


@dataclass
class TextualInversionConfig:
    placeholder: str = "S*"
    """How the token is written in the prompt file (paper notation)."""
    token: str = "<s*>"
    """The string actually added to both tokenizers."""
    initializer: str = "style"
    num_vectors: int = 1


@dataclass
class FinetunerConfig:
    kind: str = "lora"
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    selection: str | None = None
    """``ca_lora``: selection JSON of head units (``<module>@h<i>``)."""
    ca: CAConfig = field(default_factory=CAConfig)
    dreambooth: DreamBoothConfig = field(default_factory=DreamBoothConfig)
    textual_inversion: TextualInversionConfig = field(default_factory=TextualInversionConfig)


@dataclass
class TrainConfig:
    max_steps: int = 10000
    batch_size: int = 1
    learning_rate: float = 1.0e-4
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 0.0
    adam_epsilon: float = 1.0e-8
    max_grad_norm: float = 1.0
    cond_dropout: float = 0.0
    """Probability of replacing ``c`` by ∅ (the same ∅ as inference)."""
    phi_anchor_lambda: float = 0.0
    """λ of the optional ∅-anchoring term ``λ‖ε(FT,∅) − sg[ε(Base,∅)]‖²``."""
    checkpoint_every: int = 2000
    log_every: int = 50
    seed: int = 0


@dataclass
class GenerateConfig:
    num_inference_steps: int = 25
    guidance_scale: float = 5.0
    """diffusers convention: ε_u + w(ε_c − ε_u); w = 1 means no guidance."""
    guidance_rescale: float = 0.0
    images_per_prompt: int = 20
    seeds: list[int] = field(default_factory=lambda: [0])
    batch_size: int = 4
    prompts: list[str] = field(
        default_factory=lambda: ["domain", "snowy", "foggy", "night", "green_truck", "suv_motorcycle"]
    )
    """Keys: ``domain`` = c, others = refine keys of the prompt file (c + suffix)."""


@dataclass
class EvalConfig:
    clip_model: str = "openai/clip-vit-large-patch14"
    cmmd_clip_model: str = "openai/clip-vit-large-patch14-336"
    reference_dir: str | None = None
    """Real Cityscapes images for CMMD (set on the server)."""
    reference_num_images: int = 500
    detector: str = "google/owlv2-base-patch16-ensemble"
    detection_threshold: float = 0.2
    run_detection: bool = True
    batch_size: int = 16


@dataclass
class HCFGConfig:
    """Hierarchical-CFG sampling and the (w1, w2) grid (report §2.3, §4.2).

    ``basic``      ε(FT,∅)   + w1·G_refine      + w2·G_domain
    ``alt_domain`` ε(Base,∅) + w1·G_refine      + w2·G_domain,hybrid
    ``alt_refine`` ε(FT,∅)   + w1·G_refine^Base + w2·G_domain
    w1 = refine strength, w2 = domain strength; ``basic`` with w1 = w2 = w is standard CFG.

    Diff_Null split (``split_lambda`` or 3-entry ``cells``, report §7):
    ε(Base,∅) + w1·G_refine + w2·G_domain + a·Diff_∥ + λ·Diff_⊥, ``basic`` / ``alt_domain`` only.
    """

    variant: str = "basic"
    w1: list[float] = field(default_factory=lambda: [1.0, 2.0, 4.0, 7.0, 10.0])
    w2: list[float] = field(default_factory=lambda: [1.0, 2.0, 4.0, 7.0, 10.0])
    schedule: str = "constant"
    """``constant``: w(t) = w. ``linear`` / ``quadratic``: the grid value is the weight at
    the first sampling step and w(t) moves to ``schedule_end`` at t = 0 as (t/t_first)^p."""
    schedule_end: float = 1.0
    reduce_equal_weights: bool = True
    """Where w1(t) = w2(t), ``basic``/``alt_domain`` are computed as their reduced 2-branch
    guidance (report §2.3), which for ``basic`` is exactly the standard-CFG pipeline."""
    keys: list[str] = field(default_factory=lambda: ["snowy", "foggy", "night", "green_truck", "suv_motorcycle"])
    images_per_prompt: int = 20
    seeds: list[int] = field(default_factory=lambda: [0])
    batch_size: int = 4
    merge_lora: bool = False
    """``basic`` only (no Base branch): merge the LoRA into the UNet weights."""
    base_prompt_set: str | None = None
    """Prompt set of the Base branches of ``alt_refine``; TI must use ``default``."""
    image_mb_estimate: float = 2.0
    """Disk-check allowance per 1024px PNG (measured 2026-09-29: mean 1.15 MB, max 1.71 MB)."""
    split_lambda: list[float] | None = None
    """Diff_Null split: the λ (Diff_⊥ coefficient) axis of the grid; null = no split. λ = 1 (basic) /
    λ = w2 (alt_domain) is the unsplit sampler, λ = 0 removes Diff_⊥."""
    split_par: float | None = None
    """Diff_Null split: a (Diff_∥ coefficient); null = the variant's own (basic 1, alt_domain w2)."""
    cells: list[list[float]] | None = None
    """Explicit cells instead of the w1 × w2 grid: all [w1, w2] (× ``split_lambda``) or all
    [w1, w2, λ] (the split, without ``split_lambda``)."""

    @property
    def split(self) -> bool:
        return self.split_lambda is not None or bool(self.cells and len(self.cells[0]) == 3)


@dataclass
class DiagnosticsConfig:
    """Pre-experiment diagnostics (report §3, docs/Implementation detail.md §7).

    Two z_t sources, every branch evaluated at the same (z_t, t):
    ``forward`` = q(z_t|x0) on held-out Cityscapes images (never the 10 training
    images), ``trajectory`` = the FT model's standard-CFG sampling trajectory.
    """

    sources: list[str] = field(default_factory=lambda: list(DIAG_SOURCES))
    refine_keys: list[str] = field(
        default_factory=lambda: ["snowy", "foggy", "night", "green_truck", "suv_motorcycle"]
    )
    base_prompt_set: str | None = None
    """Prompt set of the Base branches; ``None`` = ``prompts.set`` (DreamBooth keeps
    "sks"). Textual Inversion must use ``default``: Base does not know S*."""
    num_heldout: int = 20
    heldout_seed: int = 0
    heldout_list: str | None = None
    """Optional pinned held-out list (paths relative to ``data.image_dir``)."""
    forward_seeds: list[int] = field(default_factory=lambda: [0])
    """One noise draw per (seed, image), shared by every t."""
    timesteps: list[int] | None = None
    """Actual timesteps (0-999) of the forward source; ``None`` = the sampler's grid
    for ``generate.num_inference_steps`` so both sources share the same t."""
    trajectory_prompts: list[str] = field(
        default_factory=lambda: ["domain", "snowy", "foggy", "night", "green_truck", "suv_motorcycle", "control"]
    )
    """Prompts that *drive* the trajectories (``control`` = unrelated c_control)."""
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    images_per_seed: int = 2
    """z_T of image i of seed s = ``image_seed(s, i)``, the same as ``generate_images``."""
    save_images: bool = True
    batch_size: int = 4
    """Samples (images or trajectories) processed together; one resumable unit."""
    max_rows: int = 16
    """UNet micro-batch in rows (branches x samples)."""
    save_tensors: list[str] = field(default_factory=list)
    """Subset of ``DIAG_TENSORS`` saved per unit as fp16 safetensors ``[N, T, C, h, w]``."""
    deterministic: bool = True


@dataclass
class StorageConfig:
    """Where results may be written and how much room they need.

    On RunPod only the network volume (``/workspace``) survives closing the pod;
    ``/``, ``/tmp`` and ``~`` are wiped. ``df`` on that volume reports the whole
    storage cluster, not this volume, so free space is ``quota - usage``.
    """

    persistent_root: str | None = None
    """Outputs must live under this directory. ``None``: ``/workspace`` when
    ``RUNPOD_VOLUME_ID`` is set, otherwise no check (Windows / local CPU box).
    The environment variable ``HCFG_PERSISTENT_ROOT`` overrides both."""
    volume_quota_gb: float | None = 200.0
    """Size of the volume at ``persistent_root`` (ignored when there is none)."""
    disk_margin: float = 1.5
    """Required free space = estimate x margin + ``min_free_gb``."""
    min_free_gb: float = 5.0


@dataclass
class Config:
    name: str = "run"
    output_dir: str = "runs"
    model: ModelConfig = field(default_factory=ModelConfig)
    perf: PerfConfig = field(default_factory=PerfConfig)
    data: DataConfig = field(default_factory=DataConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig)
    finetuner: FinetunerConfig = field(default_factory=FinetunerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    generate: GenerateConfig = field(default_factory=GenerateConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    hcfg: HCFGConfig = field(default_factory=HCFGConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    def validate(self) -> Config:
        if self.finetuner.kind not in FINETUNER_KINDS:
            raise ValueError(f"finetuner.kind must be one of {FINETUNER_KINDS}")
        if self.model.uncond not in UNCOND_MODES:
            raise ValueError(f"model.uncond must be one of {UNCOND_MODES}")
        if self.perf.mixed_precision not in MIXED_PRECISIONS:
            raise ValueError(f"perf.mixed_precision must be one of {MIXED_PRECISIONS}")
        if self.data.crop not in ("random", "center"):
            raise ValueError("data.crop must be 'random' or 'center'")
        if not 0.0 <= self.train.cond_dropout < 1.0:
            raise ValueError("train.cond_dropout must be in [0, 1)")
        if self.finetuner.kind == "textual_inversion" and self.train.phi_anchor_lambda > 0:
            raise ValueError("∅-anchoring is meaningless for Textual Inversion (Diff_Null == 0)")
        if self.finetuner.kind == "ca_lora" and not self.finetuner.selection:
            raise ValueError("ca_lora needs finetuner.selection (run scripts/measure_ca.py)")
        h = self.hcfg
        if h.variant not in HCFG_VARIANTS:
            raise ValueError(f"hcfg.variant must be one of {HCFG_VARIANTS}")
        if h.schedule not in SCHEDULE_KINDS:
            raise ValueError(f"hcfg.schedule must be one of {SCHEDULE_KINDS}")
        if h.merge_lora and h.variant != "basic":
            raise ValueError("hcfg.merge_lora needs variant 'basic' (the others use Base branches)")
        if not h.w1 or not h.w2 or min(h.images_per_prompt, h.batch_size) < 1:
            raise ValueError("hcfg: w1/w2 must be non-empty, images_per_prompt/batch_size >= 1")
        if h.cells is not None:
            sizes = {len(c) for c in h.cells}
            if not h.cells or len(sizes) != 1 or sizes - {2, 3}:
                raise ValueError("hcfg.cells must be a non-empty list of all [w1, w2] or all [w1, w2, λ]")
            if sizes == {3} and h.split_lambda is not None:
                raise ValueError("hcfg: give λ either in hcfg.cells or in hcfg.split_lambda, not both")
        if h.split_lambda is not None and not h.split_lambda:
            raise ValueError("hcfg.split_lambda must be null or non-empty")
        if h.split:
            if h.variant not in ("basic", "alt_domain"):
                raise ValueError("The Diff_Null split (hcfg.split_lambda / 3-entry cells) needs variant basic or alt_domain")
            if h.merge_lora:
                raise ValueError("hcfg.merge_lora cannot be used with the Diff_Null split (it needs Base branches)")
        elif h.split_par is not None:
            raise ValueError("hcfg.split_par needs the Diff_Null split (hcfg.split_lambda or 3-entry cells)")
        diag = self.diagnostics
        if not diag.sources or set(diag.sources) - set(DIAG_SOURCES):
            raise ValueError(f"diagnostics.sources must be a non-empty subset of {DIAG_SOURCES}")
        if set(diag.save_tensors) - set(DIAG_TENSORS):
            raise ValueError(f"diagnostics.save_tensors must be a subset of {DIAG_TENSORS}")
        if min(diag.batch_size, diag.max_rows, diag.images_per_seed, diag.num_heldout) < 1:
            raise ValueError("diagnostics batch_size/max_rows/images_per_seed/num_heldout must be >= 1")
        if diag.timesteps is not None and not all(0 <= int(t) < 1000 for t in diag.timesteps):
            raise ValueError("diagnostics.timesteps are actual timesteps in [0, 999]")
        if self.storage.disk_margin < 1.0 or self.storage.min_free_gb < 0:
            raise ValueError("storage.disk_margin must be >= 1 and storage.min_free_gb >= 0")
        return self

    @property
    def run_dir(self) -> Path:
        return resolve_path(self.output_dir) / self.name

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def resolve_path(path: str | Path) -> Path:
    """Absolute paths pass through; relative ones are anchored at the project root."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_yaml_tree(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    parent = data.pop("inherit", None)
    if parent is None:
        return data
    return _deep_merge(_load_yaml_tree((path.parent / parent).resolve()), data)


def _build(cls: type, data: dict, where: str) -> Any:
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise KeyError(f"Unknown config key(s) at '{where}': {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        hint = hints[name]
        if dataclasses.is_dataclass(hint):
            if not isinstance(value, dict):
                raise TypeError(f"'{where}{name}' must be a mapping")
            kwargs[name] = _build(hint, value, f"{where}{name}.")
        else:
            kwargs[name] = _coerce(value, hint)
    return cls(**kwargs)


def _coerce(value: Any, hint: Any) -> Any:
    """YAML reads ``1e-4`` as a string; coerce scalars to the declared type."""
    if value is None:
        return None
    candidates = get_args(hint) if get_origin(hint) is not None else (hint,)
    if str in candidates and bool not in candidates and isinstance(value, bool):
        # YAML 1.1 reads a bare `no`/`yes` (e.g. --set perf.mixed_precision=no) as a bool
        return "yes" if value else "no"
    if float in candidates and isinstance(value, (int, str)) and not isinstance(value, bool):
        return float(value)
    if int in candidates and isinstance(value, str) and value.lstrip("-").isdigit():
        return int(value)
    return value


def _apply_overrides(tree: dict, overrides: list[str]) -> dict:
    tree = copy.deepcopy(tree)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = tree
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return tree


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    tree = _load_yaml_tree(resolve_path(path))
    tree = _apply_overrides(tree, overrides or [])
    return _build(Config, tree, "").validate()


def config_from_dict(data: dict) -> Config:
    return _build(Config, data, "").validate()


def save_config(config: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False, allow_unicode=True)
