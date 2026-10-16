# Reproducing the experiments

This guide reruns everything behind the report
([`docs/Hierarchical CFG - Separate Guidance to Avoid Forgetting Problem.md`](Hierarchical%20CFG%20-%20Separate%20Guidance%20to%20Avoid%20Forgetting%20Problem.md)):
five fine-tuners → diagnostics → (w1, w2) grids → the Diff_Null split experiment. Section numbers
in brackets (e.g. *report §4.2*) refer to the report.

Every script takes `--config configs/<run>.yaml` plus dotted overrides `--set key=value ...`.
Settings are never edited in place for an experiment; anything that differs from the YAML is passed
with `--set` so the run's `config.yaml` records it.

## 0. What was used

| | |
|---|---|
| GPU | 1x NVIDIA A100-SXM4-80GB (SM 8.0) |
| Software | Python 3.12.3, torch 2.8.0+cu128, diffusers 0.40.0, transformers 5.17.0 |
| Precision | bf16 UNet / text encoders, fp32 VAE, fp32 trainable weights (see §9) |
| Base model | `stabilityai/stable-diffusion-xl-base-1.0` (fp16 variant) |

The exact resolved config of each trained run is in `results/<run>/config.yaml`; the commands below
reproduce those configs field for field.

Wall-clock budget on one A100: training ≈ 4 h, diagnostics ≈ 3.5 h, the ten grids of report §4.2
≈ 30 h, the split experiment ≈ 1.5 h.

## 1. Install and test on CPU

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m pytest -q                       # tiny random SDXL-shaped model, no downloads, ~1 min
python scripts/smoke_test.py              # all five fine-tuners end to end on the tiny model
```

The first GPU run downloads about 12 GB from Hugging Face: SDXL base 1.0, `openai/clip-vit-large-patch14`,
`openai/clip-vit-large-patch14-336` (CMMD) and `google/owlv2-base-patch16-ensemble` (detection).
Set `HF_HOME` to a large disk if needed. SDXL is distributed under the CreativeML Open RAIL++-M
license, which you accept when downloading it.

## 2. Data

Cityscapes (`leftImg8bit`) is not redistributed here; download it from
<https://www.cityscapes-dataset.com/> under its own terms.

- **Training images.** 10 images of `leftImg8bit/train`, listed in `configs/data/train10.txt`
  (relative to the train directory). They are the first 10 indices of
  `torch.Generator().manual_seed(0)` `randperm` over the sorted 2,975 train images. All five
  fine-tuners use the same 10. Point `data.image_dir` at the **whole** train directory and pin the
  10 with `data.image_list`; the diagnostics draw their held-out images from the rest of it.
- **Integrity.** `results/data_manifest.json` holds the size and sha256 of each of the 10 files:

  ```bash
  python - /path/to/cityscapes/leftImg8bit/train <<'EOF'
  import hashlib, json, sys
  from pathlib import Path
  root = Path(sys.argv[1])
  for img in json.load(open("results/data_manifest.json"))["images"]:
      ok = hashlib.sha256((root / img["file"]).read_bytes()).hexdigest() == img["sha256"]
      print("OK  " if ok else "FAIL", img["file"])
  EOF
  ```
- **CMMD reference** (optional, only for `scripts/evaluate.py` and the full GPU check): real images,
  e.g. 500 images of `leftImg8bit/val`, as `eval.reference_dir`.

The commands below use:

```bash
export IMAGES=/path/to/cityscapes/leftImg8bit/train
export LIST=configs/data/train10.txt
export SEL=configs/selections/ref_style_t81.json     # CA-LoRA head selection used in the report (§3)
export REF=/path/to/cityscapes/leftImg8bit/val       # optional
DATA=(data.image_dir=$IMAGES data.image_list=$LIST)
CONFIGS="configs/vanilla_lora.yaml configs/ca_lora_style.yaml configs/ca_lora_style_dropout.yaml configs/dreambooth_lora.yaml configs/textual_inversion.yaml"
```

## 3. Outputs and storage

All outputs go to `runs/<run>/` (not tracked by git): checkpoints, `samples/`, `diagnostics/`,
`hcfg/<variant>/w1_<a>_w2_<b>/<key>/seed<s>/<i>.png`. Small summaries worth keeping are copied to
`results/`.

| Output | Size |
|---|---|
| Checkpoints (5 runs, incl. intermediate) | ~3.9 GB |
| One 5x5 grid (2,500 PNGs, ~1.15 MB each) | ~2.9 GB |
| The ten grids of report §4.2 + split experiment | ~30 GB |

`scripts/diagnose.py` and `scripts/sweep.py` refuse to start when the output directory is not on
persistent storage or would not fit. The check is active when `HCFG_PERSISTENT_ROOT` (env) or
`storage.persistent_root` is set, or automatically on RunPod (`RUNPOD_VOLUME_ID` set → `/workspace`,
with `storage.volume_quota_gb` as the volume size because `df` there shows the whole cluster).
Elsewhere it is skipped. `--check-disk` runs only this check, for several configs at once.

## 4. GPU verification (optional, recommended)

`scripts/gpu_check.py` checks what the CPU tests cannot: real SDXL, bf16 kernels, A100 tensor cores.

```bash
python scripts/gpu_check.py --stages env exact tensorcore                       # ~5-10 min
python scripts/gpu_check.py --image-dir $IMAGES --reference-dir $REF --compare-fp32   # ~30-40 min
```

It writes `runs/gpu_check/report.json` (PASS / WARN / FAIL / SKIP per check) and exits with 1 on any
FAIL. Our quick run is `results/gpu_check/report_quick.json` (30 PASS). Key checks:

| Stage | Check | Pass |
|---|---|---|
| exact | `*:scale0_equals_base` — adapter scale 0 reproduces Base | `0.0` |
| exact | `textual_inversion:diff_null_zero` — TI leaves ε(·,∅) unchanged | `0.0` |
| exact | `ref_selection_units` / `ref_selection_modules` — CA selection maps onto SDXL | 208 / 65 |
| exact | `ca_lora:checkpoint_roundtrip` — sliced LoRA restored from the checkpoint alone | `0.0` |
| tensorcore | `*:tensor_core_share` — share of GEMM/conv/attention CUDA time on tensor cores | ≥ 0.8 |
| train | `*:sec_per_step` (1024 px, batch 1) | ≤ 0.6 s (WARN ≤ 1.0) |
| train | `bf16_speedup_over_fp32` (`--compare-fp32`) | ≥ 2 |

The same checks also run as `HCFG_GPU_TESTS=1 python -m pytest tests/test_gpu_check.py -q`.
Do not run CPU tests concurrently with GPU timing; CPU throttling inflates `sec_per_step`.

## 5. Training (report §4.1)

Run one at a time (concurrent runs distort the timings). Each run writes
`runs/<run>/finetuner-final.safetensors`, a checkpoint every 2,000 steps (TI: 1,000),
`train_log.jsonl`, `train_summary.json` and `loss_curve.png`.

```bash
python scripts/train.py --config configs/vanilla_lora.yaml          --set "${DATA[@]}"
python scripts/train.py --config configs/ca_lora_style.yaml         --set "${DATA[@]}" finetuner.selection=$SEL
python scripts/train.py --config configs/ca_lora_style_dropout.yaml --set "${DATA[@]}" finetuner.selection=$SEL
python scripts/train.py --config configs/dreambooth_lora.yaml       --set "${DATA[@]}"
python scripts/train.py --config configs/textual_inversion.yaml     --set "${DATA[@]}"
```

| Run | Steps | Trainable params | Time | s/step | Peak memory |
|---|---|---|---|---|---|
| vanilla LoRA r=64 | 10k | 92.9M | 70.8 min | 0.425 | 13.0 GB |
| style CA-LoRA | 10k | 7.75M (208 heads, 65 matrices) | 33.3 min | 0.20 | 10.4 GB |
| style CA-LoRA + cond. dropout 0.1 | 10k | 7.75M | 33.5 min | 0.20 | 10.4 GB |
| DreamBooth LoRA r=64 + prior | 10k | 92.9M | 72.4 min | 0.434 | 19.9 GB |
| Textual Inversion (1 vector per encoder) | 3k | 2,048 | 12.9 min | 0.26 | 12.8 GB |

Full numbers: `results/finetuner_train_summary.json`, per run `results/<run>/train_summary.json`.

- **CA-LoRA head selection.** The report uses the style selection measured by the CA-LoRA
  reference repository (chs35/Improving-CA-LoRA) on Cityscapes train at t = 81, top 2% of heads,
  copied to `configs/selections/ref_style_t81.json`. Both CA-LoRA runs must use the **same**
  selection. To measure a selection yourself instead (not what the report used):
  `python scripts/measure_ca.py --config configs/ca_lora_style.yaml --set "${DATA[@]}"` →
  `runs/ca_selection/selection_style_t81.json`, the default of `configs/ca_lora_style.yaml`.
- **DreamBooth** first generates 100 class images with the Base model
  (`runs/dreambooth_class_images_1024/`, reused afterwards).
- **Loss** fluctuates strongly with the sampled timestep; do not read model quality from it
  (`results/loss_curves_compare.png`). All runs should have only finite losses.

## 6. Standard-CFG samples and automatic metrics (optional)

Not used for the report's conclusions, but handy as a sanity check:

```bash
python scripts/generate.py --config configs/vanilla_lora.yaml                   # w = 5, 6 prompts x 20 images
python scripts/generate.py --config configs/base.yaml --set name=base --base     # Base model control
python scripts/evaluate.py --config configs/vanilla_lora.yaml --set eval.reference_dir=$REF
```

`runs/<run>/eval_samples.json` holds CMMD to real Cityscapes, CLIP score, CLIP zero-shot weather /
vehicle classification and OWLv2 vehicle type + colour detection per prompt. The planned
P(Cityscapes | x) domain classifier is not implemented (report §9.2).

## 7. Diagnostics (report §3, §5)

Every branch (FT and Base; ∅, c, five c_refine) is evaluated at the same (z_t, t), on two z_t sources:
`forward` = q(z_t | x0) of 20 held-out Cityscapes train images (not among the 10, fixed seed),
`trajectory` = the fine-tuned model's own standard-CFG sampling (7 prompts x 4 seeds x 2 images).

```bash
python scripts/diagnose.py --config $CONFIGS --set "${DATA[@]}" --save-tensors --check-disk
for c in $CONFIGS; do
  python scripts/diagnose.py --config $c --set "${DATA[@]}" --save-tensors     # ~35-45 min per run
done
python scripts/plot_diagnostics.py         # overlays all runs -> results/diagnostics/*.png + <run>_summary.json
```

- If a run is interrupted, rerun the same command with `--resume`; finished units are skipped and
  changed settings are refused.
- Checks: `textual_inversion`'s `summary.json` must report `meta.checks.ti_diff_null_exact_zero: true`;
  the other four `max_ms_diff_null > 0`; all runs must list the same `heldout_images.txt`.
- Our values: `results/diagnostics/<run>_summary.json` and the figures next to them; the GPU
  verification log of this stage is `results/diagnostics/gpu_check.txt`. Analysis:
  `docs/diagnostics.md`.

## 8. Hierarchical-CFG grids (report §4.2, §6)

(w1, w2) ∈ {1, 2, 4, 7, 10}², 5 refine prompts, seed 0, 20 images per cell and prompt
= 2,500 images per (run, variant). Image i of a cell starts from the same z_T in every run, variant
and cell. Rerunning the same command resumes (existing images are skipped).

```bash
python scripts/sweep.py --config $CONFIGS --check-disk

# basic form, all five runs (~2.7 h each)
for c in $CONFIGS; do python scripts/sweep.py --config $c; done

# domain alternative (~2.6-2.9 h each)
for c in configs/vanilla_lora.yaml configs/ca_lora_style.yaml configs/ca_lora_style_dropout.yaml; do
  python scripts/sweep.py --config $c --set hcfg.variant=alt_domain
done

# refine alternative (~3.6-4.1 h each)
for c in configs/vanilla_lora.yaml configs/ca_lora_style.yaml; do
  python scripts/sweep.py --config $c --set hcfg.variant=alt_refine
done
```

- Output: `runs/<run>/hcfg/<variant>/w1_<a>_w2_<b>/<key>/seed0/0000.png ...`, one `manifest.json`
  per cell, `sweep.json` with the settings fingerprint, progress in `runs/<run>/sweep.log`.
- On the w1 = w2 diagonal the basic form is computed in reduced form and is **bit-identical** to
  standard CFG at that w (`hcfg.reduce_equal_weights`, verified on the A100). Diagonal cells take
  ~2.5 s/image, the others ~3.7 s/image (3 branches), peak memory ~9 GB.
- Time schedules `hcfg.schedule=linear|quadratic` are implemented but were not used in the report.

## 9. Diff_Null split experiment (report §7)

Style CA-LoRA, w1 = 7. Diff_Null is split per sample into the component along
r = ε(FT,c) − ε(Base,c) (Diff_∥) and the rest (Diff_⊥), and the sampler becomes
ε(Base,∅) + w1·G_refine + w2·G_domain + a·Diff_∥ + λ·Diff_⊥ (basic: a = 1, alt_domain: a = w2).
Only the new cells are generated; the λ = 1 (basic) and λ = w2 (alt_domain) references are the
existing grid cells.

```bash
C=configs/ca_lora_style.yaml
python scripts/sweep.py --config $C --set "hcfg.cells=[[7,2,0],[7,4,0],[7,4,-1],[7,4,5]]"
python scripts/sweep.py --config $C --set hcfg.variant=alt_domain "hcfg.cells=[[7,2,0],[7,4,0],[7,2,1],[7,4,1],[7,4,8]]"
```

900 images, ~89 min, ~1.1 GB, output in `runs/ca_lora_style/hcfg/{basic_split,alt_domain_split}/w1_7_w2_<b>_lam_<λ>/`.
Each split step runs FT [∅, c, c_r] + Base [∅, c] (~6 s/image). Sanity check: basic with λ = 1 reproduces
the existing grid cell bit for bit:

```bash
python scripts/sweep.py --config $C --out-name split_check \n  --set "hcfg.cells=[[7,2,1]]" "hcfg.keys=[snowy]" hcfg.images_per_prompt=4
# compare runs/ca_lora_style/hcfg/split_check/w1_7_w2_2_lam_1/snowy/seed0/000{0..3}.png
#    with runs/ca_lora_style/hcfg/basic/w1_7_w2_2/snowy/seed0/000{0..3}.png
```

## 10. Scoring the generated images (report §6)

All 25,900 grid images were scored 0–10 for domain fit (D), control (C) and collapse (K) by LLM
sub-agents (Claude Opus 5.5) looking at the images; one agent per (run, variant, prompt, w1 row).
The rubric and procedure are in `docs/sample_observation_report_CLAUDE.md` §1, the per-image scores in
`docs/sample_observation_scores_CLAUDE.csv`. The orchestration itself is not part of this repository.

To compute automatic metrics on any grid cell instead:

```python
from hcfg.config import load_config
from hcfg.eval.evaluate import evaluate_samples

cfg = load_config("configs/ca_lora_style.yaml", ["eval.reference_dir=/path/to/cityscapes/leftImg8bit/val"])
evaluate_samples(cfg, "runs/ca_lora_style/hcfg/basic/w1_7_w2_2")
```

## 11. Performance notes (A100)

An earlier CA-LoRA reproduction took ~6 h for 10k steps on an A100 (2.2 s/step), against ~1 h on a
V100 in the CA-LoRA paper (~0.36 s/step). The cause was the pipeline, not the hardware:

1. fp32 weights, `mixed_precision: "no"`, no TF32 → tensor cores idle (3–5x);
2. `crop: random` disabled the latent cache, so a 1024 px crop was VAE-encoded in fp32 at every step;
3. gradient checkpointing on (30–40% overhead, unnecessary at batch 1 on 80 GB);
4. periodic CA-drift snapshots during training.

This code therefore (`perf` section of `configs/base.yaml`, `hcfg.utils.device`, `hcfg.data.latent_cache`):

- stores the frozen UNet and text encoders in bf16 and runs the UNet under bf16 autocast, with TF32 on
  and fp32 master weights for the trainable parameters; fused AdamW; `channels_last`;
- encodes each training image **once** at full width (2048x1024 → 128x256 latent) and takes the
  random 1024 px crop in latent space (a 128-latent crop); flipped images are encoded separately
  (VAE(flip(x)) ≠ flip(VAE(x)));
- turns gradient checkpointing off and takes no drift snapshots; `train_summary.json` reports pure
  training s/step, excluding checkpoint time.

Result: 0.20–0.43 s/step (§5). `scripts/gpu_check.py` (§4) measures the tensor-core share and the
bf16 / fp32 speed-up directly.

## 12. Numerics

- Training, generation and diagnostics all run in bf16; fp32 cross-checks were not done (report §9.2).
- Diagnostics follow generation numerics (bf16 weights and inputs, no autocast), compute metrics in
  float64 and use deterministic kernels. The hooked Euler loop of the diagnostics reproduces
  `generate_latents` bit for bit for the same batch.
- Changing how branches are batched changes bf16 rounding. The unreduced 3-branch basic form gives the
  same image as the reduced one only up to PSNR 19–27 dB, and diagnostic trajectories (14-row
  batches) match `samples/` at PSNR 23.7–29.7 dB. Reproductions on other GPUs or library versions
  should match the report statistically, not pixel for pixel.
