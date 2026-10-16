# Hierarchical CFG

**English** | [한국어](README.ko.md)

**Separate guidance to avoid the forgetting problem in fine-tuned diffusion models.**

Fine-tuning a text-to-image model on a handful of target images forces a trade-off. The model
picks up the target *domain* (here, the look of Cityscapes street scenes) but loses *control*:
prompts like "in snowy weather" or "a green truck on the road" stop working. Hierarchical CFG
addresses this at sampling time, with no retraining. It splits the classifier-free guidance term
into a **domain** direction and a **refine** (control) direction and gives each its own weight.

The full report is in Korean:
[`docs/Hierarchical CFG - Separate Guidance to Avoid Forgetting Problem.md`](docs/Hierarchical%20CFG%20-%20Separate%20Guidance%20to%20Avoid%20Forgetting%20Problem.md).

![Standard CFG vs hierarchical CFG, prompt "..., in snowy weather"](results/diagnostics/basic_cells_snowy_5img.jpg)

*Prompt `Photorealistic first-person urban street view, in snowy weather`, five images each from
style CA-LoRA (top) and vanilla LoRA (bottom). The first three columns are standard CFG
(w1 = w2 = 4, 7, 10); the last four are hierarchical CFG with w1 > w2. Same initial noise in
every column.*

## Method

With c = the domain prompt the model was fine-tuned on and c_refine = c + a control phrase
(`"Photorealistic first-person urban street view"` → `"..., in snowy weather"`), all ε-predictions
taken at the same (z_t, t):

| | |
|---|---|
| G_domain = ε(FT, c) − ε(FT, ∅) | domain guidance |
| G_refine = ε(FT, c_refine) − ε(FT, c) | refine (control) guidance |
| Diff_∅ = ε(FT, ∅) − ε(Base, ∅) | what fine-tuning changed in the unconditional branch |

Standard CFG is ε(FT, ∅) + w·(G_refine + G_domain): one weight for both. Hierarchical CFG
decouples them:

| Variant (`hcfg.variant`) | ε̂ |
|---|---|
| basic (`basic`) | ε(FT, ∅) + w1·G_refine + w2·G_domain |
| domain alternative (`alt_domain`) | ε(Base, ∅) + w1·G_refine + w2·(G_domain + Diff_∅) |
| refine alternative (`alt_refine`) | ε(FT, ∅) + w1·G_refine^Base + w2·G_domain |

w1 is the refine strength and w2 the domain strength. With w1 = w2 the basic form *is* standard
CFG, so the diagonal of every (w1, w2) grid is the baseline, from the same seeds. All branches of
a step go through the UNet in one batch, and Base branches reuse the same UNet with the adapter
switched off.

The report writes the unconditional condition as ∅; code and the research notes call it `null`,
Null, uncond or φ. They are the same thing, see
[`docs/Implementation detail.md`](docs/Implementation%20detail.md) §0 for the full mapping.

## Setup and results

- **Model and data:** SDXL base 1.0 fine-tuned on the same 10 Cityscapes train images with five
  fine-tuners: vanilla LoRA (r = 64), style CA-LoRA, style CA-LoRA + condition dropout, DreamBooth
  LoRA (r = 64), Textual Inversion.
- **Grid:** (w1, w2) ∈ {1, 2, 4, 7, 10}², five control prompts (snow, fog, night, green truck, blue
  SUV + red motorcycle), 20 images per cell and prompt: 25,000 images over ten (fine-tuner, variant)
  pairs, plus 900 for the Diff_∅ split experiment.
- **Scoring:** every image scored 0–10 for domain fit (D ↑), control (C ↑) and collapse (K ↓) by LLM
  sub-agents looking at the images.

Findings (report §6–9):

- **Standard CFG cannot trade domain for control.** Along the w1 = w2 diagonal, the LoRA fine-tuners
  never exceed C = 2.4, whatever the w (CA-LoRA + dropout reaches 3–4).
- **Separate weights recover control.** Under the same quality constraint, hierarchical CFG raises
  the best reachable control by 3–6 points. Examples: DreamBooth LoRA at D ≥ 7, K ≤ 3 goes from
  C 0.97 (standard CFG) to 5.76 at (w1, w2) = (7, 2). Style CA-LoRA with the refine alternative
  reaches C 7.87 at (4, 2) under D ≥ 5, K ≤ 4.
- **Practical region:** w2 ∈ [2, 4] and w1/w2 ≥ 2; (7, 2) is the best cell for most models. The
  diagnostics predicted that w1/w2 ≈ 1.5–2.5 is needed, because fine-tuning shrinks
  ‖G_refine‖/‖G_domain‖ to 0.4–0.65x of the Base model's.
- **Collapse is driven by w2.** Green tint and border noise appear from w2 ≥ 4 for every model,
  including Textual Inversion, where Diff_∅ = 0. Removing the part of Diff_∅ orthogonal to the
  domain direction did not reduce it (a negative result, report §7).

**Limitation:** all generation results rest on LLM visual scoring with one seed. The planned
quantitative evaluation (a P(Cityscapes | x) classifier, zero-shot detection) has not been done.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m pytest -q                                    # CPU, tiny random model, no downloads
python scripts/smoke_test.py                           # every stage end to end on the tiny model
```

On a GPU (an A100 80 GB was used), with Cityscapes downloaded:

```bash
DATA=(data.image_dir=/path/to/cityscapes/leftImg8bit/train data.image_list=configs/data/train10.txt)
python scripts/train.py --config configs/vanilla_lora.yaml --set "${DATA[@]}"                 # ~70 min
python scripts/sweep.py --config configs/vanilla_lora.yaml \
    --set "hcfg.cells=[[7,2]]" "hcfg.keys=[snowy]"                                          # one cell
python scripts/sweep.py --config configs/vanilla_lora.yaml                                   # full 5x5 grid
```

The full procedure, including all five fine-tuners, diagnostics, the alternative variants, the
Diff_∅ split experiment, timings and disk sizes, is in [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

## Repository layout

```
configs/    base.yaml (shared) + one file per fine-tuner (inherit: base.yaml), prompts.yaml,
            data/train10.txt (the 10 training images), selections/ (CA-LoRA head selection),
            tiny_cpu.yaml (CPU tests)
src/hcfg/   models/       SDXL wrapper, sliced (per-head) LoRA, Textual Inversion, tiny test model
            train/        trainer for all five fine-tuners
            generation/   standard CFG, hierarchical CFG sampler (hierarchical.py), (w1, w2) grid
            diagnostics/  G_refine / G_domain / Diff_∅ measured at the same (z_t, t)
            eval/         CMMD, CLIP score, CLIP zero-shot, OWLv2 detection
            ca/           CA-LoRA head sensitivity measurement
scripts/    train, generate, evaluate, sweep, diagnose, plot_diagnostics, measure_ca,
            gpu_check, smoke_test
tests/      unit and end-to-end tests on the tiny model
results/    summaries behind the report: training summaries and loss curves, diagnostics
            summaries and figures, GPU check log, training-data hashes
docs/       report and supporting documents (below)
```

## Documents

| File | Content |
|---|---|
| [`docs/Hierarchical CFG - Separate Guidance to Avoid Forgetting Problem.md`](docs/Hierarchical%20CFG%20-%20Separate%20Guidance%20to%20Avoid%20Forgetting%20Problem.md) | Final report (Korean) |
| [`docs/sample_observation_report_CLAUDE.md`](docs/sample_observation_report_CLAUDE.md) | Scores of all 25,900 generated images: rubric, every 5x5 grid, per-prompt analysis (Korean) |
| [`docs/sample_observation_scores_CLAUDE.csv`](docs/sample_observation_scores_CLAUDE.csv) | Per-image scores (model, variant, prompt, w1, w2, λ, file, D, C, K, short note) |
| [`docs/diagnostics.md`](docs/diagnostics.md) | Detailed analysis of the pre-experiment diagnostics (Korean) |
| [`docs/Implementation detail.md`](docs/Implementation%20detail.md) | Implementation rules and the report ↔ code terminology table (Korean) |
| [`docs/REPRODUCE.md`](docs/REPRODUCE.md) | Step-by-step reproduction guide |

## License

The code in this repository is released under the [MIT License](LICENSE). It covers only the code
and documents here; the following are **not** included and keep their own terms:

- **SDXL weights** (`stabilityai/stable-diffusion-xl-base-1.0`) are downloaded at run time and are
  licensed under the CreativeML Open RAIL++-M License. Fine-tuned weights (LoRA / Textual
  Inversion) are derivatives of SDXL; none are distributed here, and anyone who publishes such
  weights must do so under that license's use restrictions.
- **Cityscapes** images are not included (only file names and hashes). Obtain them from
  <https://www.cityscapes-dataset.com/> under the Cityscapes terms of use.
- The images in `results/` are SDXL outputs generated for this study.
- `configs/ca_augmentations.json` and `configs/selections/ref_style_t81.json` are taken from the
  CA-LoRA reference implementation (chs35/Improving-CA-LoRA).
