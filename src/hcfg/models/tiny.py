"""A randomly initialised, SDXL-shaped model that needs no download.

Same classes as the real checkpoint (``UNet2DConditionModel`` with ``text_time``
added conditioning, two CLIP text encoders, ``AutoencoderKL``, SDXL schedulers),
just tiny, so every code path — sliced LoRA on real diffusers ``Attention``
modules, TI on real CLIP encoders, the diffusers SDXL pipeline — runs on CPU in
seconds. The tokenizer is a byte-level CLIP BPE with no merges, written to a
temp directory on first use.
"""

from __future__ import annotations

import json
import tempfile
from functools import lru_cache
from pathlib import Path

import torch

SCHEDULER_KW = dict(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    steps_offset=1,
    timestep_spacing="leading",
    prediction_type="epsilon",
)
TINY_RESOLUTION = 32
TINY_LATENT_CHANNELS = 4
TINY_VAE_SCALE = 2  # 2 VAE blocks -> one downsampling


def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


@lru_cache(maxsize=1)
def tiny_tokenizer_dir() -> str:
    directory = Path(tempfile.mkdtemp(prefix="hcfg_tiny_tok_"))
    chars = list(_bytes_to_unicode().values())
    vocab = {}
    for c in chars:
        vocab[c] = len(vocab)
    for c in chars:
        vocab[c + "</w>"] = len(vocab)
    vocab["<|startoftext|>"] = len(vocab)
    vocab["<|endoftext|>"] = len(vocab)
    (directory / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (directory / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    return str(directory)


def build_tiny_tokenizer(max_length: int = 77):
    from transformers import CLIPTokenizer

    import inspect

    directory = tiny_tokenizer_dir()
    params = inspect.signature(CLIPTokenizer.__init__).parameters
    if "vocab_file" in params:  # transformers 4.x (slow, file-based)
        return CLIPTokenizer(
            vocab_file=f"{directory}/vocab.json",
            merges_file=f"{directory}/merges.txt",
            model_max_length=max_length,
        )
    # transformers 5.x (tokenizers backend): vocab dict + merges list
    vocab = json.loads(Path(directory, "vocab.json").read_text(encoding="utf-8"))
    return CLIPTokenizer(vocab=vocab, merges=[], model_max_length=max_length)


def build_tiny_components(seed: int = 0) -> dict:
    from diffusers import AutoencoderKL, DDPMScheduler, EulerDiscreteScheduler, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

    torch.manual_seed(seed)
    tokenizers = [build_tiny_tokenizer(), build_tiny_tokenizer()]
    vocab = len(tokenizers[0])
    eos = tokenizers[0].eos_token_id
    bos = tokenizers[0].bos_token_id

    def text_config(hidden: int, proj: int) -> CLIPTextConfig:
        return CLIPTextConfig(
            vocab_size=vocab,
            hidden_size=hidden,
            intermediate_size=2 * hidden,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=77,
            projection_dim=proj,
            hidden_act="gelu",
            bos_token_id=bos,
            eos_token_id=eos,
            pad_token_id=eos,
        )

    text_encoders = [
        CLIPTextModel(text_config(32, 32)),
        CLIPTextModelWithProjection(text_config(32, 32)),
    ]
    unet = UNet2DConditionModel(
        block_out_channels=(32, 64),
        layers_per_block=1,
        sample_size=TINY_RESOLUTION // TINY_VAE_SCALE,
        in_channels=TINY_LATENT_CHANNELS,
        out_channels=TINY_LATENT_CHANNELS,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        attention_head_dim=(2, 4),  # diffusers quirk: this is the number of heads
        use_linear_projection=True,
        addition_embed_type="text_time",
        addition_time_embed_dim=8,
        transformer_layers_per_block=(1, 1),
        projection_class_embeddings_input_dim=6 * 8 + 32,
        cross_attention_dim=64,
        norm_num_groups=8,
    )
    vae = AutoencoderKL(
        block_out_channels=(16, 32),
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        latent_channels=TINY_LATENT_CHANNELS,
        norm_num_groups=8,
        sample_size=TINY_RESOLUTION,
        scaling_factor=0.13025,
    )
    return {
        "unet": unet,
        "vae": vae,
        "text_encoders": text_encoders,
        "tokenizers": tokenizers,
        "noise_scheduler": DDPMScheduler(**SCHEDULER_KW),
        "sample_scheduler": EulerDiscreteScheduler(**SCHEDULER_KW),
    }
