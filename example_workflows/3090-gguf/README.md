# RTX 3090 / LTX-2.3 GGUF workflows

Built for a single **RTX 3090** (24 GB, compute capability 8.6) running LTX-2.3
GGUF weights. Unlike the previous revision, **these have been executed**, not just
validated against the node registry.

**Text to video**

| Workflow | What it's for |
|---|---|
| `LTX-2.3_GGUF_T2V_Baseline_3090.json` | Minimal GGUF text-to-video. Start here. |
| `LTX-2.3_GGUF_T2V_Distilled_3090.json` | **The fast path**: distilled LoRA, 8 steps, cfg 1.0, SageAttention. |
| `LTX-2.3_GGUF_T2V_STG_APG_3090.json` | STG + APG guidance. |
| `LTX-2.3_GGUF_LongClip_LowVRAM_3090.json` | 1280×704 / 193 frames, spatio-temporal decode, CPU accumulator. |

**Image to video**

| Workflow | What it's for |
|---|---|
| `LTX-2.3_GGUF_I2V_Distilled_3090.json` | **The fast path**: distilled LoRA, 8 steps, cfg 1.0, SageAttention. |
| `LTX-2.3_GGUF_I2V_Baseline_3090.json` | Undistilled reference, 20 steps, cfg 3.0. |
| `LTX-2.3_GGUF_I2V_STG_APG_3090.json` | STG + APG on top of image conditioning. |
| `LTX-2.3_GGUF_I2V_FirstLastFrame_3090.json` | Two keyframes — start at one image, land on another. |

Every I2V file points at ComfyUI's stock `input/example.png`, so they open and run
on a fresh install before you change anything.

## Two things the earlier revision got wrong

Both were found by actually running the graphs, and both made them impossible to
execute. If you have an older copy of these files, this is why they failed.

### 1. `ltx-2-3-22b-text_encoder.safetensors` is not a text encoder

It contains **four tensors** — `text_embedding_projection.{audio,video}_aggregate_embed.{weight,bias}`.
LTX-2.3's text encoder is **Gemma-3-12B**; that projection aggregates Gemma's 49
hidden states (49 × 3840 = 188160) down to the DiT's conditioning dims.

`comfy/sd.py` only takes the LTX-2.3 branch — `ltxav_te` + `LTXAVGemmaTokenizer`
(`sd.py:1809`) — when handed **exactly two** state dicts. A single-file
`CLIPLoader(type=ltxv)` falls into the LTX-0.9 **T5** branch (`sd.py:1601`)
instead, builds a T5 with no matching weights, and dies with:

```
NotImplementedError: Cannot copy out of meta tensor; no data!
```

**Use `DualCLIPLoader` with `type = ltxv`:**

| Widget | Value |
|---|---|
| `clip_name1` | a Gemma-3-12B, e.g. `gemma-3-12b-it-ablit-norms-biproj-fp8mixed.safetensors` |
| `clip_name2` | `ltx-2-3-22b-text_encoder.safetensors` |

Gemma must be **first** — `sd.py:1812` reads the tokenizer from
`clip_data[0]["spiece_model"]`, and only the Gemma file carries it.

Both live in `models/text_encoders`, so **nothing is needed in
`models/checkpoints`**. That removes the blocker the previous README described:
`LTXAVTextEncoderLoader` and `LTXVGemmaCLIPModelLoader` both list from
`checkpoints`, which is typically empty on a GGUF install.

### 2. The audio branch is not optional

`ltx-2.3-22b-dev` is an **AV** DiT. `av_model.py:854` builds audio RoPE
unconditionally, so a video-only latent hands it zero audio tokens and it dies in
`freqs_cis_matrix`:

```
RuntimeError: cannot reshape tensor of 0 elements into shape [2, 0, 32, -1]
```

Every upstream 2.3 workflow concatenates an audio latent for exactly this reason.
These graphs do the same and then discard the audio half:

```
VAELoader(LTX23_audio_vae_bf16) ─┐
                                 ├─ LTXVEmptyLatentAudio ─┐
EmptyLTXVLatentVideo ────────────────────────────────────┴─ LTXVConcatAVLatent
                                                              └─ sampler ─ LTXVSeparateAVLatent
```

`LTXVAudioVAELoader` also lists from `checkpoints`, but core **`VAELoader`** reads
`models/vae`, where `LTX23_audio_vae_bf16.safetensors` lives.

## Image to video

Three of the four I2V graphs share one conditioning path:

```
LoadImage → ImageScale (lanczos, centre crop) → LTXVPreprocess (crf 18)
          → LTXVImgToVideoConditionOnly → LTXVConcatAVLatent → sampler
```

`LTXVImgToVideoConditionOnly` VAE-encodes the image, **writes it over the first
latent frame** and sets a noise mask of `1 - strength` there (`latents.py:582-599`).
It never touches the conditioning, so it composes with CFG, STG, APG and a
distilled LoRA without going anywhere near the guide-mask attention path.

### `strength` is the dial that matters

It is adherence to the still, and it is inverted into a noise mask:

| strength | behaviour |
|---|---|
| 1.0 | first frame locked. Sharpest match, stiffest motion, most likely to sit still before anything moves. |
| **0.7** | **the default here.** Holds the composition, lets the model re-time and re-light the opening. |
| 0.4–0.5 | treats the image as a strong composition/style hint. Use when the still is off-distribution — an illustration, a screenshot, a 3D render. |
| 0.0 | equivalent to T2V. |

### Three things that are easy to get wrong

- **Keep `LTXVPreprocess`.** LTX was trained on compressed video. A pristine still
  is off-distribution, and the model spends its early steps correcting it — which
  shows up as a texture pop between frame 0 and frame 1. `img_compression = 18` is
  upstream's value.
- **Prompt the motion, not the subject.** The subject is already fully specified by
  the image; describing it again mostly buys a fight between two conditionings.
- **Aspect mismatches are centre-cropped, not letterboxed.** The graphs put an
  `ImageScale` at `crop = center` in front, which also makes the node's own
  internal bilinear resize a no-op, so what the VAE encodes is exactly what you see
  on the `ImageScale` output. If your subject is off-centre, crop it yourself.

### The I2V graphs give you sound

`ltx-2.3-22b-dev` is an audio-video DiT and generates the audio tokens whether or
not you decode them. The T2V files here throw that half away; the I2V files run it
through `LTXVAudioVAEDecode` into `CreateVideo`'s `audio` input. It costs one extra
VAE decode and nothing in sampling, and the saved `.mp4` carries a real AAC track.

### First frame + last frame is built differently

`LTX-2.3_GGUF_I2V_FirstLastFrame_3090.json` uses **`LTXVAddGuideAdvanced`**, which
rewrites the *conditioning* as well as the latent. So the two guides chain through
`positive`/`negative`/`latent` and the `CFGGuider` reads from the second guide, not
from `LTXVConditioning`. Add a third keyframe by extending that chain. It also does
its own resize, `LTXVPreprocess` and optional blur internally (`guide.py:116-133`),
which is why it needs no `ImageScale` in front.

`frame_idx` is `0` for the first frame and `-1` for the last; for 9+ frame videos
anything else is rounded down to a multiple of 8.

**Lowering `strength` below 1.0 there turns SageAttention off**, because a
part-strength guide takes LTX's guide-mask attention path — see the SageAttention
section below. That is the documented fallback, not a bug.

## Measured on this machine

RTX 3090, Q4_0 (11.8 GB), 960×544 / 121 frames, seed 42. See `PROGRESS_REPORT.md`
for the full table and method.

| Config | Steps | s/step | Wall (warm) | Peak VRAM |
|---|---|---|---|---|
| Baseline, cfg 3.0, SDPA | 20 | 11.04 | 230.8 s | 20.9 GB |
| Baseline, cfg 3.0, sage | 20 | 10.50 | 218.9 s | 20.7 GB |
| Distilled, cfg 1.0, SDPA | 8 | 11.07 | 98.2 s | 20.9 GB |
| **Distilled, cfg 1.0, sage** | **8** | **10.68** | **94.3 s** | **20.8 GB** |
| STG + APG, 20 steps | 20 | 17.29 | 303.1 s | 19.7 GB |

**230.8 s → 94.3 s end to end: 2.45×.**

Three of the four files here have been executed as shipped (baseline, distilled,
STG+APG). The long-clip one has not.

Peak VRAM is **within ~3 GB of the card's limit**. That is the headroom the
Q4_0-over-Q5_0 choice is buying.

**Use the distilled workflow.** It is ~2.5× less sampling time *and* the output is
sharper and better composed than the 20-step baseline. Everything else on this page
is a rounding error next to that.

Two measurements worth internalising if you plan to tune further:

- A **cfg-3.0 step costs the same as a cfg-1.0 step** (11.04 vs 11.07 s/step),
  despite running two conditionings instead of one *and* carrying a 2.55 GB LoRA.
  The per-forward cost is dominated by dequantising 11.8 GB of GGUF weights, not by
  arithmetic. This model is **weight-bandwidth-bound at batch 1**.
- So **the lever is fewer steps, not faster steps.** Kernel-level work (SageAttention
  here, `q8_kernels` on an Ada card) has limited headroom while the bottleneck is
  moving weights.

LoRA on GGUF is effectively free per step, despite ComfyUI-GGUF re-applying patches
after dequantisation on every weight access.

## SageAttention

`sageattention` supports SM 8.6, so unlike `q8_kernels` (which needs SM 8.9 and is
permanently out on a 3090) it **is** available here. Add the
**`🅛🅣🅧 LTXV Attention Backend`** node between the model loader and the guider and
set `backend = sage`.

**Set your expectations from the table above, not from kernel benchmarks.** The
isolated `sageattn` call is ~2.2× faster than SDPA at this clip's attention shape,
but the honest end-to-end figure is **1.06×** — 12 seconds off a 231-second
render. Attention is only a slice of a 22B DiT's step time, and ComfyUI reaches
the kernel through reshapes an isolated benchmark never pays. It is free speed
for equivalent output; it is not a 2× button.

It is opt-in and defaults to `default`, because sage's INT8 path is an
approximation. Over a full 20-step render the latents differ from SDPA by
relative L2 ≈ 0.038 (≈50.6 dB in latent space). Side-by-side frames look
equivalent — same composition, lighting and detail level — but it is not
bit-identical. Two sage runs of the same seed *are* bit-identical to each other,
so it is deterministic.

One more caveat before you conclude it "did nothing":

- **Masked attention gets no speedup at all.** The installed `sageattention` 2.2.0
  takes no `attn_mask` argument, so ComfyUI sets
  `SAGE_ATTENTION_SUPPORTS_MASK = False` and routes every masked call to SDPA.
  LTX's guide-mask path additionally passes `low_precision_attention=False` on
  each sub-call. So any guide with `strength != 1.0` or a `pixel_mask` sees none
  of it. Plain guides at strength 1.0 are unaffected.

### Do not combine it with a fused attention node

KJNodes' `LTX2MemoryEfficientSageAttentionPatch` replaces `attn1.forward`
wholesale via `add_object_patch`. Such a replacement never calls ComfyUI's
`optimized_attention`, which means:

- `LTXV Attention Backend` has no effect on self-attention, and
- **STG silently perturbs the wrong attention** — with `attn1` fused, STG index 0
  lands on the cross-attention.

This pack now logs a warning when it detects that combination. Pick one or the
other.

## Tuning for VRAM

In rough order of effect, when a job will not fit:

1. **`working_device = cpu`** on the decode node. Keeps the full-length output
   accumulator in system RAM. Costs one transfer per chunk, buys back GBs.
2. **Use the spatio-temporal decode** for long clips. The spatial-only node still
   allocates the full-length output buffer, so it does not lower the peak on a
   long clip no matter how many spatial tiles you add.
3. **Lower `temporal_tile_length`**, then raise `spatial_tiles`.
4. **`working_dtype = bfloat16`** — half the accumulator of float32, and on Ampere
   the same throughput as float16 with better range.

The decode estimates the accumulator up front and refuses with a message naming
the shortfall, rather than hanging on a CUDA OOM.

## Before running

Point these at your own files if the names differ:

| Node | Widget | Value used here |
|---|---|---|
| `UnetLoaderGGUF` | `unet_name` | `ltx-2.3-22b-dev-Q4_0.gguf` |
| `DualCLIPLoader` | `clip_name1` | `gemma-3-12b-it-ablit-norms-biproj-fp8mixed.safetensors` |
| `DualCLIPLoader` | `clip_name2` | `ltx-2-3-22b-text_encoder.safetensors` |
| `VAELoader` (video) | `vae_name` | `LTX23_video_vae_bf16.safetensors` |
| `VAELoader` (audio) | `vae_name` | `LTX23_audio_vae_bf16.safetensors` |
| `LoraLoaderModelOnly` | `lora_name` | `ltx\ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors` |
| `LoadImage` (I2V only) | `image` | `example.png` — ComfyUI's own; replace with yours |

Any Gemma-3-12B that ComfyUI can load will do for `clip_name1`, as long as it
carries `spiece_model`.
