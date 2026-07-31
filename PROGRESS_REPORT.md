# ComfyUI-LTXVidya — progress report

Fork of [`Lightricks/ComfyUI-LTXVideo`](https://github.com/Lightricks/ComfyUI-LTXVideo),
tuned for a single **RTX 3090** running **LTX-2.3 GGUF** models.

Branch point: upstream `3b9c5cd`. Current head: `6fdbf7b`.
All figures below were measured on the target machine, not estimated.

---

## Why this work happened

The node pack would not load at all (a kornia incompatibility), and once loading,
several code paths either crashed or wasted large amounts of time and VRAM. The
governing constraint throughout:

> *"I mainly use Q4_0 primarily to avoid risk of OOM hanging situation.
> ComfyUI is never kind to thin margins."*

VRAM headroom is being paid for in **model quality** — `ltx-2.3-22b-dev-Q4_0.gguf`
(11.8 GB) is chosen over the Q5_0 finetune (15.5 GB) purely for safety margin. That
makes memory work worth real quality, not just convenience.

### Target environment (verified, not assumed)

| | |
|---|---|
| GPU | RTX 3090, 24 GB, **compute capability 8.6** |
| ComfyUI | `ComfyUI-Easy-Install`, `python_embeded` (Python 3.12.10) |
| torch | 2.10.0+cu130 |
| kornia | 0.7.4 |
| Model path | `UnetLoaderGGUF` (ComfyUI-GGUF) |
| `q8_kernels` | not installed |

### What the hardware rules out

- **Q8/FP8 kernels are permanently unavailable.** `q8_kernels` requires SM 8.9
  (Ada); this is SM 8.6. LTXV's headline ~3× speedup cannot apply here.
- **`torch.compile` is a poor fit** — ComfyUI-GGUF dequantizes through custom ops,
  and this pack installs many `set_model_patch_replace` hooks (STG, PAG, FETA,
  attention bank).
- **TF32 buys little** — it affects only fp32 matmuls; GGUF dequantizes to
  bf16/fp16.

So the gains had to come from **removing CPU-side overhead** and **reducing peak
VRAM**, not from kernel or dtype tricks.

> **Correction (Phase 7).** That last sentence was wrong as reasoning, even though
> it turned out roughly right in effect. `q8_kernels` is excluded by SM 8.9, but
> **SageAttention 2 supports SM 8.6** and was already installed in this
> `python_embeded` the whole time — unused, because ComfyUI is launched with no
> flags. It went unexamined for six phases because "no kernel tricks" had been
> treated as settled. Measured end to end it is worth **1.06×**, not the ~2.2× the
> isolated kernel suggests, so the *conclusion* survives; the *premise* did not.
> See Phase 7.

---

## Phase 0 — Making it load (`3f5a5ff`)

Six open upstream PRs targeted the same break; two more clusters were live crashes.

| Issue | Effect | Upstream PRs |
|---|---|---|
| kornia removed the `pad` re-export | `ImportError` at import of `pyramid_blending`, which `__init__.py` imports — **the entire node pack failed to load** | #527 #526 #516 #508 #506 #498 |
| `apg()` called with `momentum_buffer=` it doesn't accept | `TypeError` on every sample with APG enabled | #500 |
| `noise_pred_neg` / `noise_pred_perturbed` bound only inside conditional branches | `UnboundLocalError` at cfg=1.0 (distilled) or stg=0 | #460 #507 |
| STG attention index miscount | shape crash + silently shifted indices with guide-mask attention | #503 #501 |
| Case-sensitive config assert | blocked loading checkpoints writing `PER_TOKEN_RMS` | #504 #478 |

Two decisions worth recording:

- **The kornia fix goes further than any of the six PRs.** They all keep importing
  `is_powerof_two` / `find_next_powerof_two` from kornia. Those have *never* been in
  kornia's public `__all__` — the same fragility waiting to recur. They are four
  lines, so they are now defined locally and only the genuinely public pyramid API
  is imported. Upstream PR #527's `kornia<0.9` pin was **not** adopted; the fix is
  version-agnostic and a ceiling would manufacture a future break.
- **The STG fix supersedes both upstream attempts.** Verified against ComfyUI's
  source: `_attention_with_guide_mask` splits one self-attention into up to *three*
  `optimized_attention` calls. PR #503 detects sub-calls via
  `low_precision_attention=False`, but ComfyUI's *third* sub-call omits that flag, so
  it still miscounts. PR #501 keys off an `audio_attn1` attribute ComfyUI does not
  have. The fix here uses the flag only to *enter* partition mode, then tracks a
  running query offset until the slices cover the sequence.

> **Note on kornia:** this machine has kornia **0.7.4**, which still exports `pad`.
> In this environment the fix is forward-compatibility insurance rather than what
> unblocked the import; the 0.8.x break is real but would have bitten on the next
> kornia upgrade.

---

## Phase 1 — Peak VRAM in the tiled VAE decode (`eeab61a`)

The blend mask varies only over `(height, width)` — it is identical for every frame
and every batch item. Both weight buffers were nonetheless allocated at full frame
depth:

```python
weights      (batch, image_frames, H, W, 1)
tile_weights (batch, image_frames, tile_h, tile_w, 1)
```

At the **default 1×1 tiling** those are two full-length buffers, each one third the
size of the output accumulator, holding the same 2-D mask repeated once per frame.
Allocating them as `(1, 1, H, W, 1)` broadcasts to an identical result.

**Measured A/B on the 3090** (fp16 accumulator, synthetic VAE, `max_memory_allocated`):

| Config | Before | After | Saved |
|---|---|---|---|
| 720p 121 frames, 1×1 | 2.234 GiB | 1.831 GiB | 0.403 GiB (18.0%) |
| 720p 121 frames, 2×2 | 1.267 GiB | 1.010 GiB | 0.256 GiB (20.2%) |
| 720p 241 frames, 1×1 | 4.450 GiB | 3.644 GiB | 0.806 GiB (18.1%) |
| 1080p 241 frames, 1×1 | 10.315 GiB | 8.447 GiB | **1.868 GiB** (18.1%) |

Also in this phase:

- **`bfloat16` added to `working_dtype`** — matches fp16 throughput on Ampere with
  better range, and halves accumulator memory versus float32.
- **Preflight VRAM check.** The output accumulator's size is known exactly in
  advance, so it is now compared against `get_free_memory()` and refused with a
  message naming the knobs to change. ComfyUI tends to *hang* rather than report
  cleanly on a hard CUDA OOM, so refusing up front is strictly better.
- `working_dtype` resolution no longer leaves `target_dtype` unbound for an
  unrecognised value.
- Progress bar over the tile loop; tooltips on the tiling controls, which had none.

---

## Phase 2 — Per-step normalization (`fa1156f`)

`latent_norm.py` walked the latent with nested Python loops, issuing one tiny CUDA
kernel per `(batch, channel[, frame])` element. `LTXVPerStepAdainPatcher` and
`LTXVPerStepStatNormPatcher` register these as `sampler_post_cfg_function` — so the
whole nest ran **on every denoising step**. At `[1, 128, 25, 48, 80]` the AdaIn
`per_frame` path issued **6,400 `std_mean` launches per step**; the GPU sat idle
between launches while Python drove the loop.

**Measured on the 3090, real node code:**

| Operation | Before | After | Speedup |
|---|---|---|---|
| `LTXVAdainLatent` `per_frame=True` | 712.74 ms | 0.82 ms | **870×** |
| `LTXVAdainLatent` `per_frame=False` | 26.57 ms | 1.04 ms | 25.6× |
| `LTXVStatNormLatent` `clip=False` | 181.41 ms | 5.96 ms | 30.4× |
| `LTXVStatNormLatent` `clip=True` | 220.94 ms | 7.36 ms | 30.0× |

Over a 30-step sample that is roughly **21 s per generation** removed for AdaIn and
**~5 s** for StatNorm. Peak allocation is unchanged — this is purely launch overhead.

StatNorm keeps every branch of the original: the Bessel-corrected std over in-range
values only, optional outlier clipping, the shift-by-mean fallback for near-zero
spread, and leaving a channel untouched when nothing falls in range. `torch.quantile`
is batched over channels, chunked when an input would exceed its 2²⁴ element limit,
and computed in fp32 (it rejects half/bfloat16).

Three incidental fixes:

- **`copy.deepcopy` of the latent removed** — it cloned the full tensor every step.
- **AdaIn no longer mutates its input.** It wrote a `repeat()` back into the caller's
  `reference` dict; it now broadcasts with `expand()`.
- **Constant channels no longer produce `NaN`.** `0/0` used to propagate through the
  `lerp` into the entire latent. It now falls back to the reference mean — the
  meaningful AdaIn result. *This is an intentional behaviour change*, covered by a
  test asserting the old code did NaN there.

---

## Phase 3 — Per-step syncs and Q8 gating (`9eceeda`)

**Performance**

- **`stg.py project()` cast to `float64`.** Consumer Ampere runs fp64 at 1/64 rate.
  Now fp32: **2.39 ms → 0.73 ms** per call, maximum relative deviation **2.4e-7** —
  far below the precision of the bf16 latents it operates on.
- **`timestep.item()` read twice per step** in `APGGuider`; each is a device→host
  sync that stalls the CUDA stream. Now read once.
- **`dynamic_conditioning.find_step()`** compared tensors inside a Python loop — one
  sync per candidate, every step. The sigma schedule is constant for a run, so it is
  pulled to the host once and cached.

**Q8 nodes — these were crashing wrongly on any non-Ada card**

- `LTXVQ8LoraModelLoader.load_lora` bound `quant_fn = hadamard_transform` *before*
  checking `Q8_AVAILABLE`, so with `q8_kernels` absent — the case on this machine —
  it raised a bare `NameError` and the helpful `ValueError` below was unreachable.
- `check_q8_available()` now also verifies compute capability ≥ 8.9 and names the
  actual GPU: *"requires an Ada-generation GPU or newer … but RTX 3090 reports 8.6"*.
- `LTXVPatcherVAE` imported `q8_kernels` inline with no guard at all.
- Hardcoded `"cuda"` replaced with `get_torch_device()`.

**Robustness**

- `load_stg_presets()` ran `json.load(open(...))` **at import time** with no context
  manager and no explicit encoding — a leaked handle or locale decode error there
  takes down the whole pack.
- `LTXVGemmaCLIPModelLoader` raised an opaque `TypeError` from `Path(None)` when a
  saved workflow referenced a renamed or deleted model. Now `FileNotFoundError`.

**Dependencies** — `diffusers` dropped (a hard requirement with **zero imports**
anywhere; a test now enforces this); lower bounds added for `einops` and `kornia`.

---

## Phase 4 — Allocator hygiene and OOM footguns (`6fdbf7b`)

`soft_empty_cache()` was called **nowhere** in the pack, and `free_memory()` exactly
once. Both tile loops sample or decode one tile at a time and their intermediates are
dead by the end of each iteration. Edge tiles and the first/last temporal chunk have
different shapes to interior ones — the classic pattern that fragments the CUDA
caching allocator over a long job, whose failure mode is a **mid-run OOM on a machine
with enough total memory**.

Cache release added between spatial tiles in `LTXVTiledSampler` and between temporal
chunks in `LTXVSpatioTemporalTiledVAEDecode`, with explicit `del`s so the release has
something to reclaim. Cost is negligible next to a full tile sample or VAE decode.

`LTXVDilateLatent` accepted scales up to 100×100 and allocated two buffers at the
dilated size — a mismatched guide latent could silently request a **10,000×**
allocation. It now checks against free VRAM and raises, naming both shapes and the
likely cause (the scales are normally just the target/guide resolution ratio).

---

## Phase 5 — Categories, dependency isolation, diagnostics (`c443fb9`)

### One menu tree instead of two

80 category declarations used **22 distinct strings across two top-level roots**.
ComfyUI groups the node browser by exact string, so `lightricks/LTXV` (18 nodes) and
`Lightricks/latents` rendered as two separate folders, while bare `sampling`,
`latent`, `utility`, `prompt` and `math/conversion` scattered nodes into ComfyUI's
own menus, and `ltxtricks` / `fluxtapoz` added two more roots.

Everything now sits under one `Lightricks` tree — 18 strings, one root:

```
Lightricks/{sampling, guidance, conditioning, latents, VAE, loaders, prompt,
            masks, image, IC-LoRA, HDR, audio, motion tracking, utility,
            tricks, tricks/attention, tricks/rf-edit}
```

The 18-node catch-all was split by what each node actually *does* rather than
remapped wholesale. Hosted API nodes keep their `api node/...` root deliberately —
ComfyUI collects those in a dedicated section and relocating one would hide it.

`nodes_registry.normalize_category()` now runs from the decorator for both v1
`CATEGORY` attributes and v3 schemas, so a stray category can't reintroduce a second
root. **Categories affect menu placement only** — node type IDs are untouched, so
saved workflows keep working.

### A broken optional dependency costs one node, not the pack

`nodes_registry` has always had a `skip=` mechanism that nothing used. The kornia
import in `pyramid_blending` is now guarded and registers with `skip=`;
`__init__.py` prunes any node exposing `NODE_UNAVAILABLE_REASON`.

The guard catches `Exception`, not `ImportError`, on purpose: a mismatched
kornia/torch build on Windows fails inside the DLL loader with `OSError`, which an
`except ImportError` sails straight past — precisely the scenario that motivated it.

### Attention bank and FETA

- **Attention bank** stashed each block's activations with plain `.cpu()`, which
  allocates pageable memory the driver cannot DMA out of. It now stages through a
  pinned buffer. The device→host copy stays **blocking**: nothing synchronises
  before the bank is read back on a later step, so `non_blocking` there would be a
  race for a few hundred microseconds. The host→device inject is stream-ordered and
  does use `non_blocking=True`.
- **FETA** built a boolean eye mask, expanded it to the full attention shape, and
  produced a masked *copy* of the attention matrix — three full-size allocations per
  FETA-enabled block per step — to compute a sum available as `total - diagonal`. It
  also cast to float32 before the softmax rather than letting `softmax(dtype=)` fuse
  it, and divided by zero for a single-frame clip. Scores are unchanged, asserted
  against the original as an oracle.

### Chunk sizing

`pyramid_blending` hardcoded 8 frames per chunk — wasteful at low resolution, risky
at high resolution on a card already holding a 12 GB quantized model. It now derives
from `get_free_memory()`, with the old constant as the non-CUDA fallback.

**The user-facing tile widgets are deliberately not auto-defaulted from VRAM.**
`INPUT_TYPES` is evaluated once at startup, when free memory says nothing about
conditions at execution time; a default derived there would be confidently wrong.
The preflight check from Phase 1 names the knob to change instead.

### Diagnostics

- **49 `print()` calls → module loggers**, using logging's own `%`-substitution
  rather than pre-formatted strings, so arguments are only rendered if the record is
  actually emitted. Three handlers printed a caught exception and immediately
  re-raised it; removing the print left them doing nothing, so they went too.
- **15 `assert`s → explicit raises.** `assert` is stripped under `python -O`, which
  would silently remove the validation. The ones that merely said two things "must
  match" now report the actual values.

---

## Phase 6 — Load cost, packaging, CI, polish

### Connector weights no longer cost a full checkpoint read

`load_text_embeddings_pipeline` called `load_torch_file` on the LTX-V checkpoint —
deserializing **the entire multi-GB DiT** — to extract a few MB of connector and
projection weights, and the same file was then read again by the actual model
loader.

safetensors keeps an index in its header, so individual tensors can be fetched on
demand and key-existence checks (`is_av`, `has_dual_aggregate`) cost nothing at all.
`_LazyCheckpoint` presents a read-only `Mapping` so `key in sd` / `sd[key]` code is
unchanged, and bulk access goes through `filter_prefix()` rather than `.items()` —
iterating items would pull every tensor into memory and defeat the whole exercise.
That required routing both filter sites (here and in `embeddings_connector`) through
one shared `filter_state_dict_by_prefix()` helper.

Non-safetensors checkpoints, or any failure opening the lazy reader, fall back to
the original full load with a warning.

### Packaging and CI

`pyproject.toml` adds Comfy Registry metadata and consolidates the black / isort /
ruff settings so local runs and CI cannot drift. `PublisherId` is intentionally
blank — it is only needed to push to the registry.

It deliberately carries **no `[tool.pytest.ini_options]`**: the repository root is
itself a Python package, and making it pytest's rootdir would make pytest import
`__init__.py` (and therefore all of ComfyUI) during test setup. `tests/pytest.ini`
keeps the rootdir anchored at `tests/`, and a test now guards that invariant.

`.github/workflows/ci.yml` runs three jobs: the suite on Python 3.10 and 3.12 with
CPU torch and a real kornia; ruff/black/isort at the versions the pre-commit config
pins; and `compileall` over the whole tree — which would have caught the syntax
breakage introduced during the assert sweep in a module no test imports.

While adding this I found `tests/pytest.ini` had been setting `import-mode` as an
ini key. That is a command-line flag, so pytest was silently ignoring it and warning
about an unknown config option. It is now in `addopts`.

CI earned its keep immediately by failing on its first run, for three real reasons a
Windows-only workflow could not have surfaced:

- The tree had been formatted with newer black/isort than the versions this repo's
  own `.pre-commit-config.yaml` pins, and they disagree about when to split an
  import. Reformatted with the pinned versions so local, pre-commit and CI agree.
- `numpy` and `einops` were missing from the CI environment. Both are present
  locally only because ComfyUI happens to install them — `safetensors` needs numpy
  to write a file, and `feta_enhance_utils` imports einops.
- Diagnosing that took longer than it should have, because the test helper left a
  half-initialised module in `sys.modules` after a failed import: only the first
  test reported the real `ModuleNotFoundError` and the rest reported a misleading
  `AttributeError`. The helper now drops the module and re-raises.

All four jobs are green as of `3100b7d`.

### Polish

- **`LTXAttentioOverride`** (missing "n") is the node id stored in every saved
  workflow that uses it. Renaming it would break those on load, so the correct
  spelling is registered as an **additional alias** and the misspelling stays.
- `"Low VRAMLoad Latent Upscale Model"` → `"Low VRAM Load ..."`.
- `README.md` had a broken Markdown link (`(Download here](...)`, no opening
  bracket) and listed "32GB+ VRAM" as a flat prerequisite — which reads as
  "unsupported" to a 24 GB owner. It now says 24 GB cards work via GGUF weights, the
  low-VRAM loaders and the tiled decode nodes.

---

## Phase 7 — the first frame, and what it cost to find out

Every earlier phase was verified by unit tests, oracles and microbenchmarks. None
of it had ever produced a video. Phase 7 ran the workflows on the real machine.

Both of the things that broke were **invisible to schema validation**, which is the
lesson: the previous workflows passed every structural check and still could not
execute a single step.

### What running them actually found

**1. `ltx-2-3-22b-text_encoder.safetensors` is not a text encoder.**

Its safetensors header lists **four tensors** —
`text_embedding_projection.{audio,video}_aggregate_embed.{weight,bias}`. LTX-2.3's
text encoder is **Gemma-3-12B**; this file is the projection that aggregates
Gemma's 49 hidden states (49 × 3840 = 188160) down to the DiT's conditioning dims.

`comfy/sd.py` only reaches the LTX-2.3 branch — `ltxav_te` +
`LTXAVGemmaTokenizer`, `sd.py:1809` — when handed **exactly two** state dicts. A
single-file `CLIPLoader(type=ltxv)` falls into the LTX-**0.9** T5 branch
(`sd.py:1601`), builds a T5 whose weights the file does not contain, and dies:

```
NotImplementedError: Cannot copy out of meta tensor; no data!
```

The fix is `DualCLIPLoader` with `type = ltxv`, Gemma first (`sd.py:1812` reads the
tokenizer from `clip_data[0]["spiece_model"]`, and only the Gemma file carries it).
Both files live in `models/text_encoders`.

This also **retires the "empty `models/checkpoints`" blocker** recorded in earlier
revisions of this document. `LTXAVTextEncoderLoader` needing a checkpoint is not
the constraint it appeared to be — `DualCLIPLoader` does the same job from
`text_encoders`.

**2. The audio branch is not optional.**

`ltx-2.3-22b-dev` is an **AV** DiT. `av_model.py:854` builds audio RoPE
unconditionally, so a video-only latent hands it zero audio tokens:

```
RuntimeError: cannot reshape tensor of 0 elements into shape [2, 0, 32, -1]
```

Every upstream 2.3 workflow concatenates an audio latent for exactly this reason.
The earlier note here that these graphs are "video-only" described something the
model does not permit. `LTXVAudioVAELoader` does list from `checkpoints`, but core
`VAELoader` reads `models/vae`, where the audio VAE actually is. The graphs now
build an AV latent and split the audio back off after sampling.

### The first renders

RTX 3090, `ltx-2.3-22b-dev-Q4_0.gguf`, Gemma-3-12B fp8mixed text encoder,
960×544 / 121 frames, seed 42, `LTXVTiledVAEDecode` 2×2.

| Config | Evals | Wall clock | s/step | Peak VRAM |
|---|---|---|---|---|
| Baseline, 20 steps cfg 3.0, SDPA | 40 | 230.8 s | 11.04 | 20.9 GB |
| Baseline, 20 steps cfg 3.0, sage | 40 | 218.4 s | 10.50 | 20.7 GB |

The output is a correct render of the prompt — coherent dolly motion, wet-asphalt
reflections, stable across all 121 frames.

Peak VRAM lands **~3 GB under the card's limit**, which is a concrete measurement
of the headroom the Q4_0-over-Q5_0 choice buys.

**The shipped JSON files were then executed directly**, converted to API format
rather than rebuilt by the harness — because the files are what a user loads, and
the files are what was wrong before:

| File | Wall | Peak VRAM |
|---|---|---|
| `..._T2V_Baseline_3090.json` | 230.8 s | 20.9 GB |
| `..._T2V_Distilled_3090.json` | 100.9 s | 18.4 GB |
| `..._T2V_STG_APG_3090.json` | 303.1 s | 19.7 GB |

The STG run is the first end-to-end exercise of the **Phase 0 guidance fixes** on
the real model: `apply_apg=True` completes instead of raising `TypeError`, and the
STG index counting holds. It costs 17.3 s/step against the baseline's 11.04,
which is the extra perturbed forward doing what it is supposed to.

### SageAttention: available, and smaller than it looks

`sageattention 2.2.0+cu130torch2.10` and `triton` were already installed in this
`python_embeded`, and ComfyUI is launched as `python.exe -I -W ignore::FutureWarning
ComfyUI\main.py` — **no flags**, so none of it was in use. SageAttention 2 supports
SM 8.0/8.6, so unlike `q8_kernels` it is not excluded by this card.

An isolated benchmark of `sageattn` at LTX-2.3's attention geometry (32 heads ×
128 head_dim, bf16) shows **2.16×** at 8160 tokens and **2.33×** at 22000. Those
numbers are real and they are also **not what you get**:

| | s/step | end to end |
|---|---|---|
| SDPA | 11.04 | 230.8 s |
| sage | 10.50 | 218.4 s |
| | | **1.06×** |

The gap has two causes. Attention is only a slice of a 22B DiT's step time — the
projections and MLP are untouched — and ComfyUI reaches the kernel through
`attention_sage`'s reshape path, which an isolated HND-layout benchmark never pays.
**Recorded because the first extrapolation in this work was ~2× optimistic**, and a
kernel microbenchmark is not a system measurement.

Quality: over a full 20-step render the sage latents differ from SDPA by relative
L2 **0.038** (≈50.6 dB in latent space). Side-by-side frames are equivalent — same
composition, lighting and detail. Two sage runs of one seed are bit-identical, so
it is deterministic.

It is also **not free on guided workflows**: the installed `sageattn` takes no
`attn_mask`, so core sets `SAGE_ATTENTION_SUPPORTS_MASK = False` and demotes every
masked call to SDPA, and LTX's guide-mask path additionally passes
`low_precision_attention=False`. Any guide with `strength != 1.0` or a `pixel_mask`
gets none of it.

### The distilled path — where the time actually is

`ltx-2.3-22b-distilled-1.1_lora-dynamic…` at strength 1.0, upstream's 9-sigma
schedule (**8 steps**) and **cfg 1.0**, against the baseline's 20 steps at cfg 3.0.
cfg 1.0 is the branch Phase 0's `UnboundLocalError` fix unblocked and which had
never been exercised on the real model.

| Config | Steps | s/step | Sampling | Wall (warm) | Peak VRAM |
|---|---|---|---|---|---|
| Baseline, cfg 3.0, SDPA | 20 | 11.04 | 220.8 s | 230.8 s | 20.9 GB |
| Baseline, cfg 3.0, sage | 20 | 10.50 | 210.0 s | 218.9 s | 20.7 GB |
| Distilled, cfg 1.0, SDPA | 8 | 11.07 | 88.6 s | 98.2 s | 20.9 GB |
| **Distilled, cfg 1.0, sage** | **8** | **10.68** | **85.4 s** | **94.3 s** | **20.8 GB** |

**230.8 s → 94.3 s end to end — 2.45× — and the output is better.** At 8 steps the
distilled
render is sharper and better composed than the 20-step baseline. A control run at
the same 8 steps *without* the LoRA produces unusable mush, which confirms the LoRA
is doing the work rather than the step count alone.

Two things fall out of this table that reframe the optimisation problem:

- **A cfg-3.0 step costs the same as a cfg-1.0 step** (11.04 vs 11.07 s/step), even
  though the former runs two conditionings and the latter one — and the latter also
  carries a 2.55 GB LoRA. So the per-forward cost is dominated by something
  batch-independent. For GGUF that is dequantising 11.8 GB of weights on every
  forward: **this model is weight-bandwidth-bound at batch 1, not compute-bound.**
- Which is exactly why SageAttention only returns ~4–5%. Making the *maths* faster
  cannot help much when the bottleneck is moving weights. **The lever on this
  hardware is fewer steps, not faster steps** — and `q8_kernels` being unavailable
  matters far less than six phases of this document assumed.

**LoRA on GGUF is nearly free per step.** ComfyUI-GGUF applies LoRA patches after
dequantisation on every weight access (`ComfyUI-GGUF/ops.py:170-190`), which looked
like a serious recurring cost for a rank-111 2.55 GB LoRA. Measured, it is inside
the noise: 11.07 s/step with the LoRA versus 11.04 s/step without.

*Method note:* wall-clock alone was misleading here — the first distilled pass read
140 s and the second 94 s, a difference that is entirely model **load** after a
swap, not sage. Every figure above is the sampler's own s/step, taken from ComfyUI's
progress bar, with each configuration run twice warm.

### `LTXVAttentionBackend`

A `MODEL → MODEL` node installing
`transformer_options["optimized_attention_override"]`, the hook `wrap_attn` consults
(`comfy/ldm/modules/attention.py:166-182`). Scoped to one model; no CLI flag, no
restart. **Opt-in — it defaults to `default`.**

- The backend list is read from `REGISTERED_ATTENTION_FUNCTIONS` **at call time**,
  so on a machine without sageattention the option is simply absent rather than a
  `NameError`, and `VALIDATE_INPUTS` explains the situation when a workflow saved
  elsewhere is opened.
- It routes through core's own registered function via `.__wrapped__`, which
  preserves core's `low_precision_attention` bailout and its exception fallback.
  KJNodes' equivalent calls its own sage wrapper and passes `attn_mask=` into a
  `sageattn` that has no such parameter, where it is silently discarded — i.e. it
  computes *unmasked* attention on masked inputs.
- `"optimized"` is deliberately never offered: `get_attention_function("optimized")`
  returns the live module global, which inside an STG block is STG's own
  `stg_attention`, and resolving it would double-advance STG's attention counter.
- No recursion: `wrap_attn` sets `_inside_attn_wrapper` **before** invoking the
  override, so core's internal fallbacks do not bounce back in. A test AST-compares
  the stubbed `wrap_attn` against ComfyUI's so that guarantee cannot rot.

**STG composes with it** — STG rebinds the module global to an undecorated method
and delegates to the binding it captured, which still consults the override.

It does **not** compose with fused attention nodes that replace `attn1.forward` via
`add_object_patch` (KJNodes' `LTX2MemoryEfficientSageAttentionPatch`). Those never
call `optimized_attention`, so the override does nothing *and STG silently perturbs
the wrong attention* — with `attn1` fused, STG index 0 lands on the cross-attention.
`attention_compat.warn_if_attention_bypassed()` now detects that and warns once, at
guider construction rather than per step.

---

## Workflows

`example_workflows/3090-gguf/` — eight workflows built for this machine.

| File | Purpose | Executed |
|---|---|---|
| `LTX-2.3_GGUF_T2V_Baseline_3090.json` | Minimal GGUF text-to-video; the reference point | ✅ |
| `LTX-2.3_GGUF_T2V_Distilled_3090.json` | **The fast path** — distilled LoRA, 8 steps, cfg 1.0, SageAttention | ✅ |
| `LTX-2.3_GGUF_T2V_STG_APG_3090.json` | STG + APG, exercising the guidance fixes | ✅ |
| `LTX-2.3_GGUF_LongClip_LowVRAM_3090.json` | 1280×704 / 193 frames, spatio-temporal decode, CPU accumulator | ❌ |
| `LTX-2.3_GGUF_I2V_Distilled_3090.json` | **The fast I2V path** — image conditioning + distilled LoRA | ✅ |
| `LTX-2.3_GGUF_I2V_Baseline_3090.json` | Undistilled I2V reference, 20 steps, cfg 3.0 | ✅ |
| `LTX-2.3_GGUF_I2V_STG_APG_3090.json` | STG + APG on top of image conditioning | ✅ |
| `LTX-2.3_GGUF_I2V_FirstLastFrame_3090.json` | Two keyframes, via `LTXVAddGuideAdvanced` | ✅ |

> **The first three revisions of these files could not run at all.** They were
> validated against the node registry — types, sockets, widget order — and that
> validation passed, which is exactly why the defects survived. Phase 7 replaced
> them; see *Phase 7 → What running them actually found*.

---

## Verification

- **114 tests**, run on the real ComfyUI interpreter (torch 2.10.0+cu130), passing.
  ComfyUI is stubbed in `tests/conftest.py`; torch and kornia are real, so the kornia
  checks exercise the actually-installed version.
- **Real renders.** Since Phase 7 the workflows are executed through ComfyUI's HTTP
  API, not merely validated against the node schema — which is what finally caught
  two defects that had passed schema validation three revisions running.
- **Equivalence over assertion.** The original nested-loop implementations are kept
  verbatim in the test suite as oracles, and the vectorized versions are asserted
  equal to them across `per_frame`, `factor`, `percentile`, `clip_outliers`,
  non-zero targets, single-frame references and degenerate channels.
- **Property-based checks where they are stronger than a golden value.** The tiled
  decode is asserted to reproduce the *untiled* decode exactly across several tiling
  configurations — the property that breaks first if a weight broadcast is wrong.
- **A/B against the pre-change code**, extracted from git, for every performance
  claim in this document.
- `ruff` (repo ruleset) and `black`/`isort` clean.

### What has *not* been verified

Frames now exist, so the old blanket caveat is retired. What remains open:

1. **The per-step normalization patchers have not been exercised in a real render.**
   Their 870× / 30× speedups are measured against the original implementation in
   isolation, not in a generation.
2. **The 1280×704 / 193-frame long-clip workflow has not been run.** It is the one
   shipped file still unexecuted, and it is what would validate the Phase 1 VRAM
   work and the spatio-temporal decode at scale.
3. **Whether Q5_0 now completes where it previously OOM'd.** Baseline Q4_0 peaks at
   20.9 GB of 24 GB and Q5_0 is ~3.7 GB larger, so on this evidence it will not fit
   — with or without `working_device=cpu` on the decode. Not attempted.
4. **Nothing in Appendix A** has been fixed or re-measured.

---

## Pinned — not yet done

| Item | Notes |
|---|---|
| The deferred audit | Phase 7 catalogued but did not fix the items in **Appendix A**. |
| `torch.compile` under GGUF | Ruled out on reasoning, not measurement. If ComfyUI-GGUF's op coverage improves it may become worth re-testing, but the `set_model_patch_replace` hooks remain a real obstacle. |
| Registry publish | `PublisherId` in `pyproject.toml` is blank; fill it in only if you actually want this fork on the Comfy Registry. |

---

## Commit log

| Commit | Summary |
|---|---|
| `3f5a5ff` | Fix kornia 0.8.x import break and three other crash paths |
| `eeab61a` | Cut tiled decode peak VRAM ~18% by de-duplicating blend weights |
| `fa1156f` | Vectorize per-step normalization (up to 870×) |
| `9eceeda` | Remove per-step syncs, gate Q8 nodes on Ada, tidy dependencies |
| `6fdbf7b` | Release allocator cache between tiles, bound latent dilation |
| `e8360b0` | Add this report and the RTX 3090 / LTX-2.3 GGUF workflows |
| `c443fb9` | Unify node categories, isolate optional deps, sweep diagnostics |

---

## Appendix A — audited, not yet fixed

Found during Phase 7's exploration and verified by reading the source. Recorded so
they are not re-discovered from scratch. None of it is fixed.

### Per-step CPU cost

- **`stg.py:447`** (dead twin at `:293`) — `patch_model()` re-runs **every step**,
  looping all 48 transformer blocks through `set_model_patch_replace`; each call
  copies three dicts and `patches_replace["dit"]` grows to N, so this is O(N²)
  dict-entry copies per step. **And it is dead work**: it writes
  `self.model_patcher.model_options`, not the snapshot `predict_noise` was handed.
  The dynamic per-sigma layer switch works only via the `self.stg_flag.skip_layers`
  mutation on the line above, which must be kept.
- **`stg.py:433`** — `if timestep > self.skip_steps_sigma_threshold` on a CUDA
  tensor: a blocking `__bool__` sync every step, before any work is issued.
  **`stg.py:409`** — one sync per `sigma_list` element per step. This is the same
  pattern already fixed in `dynamic_conditioning.find_step()`.
- **`stg.py:288-293`** — dead branch. `model_options["sigma_to_params_mapping"]` is
  never written anywhere in the pack, and it unpacks the 4-tuple in a **different
  order** than `stg.py:441` does, so it would be wrong if it ever fired.
- **`stg.py:27-31`**, **`guiders/parameters.py:44-49`** — the guidance expression
  evaluates `(cfg-1)*(pos-neg)` and `stg*(pos-perturbed)` even when those terms are
  the literal `0` (`stg.py:308-309, 457-458`) — ~6–8 provably-zero full-latent
  temporaries per step, on exactly the cfg=1.0 distilled path Phase 7 introduced.
- **`stg.py:256-262`** — every block is wrapped though only 1–2 skip; per block per
  forward that costs a `nullcontext()`, a fresh `[0]` list literal built *before*
  the `do_skip` guard, and an O(len) list scan.
- **`tricks/nodes/ltx_flowedit_nodes.py:116`** — `torch.randn` on the **CPU** inside
  the step loop, then a host→device copy, every step.
- **`tricks/modules/ltx_model.py:176-199`** — patchify, a fp32 pixel-coord copy and
  the RoPE `_precompute_freqs_cis` recomputed every forward from run-constant input.

### Allocation / OOM-class

- **`hdr.py:69-197`** — ~8–10 GB of fp32 video temporaries at 121 × 720p; the
  `torch.where` calls force both branches live. `:223` does a full-video
  `.cpu().numpy()` instead of streaming frame by frame.
- **`sparse_tracks.py:282-327`** — a 1080p fp32 render buffer plus three full copies,
  ~6.4 GB peak; `:376` reallocates a full H·W buffer per frame.
- **`tiled_sampler.py:133-135`** and **`looping_sampler.py:908-917`** — full-latent
  `weights` buffers whose content only varies over (H, W). This is exactly the
  Phase 1 de-duplication, never applied to the samplers.
  **`tiled_sampler.py:317-322`** additionally accumulates **bitwise identical**
  values into two separate buffers from a duplicated multiply.
- **`latents.py:752-761`** — a mask allocated at full latent channel count (128×
  oversized for LTX-2), and with no `device=`.
- **`vanish_nodes.py:145`** — three full video temporaries where `torch.lerp` needs
  one. **`pyramid_blending.py:233-238`** — 2× the full video live at the `torch.cat`.

### Correctness

- **`latent_norm.py:247-253`** — the per-step `step` counter is **never reset between
  sampler invocations**, so a multi-chunk `LTXVLoopingSampler` run walks off the end
  of `factors` and pins the last value for every later chunk.
- **`tricks/modules/ltx_model.py:127-171`** — stale against current core: does not
  accept `self_attention_mask` / `prompt_timestep`, which core now passes
  (`comfy/ldm/lightricks/model.py:1302, 1308`), and `LTXModifiedCrossAttention` has
  no `GuideAttentionMask` branch. `ModifyLTXModel` + guides therefore crashes,
  taking the attention-bank / FETA / PAG-layer family with it.
- **`tricks/utils/feta_enhance_utils.py:35`** reads
  `transformer_options["original_shape"]`, which the LTXV model never sets →
  `KeyError` whenever `feta_weight > 0`.
- **`vae_patcher.py`** is gated behind `check_q8_available()`, so its VAE memory
  estimator is unreachable on SM 8.6 and ComfyUI's stock estimate is used instead.

### Core-side, needs its own design pass

- **`comfy/ldm/lightricks/model.py:984`** rebuilds the guide self-attention mask
  inside `forward` — once per cond per step, so 3× per step under STG — including a
  ~223 MB host→device upload of the full pixel mask, a full `F.interpolate`, and a
  `torch.zeros((1,1,tracked,total))` that reaches ~793 MB bf16 for an IC-LoRA guide
  the length of the output. Every input is constant for a run. It is triggered by
  *this pack's* guide nodes, so a pack-side fix (ship a pre-downsampled mask) may be
  legitimate.
