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

## Workflows

`example_workflows/3090-gguf/` — three workflows built for this machine. Every node
type, socket name, socket type and widget order was read from the **live ComfyUI node
registry on this install**, and each file is validated against those same schemas.

| File | Purpose |
|---|---|
| `LTX-2.3_GGUF_T2V_Baseline_3090.json` | Minimal GGUF text-to-video; the reference point |
| `LTX-2.3_GGUF_T2V_STG_APG_3090.json` | STG + APG, exercising the guidance fixes |
| `LTX-2.3_GGUF_LongClip_LowVRAM_3090.json` | 1280×704 / 193 frames, spatio-temporal decode, CPU accumulator |

A finding that shaped these: **`models/checkpoints` is empty on this install**, so
`LTXAVTextEncoderLoader` and `LTXVGemmaCLIPModelLoader` — both of which list from
that folder — cannot be configured, and the shipped upstream examples that use them
will not load correctly here. The workflows therefore use core **`CLIPLoader` with
type `ltxv`** against `ltx-2-3-22b-text_encoder.safetensors`, which is the path that
actually works with a GGUF DiT.

They are also **video-only**: `LTXVAudioVAELoader` lists from `checkpoints` too, so
the AV branch of the upstream examples is not configurable here either.

---

## Verification

- **82 tests**, run on the real ComfyUI interpreter (torch 2.10.0+cu130), passing.
  ComfyUI is stubbed in `tests/conftest.py`; torch and kornia are real, so the kornia
  checks exercise the actually-installed version.
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

**No frame has been generated.** Correctness is pinned by tests and by A/B
measurement against the previous implementation, but no LTX-2.3 GGUF render has been
run end to end. The tests that matter most in practice:

1. A real render with the per-step normalization patchers enabled, before/after.
2. A long/high-res decode recording `max_memory_allocated()`.
3. Whether **Q5_0 now completes where it previously OOM'd**. Note the 1.87 GiB freed
   does **not** by itself bridge the ~3.7 GB gap between Q4_0 and Q5_0 — combined
   with `working_device=cpu` it may, but that is untested.

---

## Pinned — not yet done

| Item | Notes |
|---|---|
| Cosmetic | `low_vram_loaders.py` display name `"Low VRAMLoad …"` missing a space; `tricks/__init__.py` node-id typo `LTXAttentioOverride` (needs an **alias**, not a rename — a bare rename breaks saved workflows); `README.md:22` broken Markdown link; `README.md:23` says "32GB+ VRAM" without pointing 24 GB owners at `low_vram_loaders.py`. |
| Connector load cost | `text_embeddings_connectors.py` deserializes the entire multi-GB checkpoint to extract a few MB of connector weights, and the same file is loaded again by the model loader. Lazy safetensors key access would cut load time and transient host RAM. |
| `pyproject.toml` + CI | No machine-readable package metadata for the Comfy Registry, and the test suite has no workflow to run it — so a regression like the kornia break could land again unnoticed. Upstream PRs #330/#331 cover the packaging half. |

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
