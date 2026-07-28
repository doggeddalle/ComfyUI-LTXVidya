# RTX 3090 / LTX-2.3 GGUF workflows

Five workflows for running the LTX-2.3 22B **AV** model from GGUF weights on a
24 GB card. Every node type, socket and widget was read from a live ComfyUI node
registry and each file is validated against it — including that the model
filenames name files that actually exist.

| Workflow | What it does |
|---|---|
| `LTX-2.3_GGUF_T2V_3090.json` | Text → video **with audio**. Start here. |
| `LTX-2.3_GGUF_I2V_3090.json` | Image → video with audio. |
| `LTX-2.3_GGUF_T2A_3090.json` | Text → audio only, no video decode. |
| `LTX-2.3_GGUF_T2V_2Pass_3090.json` | T2V at half res, ×2 latent upscale, partial re-denoise. |
| `LTX-2.3_GGUF_I2V_2Pass_3090.json` | Same, image-conditioned on both passes. |

Regenerate them after renaming a model:

```
python example_workflows/3090-gguf/generate_workflows.py
```

Edit the filename constants at the top of that script rather than hand-editing the
JSON.

## Two things that are easy to get wrong

**LTX-2.3 needs _two_ text-encoder files.** A Gemma-3 12B LLM *and* a separate text
projection, loaded together by `DualCLIPLoaderGGUF` with `type = ltxv`. ComfyUI only
builds the LTX-AV encoder (`ltxav_te` + `LTXAVGemmaTokenizer`) when two state dicts
arrive together — a single-file `CLIPLoader` silently builds a different encoder.
The dual loader lists `.gguf` and `.safetensors` in the same dropdown, so a
quantized Gemma and a bf16 projection mix freely. **Slot order does not matter**;
comfy identifies each file by its contents.

**The audio VAE loads through plain `VAELoader`.** `LTXVAudioVAELoader` reads
`models/checkpoints`; if you run GGUF that folder is usually empty, so its dropdown
is empty too. Core `VAELoader` reads `models/vae` and detects an LTX audio VAE from
its keys, building the same object. There are two `VAELoader` nodes in these graphs
— one video, one audio.

## Models expected

| Slot | File |
|---|---|
| `UnetLoaderGGUF` | `ltx-2.3-22b-dev-Q4_0.gguf` |
| `DualCLIPLoaderGGUF` clip 1 | `gemma-3-12b-it-heretic-x.IQ4_XS.gguf` |
| `DualCLIPLoaderGGUF` clip 2 | `ltx-2.3_text_projection_bf16.safetensors` |
| video `VAELoader` | `LTX23_video_vae_bf16.safetensors` |
| audio `VAELoader` | `LTX23_audio_vae_bf16.safetensors` |
| `LatentUpscaleModelLoader` (2-pass only) | `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` |

Both text-encoder files must sit in `models/text_encoders`.

## Optional extras, built in

Rather than shipping separate variants, each workflow carries its alternatives
inline:

- **STG** — `LTX Apply STG` sits in the model chain **bypassed** (Ctrl+B toggles
  it). Bypass passes MODEL straight through, so it costs nothing while off. A
  `STGGuiderAdvanced` (with APG already enabled) sits **muted** next to the active
  `CFGGuider`; unmute it with Ctrl+M and move the `GUIDER` link.
- **Long clips / tight VRAM** — a muted `LTXVSpatioTemporalTiledVAEDecode` sits
  beside the active `LTXVTiledVAEDecode`. Use it for long clips: the spatial-only
  node still allocates the full-length output buffer no matter how many spatial
  tiles you give it, so tiling alone does not lower the peak.

Muted nodes are excluded from execution, so they cost nothing until enabled.

## VRAM, in the order worth trying

1. `working_device = cpu` on the decode node — keeps the full-length output
   accumulator in system RAM.
2. Switch to the spatio-temporal decode for anything long.
3. `working_dtype = bfloat16` — half the accumulator of float32, same speed as fp16
   on Ampere.
4. Raise the tile counts.
5. On the 2-pass workflows, lower `LTXVLatentUpsamplerTiled.tile_size` — the
   upsampler runs while the model is still resident.

The decode estimates its accumulator up front and refuses with a message naming the
shortfall, rather than hanging on a CUDA OOM.

## The first-pass-audio toggle (2-pass only)

`Force First Pass Audio (1) or Resample Audio (2)`:

- **1** — pass 1's audio latent passes through a zero `SolidMask`, so pass 2
  denoises none of it and the audio you heard in pass 1 survives untouched while the
  video is upscaled.
- **2** — the raw latent goes through instead, letting the partial denoise refine
  the audio alongside the video.

Both options reuse pass 1's audio deliberately. Starting pass 2 from an *empty*
audio latent would not work — it begins from a mid-schedule sigma, so there is no
path from silence to a sensible result.

## Dependencies

ComfyUI-GGUF for the loaders, plus this node pack. The **two 2-pass workflows also
need `TwoWaySwitch`**; a missing switch node makes the graph fail to load rather
than merely degrade. The other three are free of it.

## Caveat

These are validated structurally — node types, sockets, types, widget counts and
model filenames all check out against the installed nodes. They have **not** been
executed to produce a frame. Treat the first run of each as a smoke test.
