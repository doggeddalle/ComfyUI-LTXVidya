# RTX 3090 / LTX-2.3 GGUF workflows

Built and validated against the node registry of a live ComfyUI install with an
RTX 3090 (24 GB, compute capability 8.6) running LTX-2.3 GGUF models. Every node
type, socket name, socket type and widget order was read from that registry rather
than written by hand.

| Workflow | What it's for |
|---|---|
| `LTX-2.3_GGUF_T2V_Baseline_3090.json` | Minimal GGUF text-to-video. Start here. |
| `LTX-2.3_GGUF_T2V_STG_APG_3090.json` | Adds STG + APG guidance. |
| `LTX-2.3_GGUF_LongClip_LowVRAM_3090.json` | 1280×704 / 193 frames with the spatio-temporal decode and a CPU-side accumulator. |

## Why these don't look like the upstream examples

**They load the text encoder differently.** The upstream 2.3 examples use
`LTXAVTextEncoderLoader` (and the pack also offers `LTXVGemmaCLIPModelLoader`). Both
list their checkpoint from `models/checkpoints`. If you run GGUF that folder is
typically empty, so **neither node can be configured** — the dropdown has no options.

These workflows use core **`CLIPLoader` with `type = ltxv`** pointed at
`ltx-2-3-22b-text_encoder.safetensors` in `models/text_encoders`, which is the path
that works alongside a GGUF DiT loaded through `UnetLoaderGGUF`.

**They are video-only.** `LTXVAudioVAELoader` also lists from `models/checkpoints`,
so the audio branch of the upstream examples is not configurable in a GGUF-only
install either.

## Before running

Point these three widgets at your own files:

| Node | Widget | Value used here |
|---|---|---|
| `UnetLoaderGGUF` | `unet_name` | `ltx-2.3-22b-dev-Q4_0.gguf` |
| `CLIPLoader` | `clip_name` | `ltx-2-3-22b-text_encoder.safetensors` |
| `VAELoader` | `vae_name` | `LTX23_video_vae_bf16.safetensors` |

`working_dtype = bfloat16` requires the updated node pack (commit `9eceeda` or
later). On an older pack that option does not exist — choose `auto`.

## Tuning for VRAM

In rough order of effect, when a job will not fit:

1. **`working_device = cpu`** on the decode node. Keeps the full-length output
   accumulator in system RAM. Costs one transfer per chunk, buys back GBs.
2. **Use the spatio-temporal decode** for long clips. The spatial-only node still
   allocates the full-length output buffer, so it does not lower the peak on a long
   clip no matter how many spatial tiles you add.
3. **Lower `temporal_tile_length`**, then raise `spatial_tiles`.
4. **`working_dtype = bfloat16`** — half the accumulator of float32, and on Ampere
   the same throughput as float16 with better range.

The decode now estimates the accumulator up front and refuses with a message naming
the shortfall, rather than hanging on a CUDA OOM.

## A caveat worth reading

These graphs are validated structurally — node types, sockets, types and widget
counts all match the installed nodes. They have **not** been executed to produce a
video. Treat the first run as a smoke test, and check that the three filenames above
match your install.
