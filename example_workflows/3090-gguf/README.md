# RTX 3090 / LTX-2.3 GGUF workflows

 Five workflows for running the LTX-2.3 22B **AV** model from GGUF weights on a 24 GB card. Built for a single **RTX 3090** (24 GB, compute capability 8.6) running LTX-2.3 GGUF weights. Every node type, socket and widget was read from a live ComfyUI node registry and each file is validated against it — including that the model filenames name files that actually exist.

| Workflow | What it does |
|---|---|

| Workflow | What it does |
|---|---|
| `LTX-2.3_GGUF_T2V_3090.json` | Text → video **with audio**. Start here. |
| `LTX-2.3_GGUF_T2V_Baseline_3090.json` | Minimal GGUF text-to-video. Start here for a baseline. |
| `LTX-2.3_GGUF_T2V_Distilled_3090.json` | **The fast path**: distilled LoRA, 8 steps, cfg 1.0, SageAttention. |
| `LTX-2.3_GGUF_T2V_STG_APG_3090.json` | STG + APG guidance. |
| `LTX-2.3_GGUF_LongClip_LowVRAM_3090.json` | 1280×704 / 193 frames, spatio-temporal decode, CPU accumulator. |
| `LTX-2.3_GGUF_T2V_2Pass_3090.json` | T2V at half res, ×2 latent upscale, partial re-denoise. |
| `LTX-2.3_GGUF_I2V_3090.json` | Image → video with audio. |
| `LTX-2.3_GGUF_I2V_Distilled_3090.json` | **The fast path**: distilled LoRA, 8 steps, cfg 1.0, SageAttention. |
| `LTX-2.3_GGUF_I2V_Baseline_3090.json` | Undistilled reference, 20 steps, cfg 3.0. |
| `LTX-2.3_GGUF_I2V_STG_APG_3090.json` | STG + APG on top of image conditioning. |
| `LTX-2.3_GGUF_I2V_2Pass_3090.json` | 2-pass image→video upscaler workflow. |
| `LTX-2.3_GGUF_T2A_3090.json` | Text → audio only, no video decode. |

Regenerate them after renaming a model:

```
python example_workflows/3090-gguf/generate_workflows.py
```

Edit the filename constants at the top of that script rather than hand-editing the JSON.

## Two things that are easy to get wrong

**LTX-2.3 needs _two_ text-encoder files.** A Gemma-3 12B LLM *and* a separate text projection, loaded together by `DualCLIPLoaderGGUF` with `type = ltxv`. ComfyUI only builds the LTX-AV encoder (`ltxav_te` + `LTXAVGemmaTokenizer`) when two state dicts arrive together — a single-file `CLIPLoader` silently builds a different encoder. The dual loader lists `.gguf` and `.safetensors` in the same dropdown, so a quantized Gemma and a bf16 projection mix freely. **Slot order does not matter**; comfy identifies each file by content.

**The audio VAE loads through plain `VAELoader`.** `LTXVAudioVAELoader` reads `models/checkpoints`; if you run GGUF that folder is usually empty, so its dropdown is empty too. Core `VAELoader` reads `models/vae` and detects an LTX audio VAE from its keys, building the same object. There are two `VAELoader` nodes in these graphs — one video, one audio.

Both text-encoder files must sit in `models/text_encoders`.

## Optional extras, built in

 Both text-encoder files must sit in `models/text_encoders`.
