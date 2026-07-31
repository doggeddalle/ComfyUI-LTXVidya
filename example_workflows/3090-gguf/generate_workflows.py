"""Generate the RTX 3090 / LTX-2.3 GGUF example workflows.

Run this to regenerate the JSON files next to it, e.g. after renaming a model:

    python example_workflows/3090-gguf/generate_workflows.py

Every node type, socket name, socket type and widget order in SCHEMA below was
read from a live ComfyUI node registry rather than written from memory, and the
generator refuses to emit a graph that violates it. Hand-editing LiteGraph JSON
is error-prone enough that generating it is the safer option.
"""

import json
import pathlib
import uuid

OUT_DIR = pathlib.Path(__file__).resolve().parent

# --- model files -----------------------------------------------------------
# Repoint these to match your own models/ folders.
GGUF_MODEL = "ltx-2.3-22b-dev-Q4_0.gguf"
CLIP_GEMMA = "gemma-3-12b-it-heretic-x.IQ4_XS.gguf"
CLIP_PROJECTION = "ltx-2.3_text_projection_bf16.safetensors"
VIDEO_VAE = "LTX23_video_vae_bf16.safetensors"
AUDIO_VAE = "LTX23_audio_vae_bf16.safetensors"
UPSCALE_MODEL = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

# Node modes understood by the ComfyUI frontend.
ALWAYS, MUTED, BYPASSED = 0, 2, 4

WILDCARD = "*"

# name -> (link inputs [(name, type)], outputs [(name, type)], widget names in order)
SCHEMA = {
    "UnetLoaderGGUF": ([], [("MODEL", "MODEL")], ["unet_name"]),
    "DualCLIPLoaderGGUF": (
        [],
        [("CLIP", "CLIP")],
        ["clip_name1", "clip_name2", "type"],
    ),
    "VAELoader": ([], [("VAE", "VAE")], ["vae_name"]),
    "CLIPTextEncode": (
        [("clip", "CLIP")],
        [("CONDITIONING", "CONDITIONING")],
        ["text"],
    ),
    "LTXVConditioning": (
        [("positive", "CONDITIONING"), ("negative", "CONDITIONING")],
        [("positive", "CONDITIONING"), ("negative", "CONDITIONING")],
        ["frame_rate"],
    ),
    "EmptyLTXVLatentVideo": (
        [],
        [("LATENT", "LATENT")],
        ["width", "height", "length", "batch_size"],
    ),
    "LTXVEmptyLatentAudio": (
        [("audio_vae", "VAE")],
        [("LATENT", "LATENT")],
        ["frames_number", "frame_rate", "batch_size"],
    ),
    "LTXVConcatAVLatent": (
        [("video_latent", "LATENT"), ("audio_latent", "LATENT")],
        [("latent", "LATENT")],
        [],
    ),
    "LTXVSeparateAVLatent": (
        [("av_latent", "LATENT")],
        [("video_latent", "LATENT"), ("audio_latent", "LATENT")],
        [],
    ),
    "LTXVAudioVAEDecode": (
        [("samples", "LATENT"), ("audio_vae", "VAE")],
        [("Audio", "AUDIO")],
        [],
    ),
    "LTXVAudioOnlyModel": ([("model", "MODEL")], [("model", "MODEL")], []),
    "LTXVAudioOnlyEmptyVideoLatent": ([], [("latent", "LATENT")], []),
    "LTXVScheduler": (
        [("latent", "LATENT")],
        [("SIGMAS", "SIGMAS")],
        ["steps", "max_shift", "base_shift", "stretch", "terminal"],
    ),
    "SplitSigmasDenoise": (
        [("sigmas", "SIGMAS")],
        [("high_sigmas", "SIGMAS"), ("low_sigmas", "SIGMAS")],
        ["denoise"],
    ),
    "KSamplerSelect": ([], [("SAMPLER", "SAMPLER")], ["sampler_name"]),
    "RandomNoise": (
        [],
        [("NOISE", "NOISE")],
        ["noise_seed", "control_after_generate"],
    ),
    "CFGGuider": (
        [
            ("model", "MODEL"),
            ("positive", "CONDITIONING"),
            ("negative", "CONDITIONING"),
        ],
        [("GUIDER", "GUIDER")],
        ["cfg"],
    ),
    "STGGuiderAdvanced": (
        [
            ("model", "MODEL"),
            ("positive", "CONDITIONING"),
            ("negative", "CONDITIONING"),
        ],
        [("GUIDER", "GUIDER")],
        [
            "skip_steps_sigma_threshold",
            "cfg_star_rescale",
            "sigmas",
            "cfg_values",
            "stg_scale_values",
            "stg_rescale_values",
            "stg_layers_indices",
            # NB: `preset` is an STG_ADVANCED_PRESET *link* input, not a widget.
            "apply_apg",
            "apg_cfg_scale",
            "eta",
            "norm_threshold",
        ],
    ),
    "LTXVApplySTG": ([("model", "MODEL")], [("model", "MODEL")], ["block_indices"]),
    "SamplerCustomAdvanced": (
        [
            ("noise", "NOISE"),
            ("guider", "GUIDER"),
            ("sampler", "SAMPLER"),
            ("sigmas", "SIGMAS"),
            ("latent_image", "LATENT"),
        ],
        [("output", "LATENT"), ("denoised_output", "LATENT")],
        [],
    ),
    "LTXVTiledVAEDecode": (
        [("vae", "VAE"), ("latents", "LATENT")],
        [("image", "IMAGE")],
        [
            "horizontal_tiles",
            "vertical_tiles",
            "overlap",
            "last_frame_fix",
            "working_device",
            "working_dtype",
        ],
    ),
    "LTXVSpatioTemporalTiledVAEDecode": (
        [("vae", "VAE"), ("latents", "LATENT")],
        [("image", "IMAGE")],
        [
            "spatial_tiles",
            "spatial_overlap",
            "temporal_tile_length",
            "temporal_overlap",
            "last_frame_fix",
            "working_device",
            "working_dtype",
        ],
    ),
    "LTXVLatentUpsamplerTiled": (
        [
            ("samples", "LATENT"),
            ("upscale_model", "LATENT_UPSCALE_MODEL"),
            ("vae", "VAE"),
        ],
        [("LATENT", "LATENT")],
        [
            "tile_size",
            "overlap",
            "max_size_for_no_tile",
            "rotate_for_landscape",
            "debug",
        ],
    ),
    "LatentUpscaleModelLoader": (
        [],
        [("LATENT_UPSCALE_MODEL", "LATENT_UPSCALE_MODEL")],
        ["model_name"],
    ),
    "LoadImage": ([], [("IMAGE", "IMAGE"), ("MASK", "MASK")], ["image", "upload"]),
    "ImageScale": (
        [("image", "IMAGE")],
        [("IMAGE", "IMAGE")],
        ["upscale_method", "width", "height", "crop"],
    ),
    "LTXVPreprocess": (
        [("image", "IMAGE")],
        [("output_image", "IMAGE")],
        ["img_compression"],
    ),
    "LTXVImgToVideoConditionOnly": (
        [("vae", "VAE"), ("image", "IMAGE"), ("latent", "LATENT")],
        [("latent", "LATENT")],
        ["strength", "bypass"],
    ),
    "TwoWaySwitch": (
        [("input_1", WILDCARD), ("input_2", WILDCARD)],
        [("output", WILDCARD)],
        ["selection_setting"],
    ),
    "SetLatentNoiseMask": (
        [("samples", "LATENT"), ("mask", "MASK")],
        [("LATENT", "LATENT")],
        [],
    ),
    "SolidMask": ([], [("MASK", "MASK")], ["value", "width", "height"]),
    "CreateVideo": (
        [("images", "IMAGE"), ("audio", "AUDIO")],
        [("VIDEO", "VIDEO")],
        ["fps", "bit_depth"],
    ),
    "SaveVideo": ([("video", "VIDEO")], [], ["filename_prefix", "format", "codec"]),
    "SaveAudio": ([("audio", "AUDIO")], [], ["filename_prefix"]),
    # Frontend-only note node: it has no backend class, so it is skipped by the
    # registry validator.
    "MarkdownNote": ([], [], ["text"]),
}

# Inputs that may legitimately be left unconnected.
OPTIONAL_INPUTS = {
    ("CreateVideo", "audio"),
    ("TwoWaySwitch", "input_1"),
    ("TwoWaySwitch", "input_2"),
}


class Graph:
    def __init__(self):
        self.nodes = []
        self.links = []
        self._nid = 0
        self._lid = 0

    def add(
        self,
        type_name,
        widgets=None,
        pos=(0, 0),
        size=(300, 110),
        title=None,
        mode=ALWAYS,
    ):
        assert type_name in SCHEMA, f"unknown node type {type_name}"
        ins, outs, widget_names = SCHEMA[type_name]
        widgets = list(widgets or [])
        assert len(widgets) == len(widget_names), (
            f"{type_name}: {len(widgets)} widget values for "
            f"{len(widget_names)} widgets {widget_names}"
        )
        self._nid += 1
        node = {
            "id": self._nid,
            "type": type_name,
            "pos": [float(pos[0]), float(pos[1])],
            "size": [float(size[0]), float(size[1])],
            "flags": {},
            "order": len(self.nodes),
            "mode": mode,
            "inputs": [{"name": n, "type": t, "link": None} for n, t in ins],
            "outputs": [{"name": n, "type": t, "links": []} for n, t in outs],
            "properties": {"Node name for S&R": type_name},
            "widgets_values": widgets,
        }
        if title:
            node["title"] = title
        self.nodes.append(node)
        return node

    def link(self, src, src_slot, dst, dst_name):
        _, outs, _ = SCHEMA[src["type"]]
        ins, _, _ = SCHEMA[dst["type"]]
        dst_slot = next(i for i, (n, _) in enumerate(ins) if n == dst_name)
        out_type = outs[src_slot][1]
        in_type = ins[dst_slot][1]
        # A wildcard endpoint adopts the concrete type of the other side, which
        # is how the frontend records links through pass-through switch nodes.
        if out_type == WILDCARD:
            out_type = in_type
        if in_type == WILDCARD:
            in_type = out_type
        assert out_type == in_type, (
            f"type mismatch {src['type']}[{src_slot}]:{out_type} -> "
            f"{dst['type']}.{dst_name}:{in_type}"
        )
        self._lid += 1
        src["outputs"][src_slot]["links"].append(self._lid)
        dst["inputs"][dst_slot]["link"] = self._lid
        dst["inputs"][dst_slot]["type"] = in_type
        self.links.append(
            [self._lid, src["id"], src_slot, dst["id"], dst_slot, out_type]
        )

    def to_json(self):
        return {
            "id": str(uuid.uuid4()),
            "revision": 0,
            "last_node_id": self._nid,
            "last_link_id": self._lid,
            "nodes": self.nodes,
            "links": self.links,
            "groups": [],
            "config": {},
            "extra": {"ds": {"scale": 0.65, "offset": [0, 0]}},
            "version": 0.4,
        }


NEGATIVE = "worst quality, inconsistent motion, blurry, jittery, distorted"
POSITIVE = (
    "A slow cinematic dolly shot through a rain-soaked neon alley at night, "
    "reflections shimmering on wet asphalt, volumetric haze, shallow depth of "
    "field, photorealistic, 35mm film grain"
)
POSITIVE_AUDIO = (
    "Steady rain on wet pavement, distant city traffic, a low electrical hum "
    "from a neon sign, occasional far-off thunder"
)

STG_WIDGETS = [
    0.998,
    True,
    "1.0, 0.9933, 0.9850, 0.9767, 0.9008, 0.6180",
    "8, 6, 6, 4, 3, 1",
    "4, 4, 3, 2, 1, 0",
    "1, 1, 1, 1, 1, 1",
    "[29], [29], [29], [29], [29], [29]",
    True,
    1.0,
    1.0,
    0.0,
]

EXTRAS_NOTE = """## Optional extras

**STG** — `LTX Apply STG` in the model chain is **bypassed** (purple). Ctrl+B on it
to enable, then move the `GUIDER` link from `CFGGuider` to the muted
`STGGuiderAdvanced` (Ctrl+M to unmute it first). `apply_apg` is already on there.

**Lower VRAM** — on the decode node:
- set `working_device` to `cpu` to keep the full-length output accumulator out of VRAM
- set `working_dtype` to `bfloat16` (half the accumulator of float32)
- raise the tile counts

For long clips, use the muted `LTXVSpatioTemporalTiledVAEDecode` instead: it chunks
over *time* as well as space, which the spatial-only node cannot do — that one still
allocates the whole output buffer no matter how many spatial tiles you give it.
"""

MODELS_NOTE = f"""## Models this expects

| Slot | File |
|---|---|
| `UnetLoaderGGUF` | `{GGUF_MODEL}` |
| `DualCLIPLoaderGGUF` clip 1 | `{CLIP_GEMMA}` |
| `DualCLIPLoaderGGUF` clip 2 | `{CLIP_PROJECTION}` |
| video `VAELoader` | `{VIDEO_VAE}` |
| audio `VAELoader` | `{AUDIO_VAE}` |

LTX-2.3 needs **two** text-encoder files — the Gemma-3 LLM and a separate text
projection — which ComfyUI only wires up when two are passed together with
`type = ltxv`. Slot order does not matter; comfy identifies each by its contents.

The audio VAE is loaded with the plain **`VAELoader`** (from `models/vae`) rather
than `LTXVAudioVAELoader`, which reads `models/checkpoints`. Core `VAELoader`
detects an LTX audio VAE from its keys and builds the right class.
"""


def loaders(g, *, model_file=GGUF_MODEL, x=0):
    unet = g.add("UnetLoaderGGUF", [model_file], pos=(x, 0), size=(420, 60))
    clip = g.add(
        "DualCLIPLoaderGGUF",
        [CLIP_GEMMA, CLIP_PROJECTION, "ltxv"],
        pos=(x, 100),
        size=(420, 110),
        title="Dual CLIP (Gemma 3 + LTX text projection)",
    )
    vvae = g.add(
        "VAELoader", [VIDEO_VAE], pos=(x, 250), size=(420, 60), title="Video VAE"
    )
    avae = g.add(
        "VAELoader", [AUDIO_VAE], pos=(x, 340), size=(420, 60), title="Audio VAE"
    )
    return unet, clip, vvae, avae


def conditioning(g, clip, *, positive=POSITIVE, frame_rate=24.0, x=460):
    pos_txt = g.add("CLIPTextEncode", [positive], pos=(x, 0), size=(420, 220))
    neg_txt = g.add("CLIPTextEncode", [NEGATIVE], pos=(x, 250), size=(420, 150))
    g.link(clip, 0, pos_txt, "clip")
    g.link(clip, 0, neg_txt, "clip")
    cond = g.add("LTXVConditioning", [frame_rate], pos=(x, 430), size=(300, 100))
    g.link(pos_txt, 0, cond, "positive")
    g.link(neg_txt, 0, cond, "negative")
    return cond


def model_chain(g, unet, *, x=920):
    """UnetLoader -> bypassed STG -> (model). Bypass passes MODEL straight through."""
    stg = g.add(
        "LTXVApplySTG",
        ["14, 19"],
        pos=(x, 0),
        size=(300, 80),
        title="LTX Apply STG (bypassed - Ctrl+B to enable)",
        mode=BYPASSED,
    )
    g.link(unet, 0, stg, "model")
    return stg


def sampling_inputs(g, *, steps=20, seed=42, x=920, y=120):
    sampler = g.add("KSamplerSelect", ["euler"], pos=(x, y), size=(300, 60))
    noise = g.add("RandomNoise", [seed, "fixed"], pos=(x, y + 90), size=(300, 90))
    return sampler, noise


def scheduler(g, latent, *, steps=20, x=920, y=330):
    sched = g.add(
        "LTXVScheduler",
        [steps, 2.05, 0.95, True, 0.1],
        pos=(x, y),
        size=(300, 160),
    )
    g.link(latent, 0, sched, "latent")
    return sched


def guiders(g, model_node, cond, *, cfg=3.0, x=1260):
    """Active CFGGuider plus a muted STGGuiderAdvanced alternative."""
    guider = g.add("CFGGuider", [cfg], pos=(x, 0), size=(300, 120))
    g.link(model_node, 0, guider, "model")
    g.link(cond, 0, guider, "positive")
    g.link(cond, 1, guider, "negative")

    stg_guider = g.add(
        "STGGuiderAdvanced",
        STG_WIDGETS,
        pos=(x, 160),
        size=(420, 400),
        title="STG + APG guider (muted - Ctrl+M, then move the GUIDER link)",
        mode=MUTED,
    )
    g.link(model_node, 0, stg_guider, "model")
    g.link(cond, 0, stg_guider, "positive")
    g.link(cond, 1, stg_guider, "negative")
    return guider


def sample(g, noise, guider, sampler, sigmas, sigma_slot, latent, *, x=1700, y=0):
    smp = g.add("SamplerCustomAdvanced", [], pos=(x, y), size=(300, 130))
    g.link(noise, 0, smp, "noise")
    g.link(guider, 0, smp, "guider")
    g.link(sampler, 0, smp, "sampler")
    g.link(sigmas, sigma_slot, smp, "sigmas")
    g.link(latent, 0, smp, "latent_image")
    return smp


def av_outputs(g, smp, vvae, avae, *, fps=24.0, prefix="LTXV23/out", x=2060):
    """Separate the AV latent, decode both halves, mux into a video."""
    sep = g.add("LTXVSeparateAVLatent", [], pos=(x, 0), size=(300, 80))
    g.link(smp, 0, sep, "av_latent")

    dec = g.add(
        "LTXVTiledVAEDecode",
        [2, 2, 6, False, "auto", "bfloat16"],
        pos=(x, 130),
        size=(340, 200),
    )
    g.link(vvae, 0, dec, "vae")
    g.link(sep, 0, dec, "latents")

    st_dec = g.add(
        "LTXVSpatioTemporalTiledVAEDecode",
        [2, 1, 16, 1, False, "cpu", "bfloat16"],
        pos=(x, 360),
        size=(360, 230),
        title="Spatio-temporal decode (muted - use for long clips)",
        mode=MUTED,
    )
    g.link(vvae, 0, st_dec, "vae")
    g.link(sep, 0, st_dec, "latents")

    adec = g.add("LTXVAudioVAEDecode", [], pos=(x, 620), size=(300, 80))
    g.link(sep, 1, adec, "samples")
    g.link(avae, 0, adec, "audio_vae")

    video = g.add("CreateVideo", [fps, 8], pos=(x + 400, 0), size=(300, 110))
    g.link(dec, 0, video, "images")
    g.link(adec, 0, video, "audio")

    save = g.add(
        "SaveVideo", [prefix, "auto", "auto"], pos=(x + 740, 0), size=(380, 140)
    )
    g.link(video, 0, save, "video")
    return sep


def note(g, text, pos, size=(560, 380), title="Read me"):
    g.add("MarkdownNote", [text], pos=pos, size=size, title=title)


# ---------------------------------------------------------------------------
# 1. Text to video (+ audio)
# ---------------------------------------------------------------------------
def build_t2v():
    g = Graph()
    unet, clip, vvae, avae = loaders(g)
    cond = conditioning(g, clip)
    model_node = model_chain(g, unet)
    sampler, noise = sampling_inputs(g)

    vlat = g.add(
        "EmptyLTXVLatentVideo", [960, 544, 121, 1], pos=(920, 540), size=(300, 130)
    )
    alat = g.add("LTXVEmptyLatentAudio", [121, 24, 1], pos=(920, 700), size=(300, 130))
    g.link(avae, 0, alat, "audio_vae")

    av = g.add("LTXVConcatAVLatent", [], pos=(1260, 620), size=(300, 80))
    g.link(vlat, 0, av, "video_latent")
    g.link(alat, 0, av, "audio_latent")

    sched = scheduler(g, av)
    guider = guiders(g, model_node, cond)
    smp = sample(g, noise, guider, sampler, sched, 0, av)
    av_outputs(g, smp, vvae, avae, prefix="LTXV23/t2v")

    note(g, MODELS_NOTE + "\n" + EXTRAS_NOTE, (0, 460), size=(560, 620))
    return g


# ---------------------------------------------------------------------------
# 2. Image to video (+ audio)
# ---------------------------------------------------------------------------
def image_chain(g, vvae, vlat, *, width=960, height=544, x=920, y=880, load=None):
    """Scale + preprocess an image and use it to condition `vlat`.

    Pass an existing `load` node to reuse one LoadImage across both passes of a
    two-pass workflow -- each pass needs the image at its own resolution, but
    asking the user to pick the same file twice is a trap.
    """
    if load is None:
        load = g.add("LoadImage", ["example.png", "image"], pos=(x, y), size=(320, 320))
    scale = g.add(
        "ImageScale",
        ["lanczos", width, height, "center"],
        pos=(x + 340, y),
        size=(300, 130),
    )
    g.link(load, 0, scale, "image")
    pre = g.add("LTXVPreprocess", [35], pos=(x + 340, y + 160), size=(300, 80))
    g.link(scale, 0, pre, "image")

    cond_latent = g.add(
        "LTXVImgToVideoConditionOnly",
        [1.0, False],
        pos=(x + 340, y + 270),
        size=(320, 120),
    )
    g.link(vvae, 0, cond_latent, "vae")
    g.link(pre, 0, cond_latent, "image")
    g.link(vlat, 0, cond_latent, "latent")
    return cond_latent, load


def build_i2v():
    g = Graph()
    unet, clip, vvae, avae = loaders(g)
    cond = conditioning(g, clip)
    model_node = model_chain(g, unet)
    sampler, noise = sampling_inputs(g)

    vlat = g.add(
        "EmptyLTXVLatentVideo", [960, 544, 121, 1], pos=(920, 540), size=(300, 130)
    )
    img_latent, _ = image_chain(g, vvae, vlat)

    alat = g.add("LTXVEmptyLatentAudio", [121, 24, 1], pos=(920, 700), size=(300, 130))
    g.link(avae, 0, alat, "audio_vae")

    av = g.add("LTXVConcatAVLatent", [], pos=(1620, 620), size=(300, 80))
    g.link(img_latent, 0, av, "video_latent")
    g.link(alat, 0, av, "audio_latent")

    sched = scheduler(g, av)
    guider = guiders(g, model_node, cond)
    smp = sample(g, noise, guider, sampler, sched, 0, av)
    av_outputs(g, smp, vvae, avae, prefix="LTXV23/i2v")

    note(
        g,
        MODELS_NOTE + "\n## Image conditioning\n\n"
        "`ImageScale` must match `EmptyLTXVLatentVideo` (960x544 here). Set\n"
        "`LTXVImgToVideoConditionOnly.strength` below 1.0 to loosen the first\n"
        "frame, or tick `bypass` to fall back to text-only.\n\n" + EXTRAS_NOTE,
        (0, 460),
        size=(560, 680),
    )
    return g


# ---------------------------------------------------------------------------
# 3. Text to audio only
# ---------------------------------------------------------------------------
def build_t2a():
    g = Graph()
    unet, clip, vvae, avae = loaders(g)
    cond = conditioning(g, clip, positive=POSITIVE_AUDIO, frame_rate=24.0)

    audio_model = g.add(
        "LTXVAudioOnlyModel", [], pos=(920, 0), size=(300, 60), title="Audio-only model"
    )
    g.link(unet, 0, audio_model, "model")

    sampler, noise = sampling_inputs(g)

    vlat = g.add(
        "LTXVAudioOnlyEmptyVideoLatent",
        [],
        pos=(920, 540),
        size=(320, 60),
        title="Placeholder video latent",
    )
    alat = g.add("LTXVEmptyLatentAudio", [121, 24, 1], pos=(920, 640), size=(300, 130))
    g.link(avae, 0, alat, "audio_vae")

    av = g.add("LTXVConcatAVLatent", [], pos=(1260, 580), size=(300, 80))
    g.link(vlat, 0, av, "video_latent")
    g.link(alat, 0, av, "audio_latent")

    sched = scheduler(g, av)
    guider = guiders(g, audio_model, cond)
    smp = sample(g, noise, guider, sampler, sched, 0, av)

    sep = g.add("LTXVSeparateAVLatent", [], pos=(2060, 0), size=(300, 80))
    g.link(smp, 0, sep, "av_latent")
    adec = g.add("LTXVAudioVAEDecode", [], pos=(2060, 130), size=(300, 80))
    g.link(sep, 1, adec, "samples")
    g.link(avae, 0, adec, "audio_vae")
    save = g.add("SaveAudio", ["LTXV23/t2a"], pos=(2060, 250), size=(320, 110))
    g.link(adec, 0, save, "audio")

    note(
        g,
        "## LTX-2.3 GGUF - text to audio\n\n"
        "`LTXVAudioOnlyModel` wraps the DiT so only the audio branch is sampled,\n"
        "and `LTXVAudioOnlyEmptyVideoLatent` supplies the placeholder video half\n"
        "that the AV transformer still expects. There is no video decode here.\n\n"
        "Write the *sound* you want in the positive prompt, not the picture.\n\n"
        + MODELS_NOTE,
        (0, 460),
        size=(560, 620),
    )
    return g


# ---------------------------------------------------------------------------
# 4/5. Two-pass upscale
# ---------------------------------------------------------------------------
def build_two_pass(*, with_image):
    g = Graph()
    unet, clip, vvae, avae = loaders(g)
    cond = conditioning(g, clip)
    model_node = model_chain(g, unet)
    sampler, noise = sampling_inputs(g)

    # --- pass 1, half resolution ---
    vlat = g.add(
        "EmptyLTXVLatentVideo",
        [640, 352, 121, 1],
        pos=(920, 540),
        size=(300, 130),
        title="Pass 1 latent (half res)",
    )
    first_video = vlat
    load_image = None
    if with_image:
        first_video, load_image = image_chain(g, vvae, vlat, width=640, height=352)

    alat = g.add("LTXVEmptyLatentAudio", [121, 24, 1], pos=(920, 700), size=(300, 130))
    g.link(avae, 0, alat, "audio_vae")

    av1 = g.add(
        "LTXVConcatAVLatent",
        [],
        pos=(1620, 620),
        size=(300, 80),
        title="Pass 1 AV latent",
    )
    g.link(first_video, 0, av1, "video_latent")
    g.link(alat, 0, av1, "audio_latent")

    sched = scheduler(g, av1)
    guider = guiders(g, model_node, cond)
    smp1 = sample(g, noise, guider, sampler, sched, 0, av1, x=1700)

    sep1 = g.add(
        "LTXVSeparateAVLatent", [], pos=(2060, 0), size=(300, 80), title="Pass 1 result"
    )
    g.link(smp1, 0, sep1, "av_latent")

    # --- upscale the video half ---
    up_model = g.add(
        "LatentUpscaleModelLoader", [UPSCALE_MODEL], pos=(2060, 130), size=(380, 60)
    )
    ups = g.add(
        "LTXVLatentUpsamplerTiled",
        [24, 8, 32, False, False],
        pos=(2060, 220),
        size=(340, 190),
        title="Spatial upscale x2 (tiled)",
    )
    g.link(sep1, 0, ups, "samples")
    g.link(up_model, 0, ups, "upscale_model")
    g.link(vvae, 0, ups, "vae")

    second_video = ups
    if with_image:
        # Same LoadImage, re-scaled to the upscaled resolution.
        second_video, _ = image_chain(
            g, vvae, ups, width=1280, height=704, x=2060, y=700, load=load_image
        )

    # --- first-pass-audio toggle ---
    mask = g.add(
        "SolidMask",
        [0.0, 1024, 1024],
        pos=(2460, 220),
        size=(300, 130),
        title="Zero mask = freeze audio",
    )
    frozen = g.add(
        "SetLatentNoiseMask",
        [],
        pos=(2460, 380),
        size=(300, 80),
        title="Freeze pass 1 audio",
    )
    g.link(sep1, 1, frozen, "samples")
    g.link(mask, 0, frozen, "mask")

    switch = g.add(
        "TwoWaySwitch",
        [1],
        pos=(2460, 490),
        size=(340, 130),
        title="Force First Pass Audio (1) or Resample Audio (2)",
    )
    g.link(frozen, 0, switch, "input_1")
    g.link(sep1, 1, switch, "input_2")

    av2 = g.add(
        "LTXVConcatAVLatent",
        [],
        pos=(2840, 380),
        size=(300, 80),
        title="Pass 2 AV latent",
    )
    g.link(second_video, 0, av2, "video_latent")
    g.link(switch, 0, av2, "audio_latent")

    # --- pass 2, partial denoise on the upscaled latent ---
    sched2 = g.add(
        "LTXVScheduler",
        [20, 2.05, 0.95, True, 0.1],
        pos=(2840, 0),
        size=(300, 160),
        title="Pass 2 schedule",
    )
    g.link(av2, 0, sched2, "latent")
    split = g.add(
        "SplitSigmasDenoise",
        [0.4],
        pos=(2840, 190),
        size=(300, 90),
        title="Pass 2 denoise strength",
    )
    g.link(sched2, 0, split, "sigmas")

    noise2 = g.add("RandomNoise", [43, "fixed"], pos=(2840, 500), size=(300, 90))
    smp2 = sample(g, noise2, guider, sampler, split, 1, av2, x=3200)

    av_outputs(
        g,
        smp2,
        vvae,
        avae,
        prefix="LTXV23/i2v_2pass" if with_image else "LTXV23/t2v_2pass",
        x=3560,
    )

    note(
        g,
        "## LTX-2.3 GGUF - two-pass upscale\n\n"
        "Pass 1 samples at half resolution, the video latent is upscaled x2 with\n"
        "`LTXVLatentUpsamplerTiled`, and pass 2 re-denoises only partway\n"
        "(`SplitSigmasDenoise` -> `low_sigmas`) to add detail without redrawing.\n\n"
        "### The audio toggle\n\n"
        "`Force First Pass Audio (1) or Resample Audio (2)`:\n\n"
        "- **1** - pass 1's audio latent goes through a zero `SolidMask`, so pass 2\n"
        "  denoises none of it and the audio you heard in pass 1 survives intact.\n"
        "- **2** - the raw latent is passed instead, letting the partial denoise\n"
        "  refine the audio alongside the video.\n\n"
        "Both options reuse pass 1's audio on purpose. Starting pass 2 from an\n"
        "*empty* audio latent would not work: it begins from a mid-schedule sigma,\n"
        "so there is no path from silence to a sensible result.\n\n"
        "**Requires the `TwoWaySwitch` node**, unlike the other four workflows.\n\n"
        "### VRAM\n\n"
        "`LTXVLatentUpsamplerTiled.tile_size` is the knob to lower first if pass 2\n"
        "will not fit; the upsampler runs while the model is still resident.\n\n"
        + MODELS_NOTE,
        (0, 460),
        size=(560, 900),
    )
    return g


# ---------------------------------------------------------------------------
def validate(doc, name):
    nodes = {n["id"]: n for n in doc["nodes"]}
    problems = []
    seen = set()

    for link in doc["links"]:
        lid, src, sslot, dst, dslot, ltype = link
        if lid in seen:
            problems.append(f"duplicate link id {lid}")
        seen.add(lid)
        if src not in nodes or dst not in nodes:
            problems.append(f"link {lid} references a missing node")
            continue
        s, d = nodes[src], nodes[dst]
        if sslot >= len(s["outputs"]):
            problems.append(f"link {lid}: {s['type']} has no output {sslot}")
            continue
        if dslot >= len(d["inputs"]):
            problems.append(f"link {lid}: {d['type']} has no input {dslot}")
            continue
        for endpoint, decl in (
            (s["outputs"][sslot]["type"], "output"),
            (d["inputs"][dslot]["type"], "input"),
        ):
            if endpoint not in (ltype, WILDCARD):
                problems.append(
                    f"link {lid}: {ltype} mismatches {s['type']}->{d['type']} {decl}"
                )
        if lid not in s["outputs"][sslot]["links"]:
            problems.append(f"link {lid} missing from {s['type']} output")
        if d["inputs"][dslot]["link"] != lid:
            problems.append(f"link {lid} not recorded on {d['type']} input")

    for node in doc["nodes"]:
        _, _, widget_names = SCHEMA[node["type"]]
        if len(node["widgets_values"]) != len(widget_names):
            problems.append(f"{node['type']}: widget count mismatch")
        for inp in node["inputs"]:
            if inp["link"] is None:
                if (node["type"], inp["name"]) not in OPTIONAL_INPUTS:
                    problems.append(f"{node['type']}.{inp['name']} unconnected")
            elif inp["link"] not in seen:
                problems.append(f"{node['type']}.{inp['name']} dangling link")

    # LTXV shape rules: frames must be 8n+1, dimensions multiples of 32.
    for node in doc["nodes"]:
        if node["type"] == "EmptyLTXVLatentVideo":
            width, height, length, _ = node["widgets_values"]
            if width % 32 or height % 32:
                problems.append(f"{width}x{height} is not a multiple of 32")
            if (length - 1) % 8:
                problems.append(f"length {length} is not 8n+1")

    if problems:
        raise SystemExit(f"[{name}] VALIDATION FAILED:\n  " + "\n  ".join(problems))
    return len(doc["nodes"]), len(doc["links"])


WORKFLOWS = [
    ("LTX-2.3_GGUF_T2V_3090.json", build_t2v),
    ("LTX-2.3_GGUF_I2V_3090.json", build_i2v),
    ("LTX-2.3_GGUF_T2A_3090.json", build_t2a),
    ("LTX-2.3_GGUF_T2V_2Pass_3090.json", lambda: build_two_pass(with_image=False)),
    ("LTX-2.3_GGUF_I2V_2Pass_3090.json", lambda: build_two_pass(with_image=True)),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, builder in WORKFLOWS:
        doc = builder().to_json()
        n_nodes, n_links = validate(doc, filename)
        (OUT_DIR / filename).write_text(
            json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8"
        )
        print(f"OK  {filename}: {n_nodes} nodes, {n_links} links")


if __name__ == "__main__":
    main()
