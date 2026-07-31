"""Structural guarantees for the 3090/GGUF example workflows.

These check the shape of the generated graphs -- link consistency, the loaders
that LTX-2.3 actually requires, and that the optional extras stay inert. They
deliberately need neither ComfyUI nor torch, so CI can run them anywhere.

Validation *against a live node registry* (socket names, widget counts, model
filenames) happens in the generator's own workflow; see
example_workflows/3090-gguf/generate_workflows.py.
"""

import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO / "example_workflows" / "3090-gguf"

BYPASSED, MUTED = 4, 2


def _workflows():
    return sorted(WORKFLOW_DIR.glob("*.json"))


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_workflow_set_is_the_expected_five():
    assert {p.name for p in _workflows()} == {
        "LTX-2.3_GGUF_T2V_3090.json",
        "LTX-2.3_GGUF_I2V_3090.json",
        "LTX-2.3_GGUF_T2A_3090.json",
        "LTX-2.3_GGUF_T2V_2Pass_3090.json",
        "LTX-2.3_GGUF_I2V_2Pass_3090.json",
    }


@pytest.mark.parametrize("path", _workflows(), ids=lambda p: p.name)
def test_links_are_consistent(path):
    """Every link must reference real slots and be recorded on both endpoints."""
    doc = _load(path)
    nodes = {n["id"]: n for n in doc["nodes"]}

    for link_id, src, src_slot, dst, dst_slot, _type in doc["links"]:
        assert (
            src in nodes and dst in nodes
        ), f"link {link_id} references a missing node"
        source, target = nodes[src], nodes[dst]
        assert src_slot < len(source["outputs"]), f"{source['type']} output {src_slot}"
        assert dst_slot < len(target["inputs"]), f"{target['type']} input {dst_slot}"
        assert link_id in source["outputs"][src_slot]["links"]
        assert target["inputs"][dst_slot]["link"] == link_id


@pytest.mark.parametrize("path", _workflows(), ids=lambda p: p.name)
def test_uses_the_dual_clip_loader(path):
    """A single CLIPLoader silently builds the wrong encoder for LTX-2.3.

    ComfyUI only assembles the LTX-AV text encoder when two state dicts arrive
    together with type=ltxv.
    """
    doc = _load(path)
    types = [n["type"] for n in doc["nodes"]]

    assert types.count("DualCLIPLoaderGGUF") == 1
    assert "CLIPLoader" not in types

    loader = next(n for n in doc["nodes"] if n["type"] == "DualCLIPLoaderGGUF")
    clip1, clip2, clip_type = loader["widgets_values"]
    assert clip_type == "ltxv"
    # One Gemma LLM and one text projection -- either slot order is valid,
    # because comfy identifies each file by its contents.
    names = {clip1.lower(), clip2.lower()}
    assert any("gemma" in n for n in names), names
    assert any("projection" in n for n in names), names


@pytest.mark.parametrize("path", _workflows(), ids=lambda p: p.name)
def test_has_the_audio_branch(path):
    """The 22B is an AV model; a video-only graph produces silent output."""
    doc = _load(path)
    types = [n["type"] for n in doc["nodes"]]

    for required in (
        "LTXVEmptyLatentAudio",
        "LTXVConcatAVLatent",
        "LTXVSeparateAVLatent",
        "LTXVAudioVAEDecode",
    ):
        assert required in types, f"{path.name} is missing {required}"

    # The audio VAE must come from core VAELoader: LTXVAudioVAELoader reads
    # models/checkpoints, which is empty in a GGUF-only install.
    assert "LTXVAudioVAELoader" not in types
    assert types.count("VAELoader") == 2, "expected one video and one audio VAE"


@pytest.mark.parametrize("path", _workflows(), ids=lambda p: p.name)
def test_latent_shape_rules(path):
    """LTXV wants 8n+1 frames and dimensions that are multiples of 32."""
    doc = _load(path)
    for node in doc["nodes"]:
        if node["type"] != "EmptyLTXVLatentVideo":
            continue
        width, height, length, _batch = node["widgets_values"]
        assert width % 32 == 0 and height % 32 == 0, f"{width}x{height}"
        assert (length - 1) % 8 == 0, length


@pytest.mark.parametrize("path", _workflows(), ids=lambda p: p.name)
def test_optional_extras_are_inert_until_enabled(path):
    """Whatever extras a workflow carries must not run until switched on."""
    doc = _load(path)
    modes = {}
    for node in doc["nodes"]:
        modes.setdefault(node["type"], []).append(node.get("mode", 0))

    expected = {
        "LTXVApplySTG": BYPASSED,
        "STGGuiderAdvanced": MUTED,
        "LTXVSpatioTemporalTiledVAEDecode": MUTED,
    }
    for node_type, mode in expected.items():
        if node_type in modes:
            assert modes[node_type] == [mode], f"{path.name}: {node_type}"


@pytest.mark.parametrize(
    "path",
    [p for p in _workflows() if "T2A" not in p.name],
    ids=lambda p: p.name,
)
def test_video_workflows_offer_stg(path):
    """STG perturbs spatial attention, so it belongs in the video graphs.

    T2A is excluded on purpose: it samples only the audio branch.
    """
    doc = _load(path)
    modes = {n["type"]: n.get("mode", 0) for n in doc["nodes"]}
    assert modes.get("LTXVApplySTG") == BYPASSED
    assert modes.get("STGGuiderAdvanced") == MUTED


def test_t2a_has_no_video_output():
    doc = _load(WORKFLOW_DIR / "LTX-2.3_GGUF_T2A_3090.json")
    types = [n["type"] for n in doc["nodes"]]

    assert "LTXVAudioOnlyModel" in types
    assert "LTXVAudioOnlyEmptyVideoLatent" in types
    assert "SaveAudio" in types
    assert "CreateVideo" not in types
    assert "SaveVideo" not in types


def test_two_pass_reuses_one_load_image():
    """Two LoadImage nodes would mean picking the same file twice."""
    doc = _load(WORKFLOW_DIR / "LTX-2.3_GGUF_I2V_2Pass_3090.json")
    types = [n["type"] for n in doc["nodes"]]

    assert types.count("LoadImage") == 1
    assert types.count("ImageScale") == 2  # one per pass resolution


@pytest.mark.parametrize(
    "name",
    ["LTX-2.3_GGUF_T2V_2Pass_3090.json", "LTX-2.3_GGUF_I2V_2Pass_3090.json"],
)
def test_audio_toggle_is_wired(name):
    """Both switch inputs come from pass 1; input_1 is frozen by a zero mask."""
    doc = _load(WORKFLOW_DIR / name)
    nodes = {n["id"]: n for n in doc["nodes"]}
    types = [n["type"] for n in doc["nodes"]]

    assert types.count("TwoWaySwitch") == 1
    switch = next(n for n in doc["nodes"] if n["type"] == "TwoWaySwitch")
    assert switch["widgets_values"] == [1], "default should force first-pass audio"

    feeders = {}
    for _link_id, src, _ss, dst, dst_slot, _t in doc["links"]:
        if dst == switch["id"]:
            feeders[switch["inputs"][dst_slot]["name"]] = nodes[src]["type"]
    assert feeders.get("input_1") == "SetLatentNoiseMask", feeders
    assert feeders.get("input_2") == "LTXVSeparateAVLatent", feeders

    mask = next(n for n in doc["nodes"] if n["type"] == "SolidMask")
    assert mask["widgets_values"][0] == 0.0, "mask must be zero to freeze the audio"


@pytest.mark.parametrize(
    "name",
    ["LTX-2.3_GGUF_T2V_2Pass_3090.json", "LTX-2.3_GGUF_I2V_2Pass_3090.json"],
)
def test_two_pass_partially_denoises_the_second_pass(name):
    """Pass 2 must run a partial denoise, not a full one from scratch."""
    doc = _load(WORKFLOW_DIR / name)
    nodes = {n["id"]: n for n in doc["nodes"]}
    types = [n["type"] for n in doc["nodes"]]

    assert "LTXVLatentUpsamplerTiled" in types
    split = next(n for n in doc["nodes"] if n["type"] == "SplitSigmasDenoise")
    assert 0.0 < split["widgets_values"][0] < 1.0

    # The second sampler must take low_sigmas (slot 1), not the full schedule.
    samplers = [n for n in doc["nodes"] if n["type"] == "SamplerCustomAdvanced"]
    assert len(samplers) == 2
    sigma_sources = []
    for sampler in samplers:
        slot = next(
            i for i, inp in enumerate(sampler["inputs"]) if inp["name"] == "sigmas"
        )
        link_id = sampler["inputs"][slot]["link"]
        link = next(link for link in doc["links"] if link[0] == link_id)
        sigma_sources.append((nodes[link[1]]["type"], link[2]))
    assert ("SplitSigmasDenoise", 1) in sigma_sources, sigma_sources


def test_generator_and_readme_ship_with_the_workflows():
    assert (WORKFLOW_DIR / "generate_workflows.py").exists()
    assert (WORKFLOW_DIR / "README.md").exists()
