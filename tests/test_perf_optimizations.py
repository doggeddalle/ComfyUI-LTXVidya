"""Tests for the RTX 3090 / LTX-2.3 GGUF performance work.

Each test pins a behaviour that an optimization must not change. ComfyUI is
stubbed by tests/conftest.py.
"""

import ast
import importlib.util
import pathlib
import sys
import types

import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]

_pkg = sys.modules.get("ltxpkg")
if _pkg is None:
    _pkg = types.ModuleType("ltxpkg")
    _pkg.__path__ = [str(REPO)]
    sys.modules["ltxpkg"] = _pkg


def _load(dotted, relpath):
    if dotted in sys.modules:
        return sys.modules[dotted]
    spec = importlib.util.spec_from_file_location(dotted, REPO / relpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = module
    spec.loader.exec_module(module)
    return module


tiled_vae_decode = _load("ltxpkg.tiled_vae_decode", "tiled_vae_decode.py")


# ===========================================================================
# Tiled VAE decode -- memory layout changed, results must not
# ===========================================================================
TIME_SCALE, WIDTH_SCALE, HEIGHT_SCALE = 8, 32, 32


class _DeterministicVAE:
    """A stand-in VAE whose decode is an exact nearest-neighbour upsample.

    Because the decode of a latent crop equals the corresponding crop of the
    decode of the whole latent, a correct tiled decode must reproduce the
    untiled decode exactly. That makes the blend weights a partition of unity
    and turns any broadcasting mistake into a visible numeric difference.
    """

    downscale_index_formula = (TIME_SCALE, WIDTH_SCALE, HEIGHT_SCALE)

    def decode(self, samples):
        # samples: [B, C, F, H, W] -> image [B*frames, H*hs, W*ws, 3]
        plane = samples[:, 0]  # [B, F, H, W]
        first, rest = plane[:, :1], plane[:, 1:]
        if rest.shape[1]:
            rest = rest.repeat_interleave(TIME_SCALE, dim=1)
        frames = torch.cat([first, rest], dim=1)
        frames = frames.repeat_interleave(HEIGHT_SCALE, dim=-2)
        frames = frames.repeat_interleave(WIDTH_SCALE, dim=-1)
        image = frames.unsqueeze(-1).expand(*frames.shape, 3)
        b, f = image.shape[0], image.shape[1]
        return image.reshape(b * f, image.shape[2], image.shape[3], 3).contiguous()


def _positional_latent(frames=3, height=8, width=8, channels=4):
    """Latent whose values encode absolute position, so crops stay identifiable."""
    f = torch.arange(frames).view(1, 1, frames, 1, 1) * 10000.0
    y = torch.arange(height).view(1, 1, 1, height, 1) * 100.0
    x = torch.arange(width).view(1, 1, 1, 1, width) * 1.0
    base = (f + y + x).expand(1, channels, frames, height, width).clone()
    return base.to(torch.float32)


def test_tiled_decode_matches_untiled_decode():
    """2x2 tiling with overlap must reconstruct the untiled result exactly."""
    node = tiled_vae_decode.LTXVTiledVAEDecode()
    vae = _DeterministicVAE()
    latents = {"samples": _positional_latent()}

    untiled = node.decode(vae, latents, 1, 1, 1, False, "auto", "float32")[0]
    tiled = node.decode(vae, latents, 2, 2, 2, False, "auto", "float32")[0]

    assert tiled.shape == untiled.shape
    assert torch.allclose(tiled, untiled, atol=1e-4), (
        f"tiled decode diverged from untiled by "
        f"{(tiled - untiled).abs().max().item()}"
    )


@pytest.mark.parametrize(
    "h_tiles,v_tiles,overlap", [(2, 1, 1), (1, 3, 2), (3, 2, 1), (2, 2, 4)]
)
def test_tiled_decode_partition_of_unity(h_tiles, v_tiles, overlap):
    """Blend weights must sum to 1 everywhere for any tiling configuration."""
    node = tiled_vae_decode.LTXVTiledVAEDecode()
    vae = _DeterministicVAE()
    latents = {"samples": _positional_latent(height=12, width=12)}

    untiled = node.decode(vae, latents, 1, 1, 1, False, "auto", "float32")[0]
    tiled = node.decode(
        vae, latents, h_tiles, v_tiles, overlap, False, "auto", "float32"
    )[0]

    assert torch.allclose(tiled, untiled, atol=1e-4)


def test_weight_accumulator_is_frame_and_batch_independent():
    """The blend mask varies only over (H, W); allocating it per frame is waste.

    Guards the memory optimization itself: a regression to a full-size weights
    buffer would silently restore hundreds of MB of VRAM use on long clips.
    """
    source = (REPO / "tiled_vae_decode.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    decode = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "decode"
    )
    zeros_shapes = [
        call.args[0]
        for call in ast.walk(decode)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in ("zeros", "ones")
        and call.args
        and isinstance(call.args[0], ast.Tuple)
    ]
    # The output accumulator legitimately spans batch+frames; every weight
    # buffer must lead with literal 1, 1.
    weight_shapes = [s for s in zeros_shapes if len(s.elts) == 5 and s.elts[-1]]
    leading = [
        [getattr(e, "value", None) for e in shape.elts[:2]] for shape in weight_shapes
    ]
    assert [1, 1] in leading, "expected a (1, 1, H, W, 1) weight buffer"


def test_bfloat16_is_offered_as_working_dtype():
    assert "bfloat16" in tiled_vae_decode._WORKING_DTYPES
    samples = torch.zeros(1, dtype=torch.float16)
    assert (
        tiled_vae_decode._resolve_working_dtype("bfloat16", samples) is torch.bfloat16
    )


def test_working_dtype_auto_follows_latents():
    samples = torch.zeros(1, dtype=torch.bfloat16)
    assert tiled_vae_decode._resolve_working_dtype("auto", samples) is torch.bfloat16
    # An unrecognised value must fall back rather than raise UnboundLocalError,
    # which is what the original if/elif chain did.
    assert (
        tiled_vae_decode._resolve_working_dtype("nonsense", samples) is torch.bfloat16
    )


def test_headroom_check_is_a_noop_on_cpu():
    tiled_vae_decode._check_decode_headroom(
        (1, 1000, 1000, 1000, 3), torch.float32, torch.device("cpu"), hint="x"
    )


def test_headroom_check_raises_actionable_error(monkeypatch):
    """A predictable allocation should refuse cleanly rather than OOM-hang."""
    monkeypatch.setattr(
        sys.modules["comfy.model_management"],
        "get_free_memory",
        lambda device=None: 1 * 2**30,
    )
    with pytest.raises(RuntimeError, match="working_device=cpu"):
        tiled_vae_decode._check_decode_headroom(
            (1, 241, 1088, 1920, 3),
            torch.float16,
            torch.device("cuda"),
            hint="Set working_device=cpu, or lower the resolution.",
        )
