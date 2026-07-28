"""Tests for the RTX 3090 / LTX-2.3 GGUF performance work.

Each test pins a behaviour that an optimization must not change. ComfyUI is
stubbed by tests/conftest.py.
"""

import ast
import copy
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
latent_norm = _load("ltxpkg.latent_norm", "latent_norm.py")


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


# ===========================================================================
# latent_norm -- vectorized; must match the original per-element loops
# ===========================================================================
def _reference_adain(latents, reference, factor, per_frame=False):
    """The original nested-loop implementation, kept verbatim as an oracle."""
    latents_copy = copy.deepcopy(latents)
    reference = copy.deepcopy(reference)
    t = latents_copy["samples"]

    if per_frame:
        if reference["samples"].size(2) == 1:
            reference["samples"] = reference["samples"].repeat(1, 1, t.size(2), 1, 1)
        elif t.size(2) > reference["samples"].size(2):
            raise ValueError("Latents have more frames than reference")

    for i in range(t.size(0)):
        for c in range(t.size(1)):
            if not per_frame:
                r_sd, r_mean = torch.std_mean(reference["samples"][i, c], dim=None)
                i_sd, i_mean = torch.std_mean(t[i, c], dim=None)
                t[i, c] = ((t[i, c] - i_mean) / i_sd) * r_sd + r_mean
            else:
                for f in range(t.size(2)):
                    r_sd, r_mean = torch.std_mean(
                        reference["samples"][i, c, f], dim=None
                    )
                    i_sd, i_mean = torch.std_mean(t[i, c, f], dim=None)
                    t[i, c, f] = ((t[i, c, f] - i_mean) / i_sd) * r_sd + r_mean

    return torch.lerp(latents["samples"], t, factor)


def _reference_statnorm(
    latents, target_mean, target_std, percentile, factor, clip_outliers
):
    """The original nested-loop implementation, kept verbatim as an oracle."""
    latents_copy = copy.deepcopy(latents)
    t = latents_copy["samples"]
    lower_percentile = (100 - percentile) / 2
    upper_percentile = 100 - lower_percentile

    for i in range(t.size(0)):
        for c in range(t.size(1)):
            channel_data = t[i, c]
            original_shape = channel_data.shape
            channel_flat = channel_data.flatten()

            lower_bound = torch.quantile(channel_flat, lower_percentile / 100)
            upper_bound = torch.quantile(channel_flat, upper_percentile / 100)
            mask = (channel_flat >= lower_bound) & (channel_flat <= upper_bound)

            if mask.sum() > 0:
                filtered_data = channel_flat[mask]
                current_mean = filtered_data.mean()
                current_std = filtered_data.std()

                if current_std > 1e-8:
                    normalized_flat = (
                        (channel_flat - current_mean) / current_std
                    ) * target_std + target_mean

                    if clip_outliers:
                        normalized_lower = (
                            (lower_bound - current_mean) / current_std
                        ) * target_std + target_mean
                        normalized_upper = (
                            (upper_bound - current_mean) / current_std
                        ) * target_std + target_mean
                        normalized_flat = torch.where(
                            channel_flat < lower_bound,
                            normalized_lower,
                            normalized_flat,
                        )
                        normalized_flat = torch.where(
                            channel_flat > upper_bound,
                            normalized_upper,
                            normalized_flat,
                        )

                    t[i, c] = normalized_flat.reshape(original_shape)
                else:
                    t[i, c] = channel_data - current_mean + target_mean

    # NB: lerp from the untouched original, not latents_copy["samples"] --
    # that name aliases `t`, which has been mutated in place above.
    return torch.lerp(latents["samples"], t, factor)


def _latent(b=2, c=5, f=4, h=6, w=7, seed=0):
    torch.manual_seed(seed)
    return {"samples": torch.randn(b, c, f, h, w)}


@pytest.mark.parametrize("per_frame", [False, True])
@pytest.mark.parametrize("factor", [1.0, 0.5])
def test_adain_matches_reference_loop(per_frame, factor):
    latents, reference = _latent(seed=1), _latent(seed=2)

    expected = _reference_adain(latents, reference, factor, per_frame)
    actual = latent_norm.LTXVAdainLatent().batch_normalize(
        latents, reference, factor, per_frame
    )[0]["samples"]

    assert torch.allclose(
        actual, expected, atol=1e-5
    ), f"max diff {(actual - expected).abs().max().item()}"


def test_adain_broadcasts_single_frame_reference():
    latents = _latent(seed=3)
    reference = _latent(f=1, seed=4)

    expected = _reference_adain(latents, reference, 1.0, per_frame=True)
    actual = latent_norm.LTXVAdainLatent().batch_normalize(
        latents, reference, 1.0, True
    )[0]["samples"]

    assert torch.allclose(actual, expected, atol=1e-5)


def test_adain_does_not_mutate_the_reference_input():
    """The old code wrote a repeat() back into the caller's reference dict."""
    latents = _latent(seed=5)
    reference = _latent(f=1, seed=6)
    before = reference["samples"].clone()

    latent_norm.LTXVAdainLatent().batch_normalize(latents, reference, 1.0, True)

    assert reference["samples"].shape == before.shape
    assert torch.equal(reference["samples"], before)


def test_adain_rejects_reference_shorter_than_latents():
    latents = _latent(f=8, seed=7)
    reference = _latent(f=4, seed=8)
    with pytest.raises(ValueError, match="more frames than reference"):
        latent_norm.LTXVAdainLatent().batch_normalize(latents, reference, 1.0, True)


def test_adain_constant_channel_no_longer_produces_nan():
    """Deliberate behaviour change: 0/0 used to NaN out the whole latent."""
    latents = {"samples": torch.ones(1, 2, 2, 3, 3)}  # zero variance everywhere
    reference = _latent(b=1, c=2, f=2, h=3, w=3, seed=9)

    actual = latent_norm.LTXVAdainLatent().batch_normalize(
        latents, reference, 1.0, False
    )[0]["samples"]

    assert torch.isfinite(actual).all()
    assert torch.isnan(_reference_adain(latents, reference, 1.0, False)).any()


@pytest.mark.parametrize("clip_outliers", [False, True])
@pytest.mark.parametrize("percentile,factor", [(95.0, 1.0), (80.0, 0.5), (100.0, 1.0)])
def test_statnorm_matches_reference_loop(clip_outliers, percentile, factor):
    latents = _latent(seed=10)

    expected = _reference_statnorm(latents, 0.0, 1.0, percentile, factor, clip_outliers)
    actual = latent_norm.LTXVStatNormLatent().statistical_normalize(
        latents, 0.0, 1.0, percentile, factor, clip_outliers
    )[0]["samples"]

    assert torch.allclose(
        actual, expected, atol=1e-5
    ), f"max diff {(actual - expected).abs().max().item()}"


def test_statnorm_matches_reference_with_nonzero_targets():
    latents = _latent(seed=11)

    expected = _reference_statnorm(latents, 0.25, 2.0, 90.0, 1.0, True)
    actual = latent_norm.LTXVStatNormLatent().statistical_normalize(
        latents, 0.25, 2.0, 90.0, 1.0, True
    )[0]["samples"]

    assert torch.allclose(actual, expected, atol=1e-5)


def test_statnorm_constant_channel_matches_reference_fallback():
    """Zero-spread channel must take the shift-by-mean branch, as before."""
    samples = torch.randn(1, 3, 2, 4, 4)
    samples[0, 1] = 5.0  # constant channel -> std below threshold
    latents = {"samples": samples}

    expected = _reference_statnorm(latents, 0.0, 1.0, 95.0, 1.0, False)
    actual = latent_norm.LTXVStatNormLatent().statistical_normalize(
        latents, 0.0, 1.0, 95.0, 1.0, False
    )[0]["samples"]

    assert torch.allclose(actual, expected, atol=1e-5)
    assert torch.isfinite(actual).all()


def test_norm_nodes_do_not_deepcopy_the_latent():
    """copy.deepcopy cloned the full latent every step; these run per step."""
    source = (REPO / "latent_norm.py").read_text(encoding="utf-8")
    assert "copy.deepcopy" not in source
    assert "import copy" not in source


def test_batched_quantile_matches_torch_quantile():
    flat = torch.randn(2, 6, 500)
    q = torch.tensor([0.025, 0.975])

    expected = torch.quantile(flat, q, dim=-1)
    assert torch.allclose(latent_norm._batched_quantile(flat, q), expected, atol=1e-6)


def test_batched_quantile_chunks_without_changing_results(monkeypatch):
    """Force the chunked path and confirm it agrees with the direct call."""
    flat = torch.randn(2, 6, 500)
    q = torch.tensor([0.1, 0.9])
    expected = torch.quantile(flat, q, dim=-1)

    monkeypatch.setattr(latent_norm, "_QUANTILE_MAX_ELEMENTS", 1000)
    chunked = latent_norm._batched_quantile(flat, q)

    assert chunked.shape == expected.shape
    assert torch.allclose(chunked, expected, atol=1e-6)
