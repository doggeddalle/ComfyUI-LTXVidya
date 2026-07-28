"""Tests for the RTX 3090 / LTX-2.3 GGUF performance work.

Each test pins a behaviour that an optimization must not change. ComfyUI is
stubbed by tests/conftest.py.
"""

import ast
import copy
import importlib.util
import pathlib
import re
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


def _package(dotted, relpath):
    """Register a namespace package so relative imports inside it resolve."""
    if dotted not in sys.modules:
        pkg = types.ModuleType(dotted)
        pkg.__path__ = [str(REPO / relpath)]
        sys.modules[dotted] = pkg
    return sys.modules[dotted]


def _load(dotted, relpath):
    if dotted in sys.modules:
        return sys.modules[dotted]
    spec = importlib.util.spec_from_file_location(dotted, REPO / relpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = module
    spec.loader.exec_module(module)
    return module


_package("ltxpkg.tricks", "tricks")
_package("ltxpkg.tricks.modules", "tricks/modules")
_package("ltxpkg.tricks.utils", "tricks/utils")


tiled_vae_decode = _load("ltxpkg.tiled_vae_decode", "tiled_vae_decode.py")
latent_norm = _load("ltxpkg.latent_norm", "latent_norm.py")
pyramid_blending = _load("ltxpkg.pyramid_blending", "pyramid_blending.py")


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


def test_spatiotemporal_decode_runs_and_matches_shape():
    """The chunk loop now dels its intermediates; make sure it still completes."""
    node = tiled_vae_decode.LTXVSpatioTemporalTiledVAEDecode()
    vae = _DeterministicVAE()
    latents = {"samples": _positional_latent(frames=9, height=8, width=8)}

    reference = tiled_vae_decode.LTXVTiledVAEDecode().decode(
        vae, latents, 1, 1, 1, False, "auto", "float32"
    )[0]
    out = node.decode_spatial_temporal(
        vae,
        latents,
        spatial_tiles=2,
        spatial_overlap=1,
        temporal_tile_length=4,
        temporal_overlap=1,
        last_frame_fix=False,
        working_device="auto",
        working_dtype="float32",
    )[0]

    assert out.shape == reference.shape
    assert torch.isfinite(out).all()


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


def test_apg_project_does_not_use_float64():
    """Consumer Ampere runs fp64 at 1/64 rate; fp32 is enough for this projection."""
    stg = _load("ltxpkg.stg", "stg.py")
    source = (REPO / "stg.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    project = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "project"
    )
    calls = {
        node.func.attr
        for node in ast.walk(project)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "double" not in calls

    v0 = torch.randn(1, 4, 6, 8)
    v1 = torch.randn(1, 4, 6, 8)
    parallel, orthogonal = stg.project(v0, v1)
    # The decomposition must still reconstruct the input and be orthogonal.
    assert torch.allclose(parallel + orthogonal, v0, atol=1e-4)
    assert (parallel * orthogonal).sum().abs() < 1e-2


def test_apg_project_preserves_input_dtype():
    stg = _load("ltxpkg.stg", "stg.py")
    v0 = torch.randn(1, 4, 4, 4, dtype=torch.float32)
    parallel, orthogonal = stg.project(v0, torch.randn_like(v0))
    assert parallel.dtype == torch.float32
    assert orthogonal.dtype == torch.float32


def test_find_step_matches_naive_scan_and_caches_schedule():
    dynamic_conditioning = _load(
        "ltxpkg.dynamic_conditioning", "dynamic_conditioning.py"
    )
    node = dynamic_conditioning.DynamicConditioning()
    schedule = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])

    def naive(sigma):
        for i, step_sigma in enumerate(schedule):
            if step_sigma <= sigma:
                return i
        return len(schedule) - 1

    for value in [1.5, 1.0, 0.9, 0.5, 0.05, -1.0]:
        sigma = torch.tensor([value])
        assert node.find_step(sigma, schedule) == naive(sigma)

    # The schedule is pulled to host once and reused across steps.
    assert node._step_sigmas == schedule.tolist()


def test_q8_nodes_reject_pre_ada_gpus(monkeypatch):
    """An SM86 card must get a clear message, not a CUDA failure."""
    q8_nodes = _load("ltxpkg.q8_nodes", "q8_nodes.py")
    monkeypatch.setattr(q8_nodes, "Q8_AVAILABLE", True)
    monkeypatch.setattr(q8_nodes.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(q8_nodes.torch.cuda, "get_device_capability", lambda: (8, 6))
    monkeypatch.setattr(q8_nodes.torch.cuda, "get_device_name", lambda: "RTX 3090")

    with pytest.raises(RuntimeError, match="Ada-generation GPU or newer"):
        q8_nodes.check_q8_available()


def test_q8_nodes_accept_ada_and_newer(monkeypatch):
    q8_nodes = _load("ltxpkg.q8_nodes", "q8_nodes.py")
    monkeypatch.setattr(q8_nodes, "Q8_AVAILABLE", True)
    monkeypatch.setattr(q8_nodes.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(q8_nodes.torch.cuda, "get_device_capability", lambda: (8, 9))
    monkeypatch.setattr(q8_nodes.torch.cuda, "get_device_name", lambda: "RTX 4090")

    q8_nodes.check_q8_available()


def test_q8_lora_checks_availability_before_touching_kernels():
    """quant_fn = hadamard_transform used to NameError ahead of the guard."""
    source = (REPO / "q8_nodes.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    load_lora = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "load_lora"
    )
    first = load_lora.body[0]
    while isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
        load_lora.body.pop(0)  # skip a docstring
        first = load_lora.body[0]

    assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
    assert first.value.func.id == "check_q8_available"


def test_stg_presets_are_read_with_explicit_encoding():
    source = (REPO / "stg.py").read_text(encoding="utf-8")
    assert "json.load(open(" not in source
    assert 'encoding="utf-8"' in source


def test_requirements_drop_unused_diffusers():
    requirements = (REPO / "requirements.txt").read_text(encoding="utf-8")
    assert "diffusers" not in requirements

    # Skip tests/ -- this file mentions the name in its own assertions.
    sources = [
        path
        for path in list(REPO.glob("*.py")) + list(REPO.glob("*/*.py"))
        if "tests" not in path.parts
    ]
    importers = [
        path.name
        for path in sources
        if re.search(
            r"^\s*(import diffusers|from diffusers)",
            path.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
    ]
    assert not importers, f"diffusers is actually imported by {importers}"


# ===========================================================================
# Node categories -- one coherent menu tree
# ===========================================================================
def _declared_categories():
    cats = {}
    for path in (
        list(REPO.glob("*.py"))
        + list(REPO.glob("*/*.py"))
        + list(REPO.glob("*/*/*.py"))
    ):
        if "tests" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if (
                            isinstance(target, ast.Name)
                            and target.id == "CATEGORY"
                            and isinstance(stmt.value, ast.Constant)
                        ):
                            cats[node.name] = stmt.value.value
                elif isinstance(stmt, ast.FunctionDef) and stmt.name == "define_schema":
                    for sub in ast.walk(stmt):
                        if (
                            isinstance(sub, ast.keyword)
                            and sub.arg == "category"
                            and isinstance(sub.value, ast.Constant)
                        ):
                            cats[node.name] = sub.value.value
    return cats


def test_every_category_lives_under_one_root():
    """Two roots differing only by case render as two top-level menus."""
    import ltxpkg.nodes_registry as registry  # noqa: F401

    nodes_registry = _load("ltxpkg.nodes_registry", "nodes_registry.py")
    allowed_external = tuple(
        r.casefold() for r in nodes_registry.EXTERNAL_CATEGORY_ROOTS
    )

    offenders = {}
    for cls_name, category in _declared_categories().items():
        root = category.split("/", 1)[0]
        if root == nodes_registry.DEFAULT_CATEGORY_NAME:
            continue
        if root.casefold() in allowed_external:
            continue
        offenders[cls_name] = category

    assert not offenders, f"categories outside the Lightricks tree: {offenders}"


def test_no_case_variant_roots():
    roots = {c.split("/", 1)[0] for c in _declared_categories().values()}
    lowered = [r.casefold() for r in roots]
    assert len(lowered) == len(
        set(lowered)
    ), f"roots differing only by case produce duplicate menus: {sorted(roots)}"


@pytest.mark.parametrize(
    "given,expected",
    [
        ("lightricks/LTXV", "Lightricks/LTXV"),
        ("LIGHTRICKS/latents", "Lightricks/latents"),
        ("Lightricks/latents", "Lightricks/latents"),
        ("sampling", "Lightricks/sampling"),
        ("latent/video", "Lightricks/latent/video"),
        ("api node/text/Lightricks", "api node/text/Lightricks"),
        (None, "Lightricks"),
        ("", "Lightricks"),
    ],
)
def test_normalize_category(given, expected):
    nodes_registry = _load("ltxpkg.nodes_registry", "nodes_registry.py")
    assert nodes_registry.normalize_category(given) == expected


# ===========================================================================
# Optional dependency isolation
# ===========================================================================
def test_kornia_import_is_guarded_and_broad():
    """A broken kornia must cost one node, not the whole pack.

    The except clause has to be broad: a mismatched kornia/torch build on
    Windows raises OSError from the DLL loader, not ImportError.
    """
    tree = ast.parse((REPO / "pyramid_blending.py").read_text(encoding="utf-8"))
    guarded = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        imports_kornia = any(
            isinstance(sub, ast.ImportFrom) and (sub.module or "").startswith("kornia")
            for sub in ast.walk(node)
        )
        if not imports_kornia:
            continue
        for handler in node.handlers:
            guarded.append(
                handler.type is None
                or (
                    isinstance(handler.type, ast.Name)
                    and handler.type.id == "Exception"
                )
            )
    assert guarded, "the kornia import is not inside a try block"
    assert all(guarded), "kornia guard catches too narrow an exception type"


def test_pyramid_node_declares_availability_hook():
    assert hasattr(
        pyramid_blending.LTXVLaplacianPyramidBlend, "NODE_UNAVAILABLE_REASON"
    )
    # kornia is installed in the test environment, so it must be available here.
    assert pyramid_blending.LTXVLaplacianPyramidBlend.NODE_UNAVAILABLE_REASON is None


def test_init_prunes_unavailable_nodes():
    source = (REPO / "__init__.py").read_text(encoding="utf-8")
    assert "NODE_UNAVAILABLE_REASON" in source
    assert "del NODE_CLASS_MAPPINGS[" in source


# ===========================================================================
# FETA -- same scores, without the per-call mask allocations
# ===========================================================================
def _reference_feta_score(query_image, key_image, head_dim, num_frames, weight):
    """Original implementation, kept as an oracle."""
    scale = head_dim**-0.5
    query_image = query_image * scale
    attn_temp = query_image @ key_image.transpose(-2, -1)
    attn_temp = attn_temp.to(torch.float32)
    attn_temp = attn_temp.softmax(dim=-1)
    attn_temp = attn_temp.reshape(-1, num_frames, num_frames)
    diag_mask = torch.eye(num_frames, device=attn_temp.device).bool()
    diag_mask = diag_mask.unsqueeze(0).expand(attn_temp.shape[0], -1, -1)
    attn_wo_diag = attn_temp.masked_fill(diag_mask, 0)
    num_off_diag = num_frames * num_frames - num_frames
    mean_scores = attn_wo_diag.sum(dim=(1, 2)) / num_off_diag
    return (mean_scores.mean() * (num_frames + weight)).clamp(min=1)


@pytest.mark.parametrize("num_frames,weight", [(4, 4.0), (7, 2.0), (2, 0.0)])
def test_feta_score_matches_reference(num_frames, weight):
    feta = _load(
        "ltxpkg.tricks.utils.feta_enhance_utils", "tricks/utils/feta_enhance_utils.py"
    )
    torch.manual_seed(0)
    head_dim = 8
    q = torch.randn(6, 3, num_frames, head_dim)
    k = torch.randn(6, 3, num_frames, head_dim)

    expected = _reference_feta_score(q, k, head_dim, num_frames, weight)
    actual = feta._feta_score(q, k, head_dim, num_frames, weight)

    assert torch.allclose(
        actual, expected, atol=1e-5
    ), f"{actual.item()} != {expected.item()}"


def test_feta_score_single_frame_is_finite():
    """One frame has no off-diagonal mass; the original divided by zero."""
    feta = _load(
        "ltxpkg.tricks.utils.feta_enhance_utils", "tricks/utils/feta_enhance_utils.py"
    )
    q = torch.randn(4, 2, 1, 8)
    k = torch.randn(4, 2, 1, 8)

    actual = feta._feta_score(q, k, 8, 1, 4.0)

    assert torch.isfinite(actual).all()
    assert torch.isnan(_reference_feta_score(q, k, 8, 1, 4.0)).any()


def test_feta_does_not_build_an_eye_mask_per_call():
    source = (REPO / "tricks/utils/feta_enhance_utils.py").read_text(encoding="utf-8")
    assert "torch.eye" not in source
    assert "masked_fill" not in source


# ===========================================================================
# Attention bank -- pinned host staging
# ===========================================================================
def test_attention_bank_stages_through_pinned_memory():
    source = (REPO / "tricks/modules/ltx_model.py").read_text(encoding="utf-8")
    assert "pin_memory=True" in source
    assert "_stash_on_cpu" in source
    # device->host must stay blocking; only the host->device inject is async.
    # Check actual call keywords, not the source text -- the docstring explains
    # the tradeoff and legitimately mentions non_blocking.
    tree = ast.parse(source)
    stash = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_stash_on_cpu"
    )
    async_copies = [
        n
        for n in ast.walk(stash)
        if isinstance(n, ast.Call)
        and any(kw.arg == "non_blocking" for kw in n.keywords)
    ]
    assert not async_copies, "device->host copy must not be non_blocking"


def test_stash_on_cpu_roundtrips_values():
    ltx_model = _load("ltxpkg.tricks.modules.ltx_model", "tricks/modules/ltx_model.py")
    source = torch.randn(2, 3, 4)
    assert torch.equal(ltx_model._stash_on_cpu(source), source)


# ===========================================================================
# VRAM-derived blend chunk size
# ===========================================================================
def test_blend_chunk_size_falls_back_off_cuda():
    assert (
        pyramid_blending._auto_chunk_size(512, 512, 4, torch.device("cpu"))
        == pyramid_blending._CHUNK_SIZE_FALLBACK
    )
    assert (
        pyramid_blending._auto_chunk_size(512, 512, 4, None)
        == pyramid_blending._CHUNK_SIZE_FALLBACK
    )


def test_blend_chunk_size_scales_with_free_memory(monkeypatch):
    mm = sys.modules["comfy.model_management"]

    monkeypatch.setattr(mm, "get_free_memory", lambda device=None: 16 * 2**30)
    roomy = pyramid_blending._auto_chunk_size(512, 512, 4, torch.device("cuda"))

    monkeypatch.setattr(mm, "get_free_memory", lambda device=None: 256 * 2**20)
    tight = pyramid_blending._auto_chunk_size(512, 512, 4, torch.device("cuda"))

    assert roomy > tight >= 1
    assert roomy <= pyramid_blending._CHUNK_SIZE_MAX


def test_blend_chunk_size_never_returns_zero(monkeypatch):
    mm = sys.modules["comfy.model_management"]
    monkeypatch.setattr(mm, "get_free_memory", lambda device=None: 1)
    assert pyramid_blending._auto_chunk_size(4096, 4096, 4, torch.device("cuda")) == 1


# ===========================================================================
# Diagnostics sweeps
# ===========================================================================
def _package_sources():
    return [
        p
        for p in list(REPO.glob("*.py"))
        + list(REPO.glob("*/*.py"))
        + list(REPO.glob("*/*/*.py"))
        if "tests" not in p.parts
    ]


def test_no_print_calls_remain():
    offenders = []
    for path in _package_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"print() should be logging: {offenders}"


def test_no_asserts_validate_user_input():
    """assert is stripped under python -O, taking the validation with it."""
    offenders = []
    for path in _package_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"convert to explicit raises: {offenders}"


def test_logging_calls_are_lazy():
    """logger.info('x %s', v), not logger.info('x %s' % v) -- don't pre-format."""
    offenders = []
    for path in _package_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"debug", "info", "warning", "error"}
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "logger"
            ):
                continue
            if node.args and isinstance(node.args[0], ast.BinOp):
                if isinstance(node.args[0].op, ast.Mod):
                    offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"pre-formatted log messages: {offenders}"


def test_batched_quantile_chunks_without_changing_results(monkeypatch):
    """Force the chunked path and confirm it agrees with the direct call."""
    flat = torch.randn(2, 6, 500)
    q = torch.tensor([0.1, 0.9])
    expected = torch.quantile(flat, q, dim=-1)

    monkeypatch.setattr(latent_norm, "_QUANTILE_MAX_ELEMENTS", 1000)
    chunked = latent_norm._batched_quantile(flat, q)

    assert chunked.shape == expected.shape
    assert torch.allclose(chunked, expected, atol=1e-6)
