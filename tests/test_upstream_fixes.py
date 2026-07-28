"""Regression tests for the compatibility/crash fixes applied to this fork.

ComfyUI is stubbed by tests/conftest.py; torch and kornia are real, so the
kornia compatibility checks exercise the actual installed version.

    pytest tests/
"""

import ast
import importlib.util
import pathlib
import sys
import types

import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# load repo modules (the package directory name contains a hyphen)
# ---------------------------------------------------------------------------
_pkg = types.ModuleType("ltxpkg")
_pkg.__path__ = [str(REPO)]
sys.modules["ltxpkg"] = _pkg


def _load(dotted, relpath):
    spec = importlib.util.spec_from_file_location(dotted, REPO / relpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = module
    spec.loader.exec_module(module)
    return module


def _load_func_by_ast(relpath, func_name):
    """Extract one self-contained function without importing its module."""
    path = REPO / relpath
    tree = ast.parse(path.read_text(encoding="utf-8"))
    func = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == func_name
    )
    module = ast.Module(body=[func], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[func_name]


pyramid_blending = _load("ltxpkg.pyramid_blending", "pyramid_blending.py")
stg = _load("ltxpkg.stg", "stg.py")
_config_value_matches = _load_func_by_ast(
    "text_embeddings_connectors.py", "_config_value_matches"
)


# ===========================================================================
# kornia compatibility -- an ImportError here takes down the whole node pack
# ===========================================================================
def test_only_public_kornia_pyramid_api_is_imported():
    """`pad` / the power-of-two helpers must not be imported from kornia.

    kornia dropped the `pad` re-export in 0.8.x, and the power-of-two helpers
    have never been in kornia's `__all__`.
    """
    tree = ast.parse((REPO / "pyramid_blending.py").read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("kornia")
        for alias in node.names
    }
    assert not imported & {"pad", "is_powerof_two", "find_next_powerof_two"}
    assert imported <= {"PyrUp", "build_laplacian_pyramid", "build_pyramid"}


def test_public_kornia_pyramid_api_available_in_installed_version():
    from kornia.geometry.transform.pyramid import (  # noqa: F401
        PyrUp,
        build_laplacian_pyramid,
        build_pyramid,
    )


def test_power_of_two_helpers_match_reference():
    for n in range(1, 300):
        assert pyramid_blending.is_powerof_two(n) == (n & (n - 1) == 0)
        nxt = pyramid_blending.find_next_powerof_two(n)
        assert nxt >= n
        assert pyramid_blending.is_powerof_two(nxt)
        assert nxt // 2 < n
    assert not pyramid_blending.is_powerof_two(0)


@pytest.mark.parametrize("size", [(64, 64), (67, 129), (100, 100), (33, 65)])
def test_pyramid_blend_runs_and_preserves_shape(size):
    height, width = size
    image_a = torch.rand(3, 3, height, width)
    image_b = torch.rand(3, 3, height, width)
    mask = torch.rand(3, 1, height, width)

    out = pyramid_blending._pyramid_blend(image_a, image_b, mask, max_level=4)

    assert out.shape == (3, 3, height, width)
    assert torch.isfinite(out).all()


def test_pyramid_blend_selects_correct_source():
    """A solid mask must return one source untouched, padding notwithstanding."""
    image_a = torch.full((1, 3, 96, 96), 0.8)
    image_b = torch.full((1, 3, 96, 96), 0.2)

    blended_a = pyramid_blending._pyramid_blend(
        image_a, image_b, torch.ones(1, 1, 96, 96), max_level=4
    )
    blended_b = pyramid_blending._pyramid_blend(
        image_a, image_b, torch.zeros(1, 1, 96, 96), max_level=4
    )

    assert torch.allclose(blended_a, image_a, atol=1e-4)
    assert torch.allclose(blended_b, image_b, atol=1e-4)


# ===========================================================================
# apg() call site must match its signature
# ===========================================================================
def test_apg_call_sites_use_supported_kwargs_only():
    import inspect

    params = set(inspect.signature(stg.apg).parameters)
    tree = ast.parse((REPO / "stg.py").read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "apg"
    ]

    assert calls, "expected at least one apg() call site"
    for call in calls:
        for keyword in call.keywords:
            assert keyword.arg in params, f"apg() got unsupported kwarg {keyword.arg!r}"


def test_apg_runs_with_call_site_kwargs():
    pos = torch.rand(1, 4, 8, 8)
    neg = torch.rand(1, 4, 8, 8)

    out = stg.apg(pos, neg, cfg_scale=3.0, eta=1.0, norm_threshold=0.0)

    assert out.shape == pos.shape
    assert torch.isfinite(out).all()


# ===========================================================================
# STG index counting vs. comfy's partitioned guide-mask self-attention
# ===========================================================================
SEQ, DIM, HEADS = 40, 16, 2
GUIDE_START, TRACKED_END = 10, 25


def _make_patch(attn_idx):
    patch = stg.PatchAttention(attn_idx)
    from comfy.ldm.modules import attention

    patch.original_attention = attention.optimized_attention
    patch.original_attention_masked = attention.optimized_attention_masked
    return patch


def _run_guide_mask_attention(patch, v):
    """Mimic comfy's _attention_with_guide_mask: 3 calls over query slices.

    Only the first two sub-calls pass low_precision_attention=False, matching
    comfy/ldm/lightricks/model.py.
    """
    q = torch.rand(1, SEQ, DIM)
    out = torch.empty(1, SEQ, DIM)
    out[:, :GUIDE_START, :] = patch.stg_attention(
        q[:, :GUIDE_START, :], v, v, HEADS, low_precision_attention=False
    )
    out[:, GUIDE_START:TRACKED_END, :] = patch.stg_attention(
        q[:, GUIDE_START:TRACKED_END, :], v, v, HEADS, low_precision_attention=False
    )
    out[:, TRACKED_END:, :] = patch.stg_attention(q[:, TRACKED_END:, :], v, v, HEADS)
    return out


def test_guide_mask_split_counts_as_one_attention_index():
    """One logical self-attention must consume exactly one STG index."""
    patch = _make_patch([99])  # skip nothing

    _run_guide_mask_attention(patch, torch.rand(1, SEQ, DIM))

    assert patch.current_idx == 0, (
        f"partitioned sub-calls consumed {patch.current_idx + 1} indices; "
        "every later index (e.g. audio_attn_idx) would be shifted"
    )
    assert patch._guide_offset == 0, "partition state must reset after full coverage"


def test_guide_mask_split_skip_returns_matching_slices():
    """Skipping must yield the matching slice of v, not the whole sequence."""
    patch = _make_patch([0])
    v = torch.rand(1, SEQ, DIM)

    out = _run_guide_mask_attention(patch, v)  # raises on shape mismatch

    assert torch.allclose(out, v)


def test_consecutive_attentions_get_successive_indices():
    patch = _make_patch([99])
    v = torch.rand(1, SEQ, DIM)

    _run_guide_mask_attention(patch, v)
    _run_guide_mask_attention(patch, v)

    assert patch.current_idx == 1


def test_two_call_partition_when_guide_start_is_zero():
    """guide_start == 0 produces only two sub-calls; still one index."""
    patch = _make_patch([0])
    v = torch.rand(1, SEQ, DIM)
    q = torch.rand(1, SEQ, DIM)

    first = patch.stg_attention(
        q[:, :TRACKED_END, :], v, v, HEADS, low_precision_attention=False
    )
    second = patch.stg_attention(q[:, TRACKED_END:, :], v, v, HEADS)

    assert first.shape[1] == TRACKED_END
    assert second.shape[1] == SEQ - TRACKED_END
    assert patch.current_idx == 0
    assert patch._guide_offset == 0
    assert torch.allclose(torch.cat([first, second], dim=1), v)


def test_plain_attention_keeps_one_call_per_index():
    """Unpartitioned attention must behave exactly as before."""
    patch = _make_patch([1])
    v = torch.rand(1, SEQ, DIM)
    q = torch.rand(1, SEQ, DIM)

    assert torch.allclose(patch.stg_attention(q, v, v, HEADS), torch.zeros_like(q))
    assert torch.allclose(patch.stg_attention(q, v, v, HEADS), v)  # index 1 skipped
    assert patch.current_idx == 1


# ===========================================================================
# case-insensitive checkpoint config comparison
# ===========================================================================
def test_config_value_matches_ignores_string_case():
    assert _config_value_matches("per_token_rms", "per_token_rms")
    assert _config_value_matches("PER_TOKEN_RMS", "per_token_rms")
    assert _config_value_matches("Per_Token_RMS", "per_token_rms")
    assert not _config_value_matches("other_norm", "per_token_rms")


def test_config_value_matches_keeps_non_strings_strict():
    assert _config_value_matches(False, False)
    assert not _config_value_matches("False", False)
    assert not _config_value_matches(True, False)
    assert not _config_value_matches(None, False)


# ===========================================================================
# multimodal guider must bind its predictions before the post-cfg replay
# ===========================================================================
def test_multimodal_guider_binds_predictions_unconditionally():
    source = (REPO / "guiders" / "multimodal_guider.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    func = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            isinstance(stmt, ast.For)
            and "sampler_post_cfg_function" in ast.dump(stmt.iter)
            for stmt in ast.walk(node)
        )
    )
    unconditional = {
        target.id
        for stmt in func.body
        if isinstance(stmt, ast.Assign)
        for target in stmt.targets
        if isinstance(target, ast.Name)
    }

    for name in ("noise_pred_neg", "noise_pred_perturbed"):
        assert name in unconditional, (
            f"{name} is only bound inside a conditional branch, so the "
            "sampler_post_cfg_function replay raises UnboundLocalError when "
            "that branch is skipped (cfg=1.0 / stg=0)"
        )
