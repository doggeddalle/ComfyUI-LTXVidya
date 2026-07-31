"""Tests for LTXVAttentionBackend and the attention-bypass detector.

None of these need a GPU or the sageattention package: every backend under test
is a fake the test registers into ComfyUI's attention registry itself.

    pytest tests/
"""

import ast
import contextlib
import importlib.util
import logging
import pathlib
import sys
import types

import comfy.ldm.modules.attention as attention
import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]

_pkg = sys.modules.get("ltxpkg")
if _pkg is None:
    _pkg = types.ModuleType("ltxpkg")
    _pkg.__path__ = [str(REPO)]
    sys.modules["ltxpkg"] = _pkg


def _load(dotted, relpath):
    # Reuse an already-imported module: test_upstream_fixes.py loads ltxpkg.stg,
    # and executing it twice would build a second STGFlag class (breaking
    # isinstance checks) and re-read the preset JSON.
    existing = sys.modules.get(dotted)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(dotted, REPO / relpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(dotted, None)
        raise
    return module


attention_compat = _load("ltxpkg.attention_compat", "attention_compat.py")
attention_backend = _load("ltxpkg.attention_backend", "attention_backend.py")
stg = _load("ltxpkg.stg", "stg.py")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class FakeModelPatcher:
    """Mirrors the ModelPatcher surface these code paths touch.

    clone() copies model_options the way comfy.utils.deepcopy_list_dict does --
    recursing into dicts and lists, returning everything else by reference. That
    is what makes a nested transformer_options write not leak upstream while an
    override closure survives by identity.
    """

    def __init__(self):
        self.model_options = {"transformer_options": {}}
        self.object_patches = {}
        self.object_patches_backup = {}

    @staticmethod
    def _deepcopy_list_dict(obj):
        if isinstance(obj, dict):
            return {k: FakeModelPatcher._deepcopy_list_dict(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [FakeModelPatcher._deepcopy_list_dict(v) for v in obj]
        return obj

    def clone(self):
        other = FakeModelPatcher()
        other.model_options = self._deepcopy_list_dict(self.model_options)
        other.object_patches = self.object_patches.copy()
        other.object_patches_backup = self.object_patches_backup
        return other


@contextlib.contextmanager
def registered_backend(name="fakesage", body=None):
    """Register a wrap_attn-decorated recorder under `name` for the duration."""
    calls = []

    if body is None:

        def body(q, k, v, heads, *args, **kwargs):
            calls.append(kwargs)
            return torch.full_like(q, 7.0)

    wrapped = attention.wrap_attn(body)
    attention.REGISTERED_ATTENTION_FUNCTIONS[name] = wrapped
    try:
        yield wrapped, calls
    finally:
        attention.REGISTERED_ATTENTION_FUNCTIONS.pop(name, None)


def _qkv(n=4, d=8):
    return (torch.zeros(1, n, d), torch.zeros(1, n, d), torch.ones(1, n, d))


def _node():
    return attention_backend.LTXVAttentionBackend()


# ===========================================================================
# installing the override
# ===========================================================================
def test_override_installed_on_clone_not_original():
    model = FakeModelPatcher()
    with registered_backend() as (_, _calls):
        (out,) = _node().set_backend(model, "fakesage")
    assert out is not model
    assert "optimized_attention_override" in out.model_options["transformer_options"]
    assert (
        "optimized_attention_override" not in model.model_options["transformer_options"]
    )


def test_override_survives_a_further_clone():
    """STGGuider.__init__ clones the model again; the override must ride along."""
    model = FakeModelPatcher()
    with registered_backend():
        (out,) = _node().set_backend(model, "fakesage")
    again = out.clone()
    a = out.model_options["transformer_options"]
    b = again.model_options["transformer_options"]
    assert a is not b, "transformer_options must be a distinct dict per clone"
    assert a["optimized_attention_override"] is b["optimized_attention_override"]


def test_default_backend_installs_nothing():
    model = FakeModelPatcher()
    (out,) = _node().set_backend(model, attention_backend.DEFAULT_BACKEND)
    assert out is not model
    assert (
        "optimized_attention_override" not in out.model_options["transformer_options"]
    )


def test_remove_existing_override_clears_it():
    model = FakeModelPatcher()
    with registered_backend():
        (out,) = _node().set_backend(model, "fakesage")
    (cleared,) = _node().set_backend(
        out, attention_backend.DEFAULT_BACKEND, remove_existing_override=True
    )
    assert (
        "optimized_attention_override"
        not in cleared.model_options["transformer_options"]
    )


# ===========================================================================
# backend discovery and validation
# ===========================================================================
def test_backend_list_is_derived_from_registry():
    before = attention_backend.available_backends()
    assert before[0] == attention_backend.DEFAULT_BACKEND
    assert "fakesage" not in before
    with registered_backend():
        during = attention_backend.available_backends()
        assert "fakesage" in during
        options = _node().INPUT_TYPES()["required"]["backend"]
        assert "fakesage" in options[0]
        assert options[1]["default"] == attention_backend.DEFAULT_BACKEND
    assert "fakesage" not in attention_backend.available_backends()


def test_optimized_is_never_offered():
    """get_attention_function('optimized') returns whatever is bound right now,
    which inside an STG block is STG itself -- offering it would double-advance
    STG's attention counter."""
    with registered_backend(name="optimized"):
        assert "optimized" not in attention_backend.available_backends()
        with pytest.raises(RuntimeError, match="currently bound"):
            attention_backend.resolve_backend("optimized")


def test_missing_backend_raises_readable_error():
    model = FakeModelPatcher()
    with pytest.raises(RuntimeError) as excinfo:
        _node().set_backend(model, "sage")
    message = str(excinfo.value)
    assert "sage" in message
    assert "Available" in message and attention_backend.DEFAULT_BACKEND in message


def test_validate_inputs_reports_unavailable_backend():
    assert _node().VALIDATE_INPUTS(attention_backend.DEFAULT_BACKEND) is True
    result = _node().VALIDATE_INPUTS("sage")
    assert isinstance(result, str) and "sageattention" in result


def test_resolve_returns_the_undecorated_body():
    with registered_backend() as (wrapped, _):
        assert attention_backend.resolve_backend("fakesage") is wrapped.__wrapped__


def test_refuses_a_backend_that_is_not_wrap_attn_decorated():
    def bare(q, k, v, heads, *args, **kwargs):
        return q

    attention.REGISTERED_ATTENTION_FUNCTIONS["bare"] = bare
    try:
        with pytest.raises(RuntimeError, match="recurse"):
            attention_backend.resolve_backend("bare")
    finally:
        attention.REGISTERED_ATTENTION_FUNCTIONS.pop("bare", None)


# ===========================================================================
# dispatch
# ===========================================================================
def test_fake_backend_receives_the_call():
    model = FakeModelPatcher()
    with registered_backend() as (_, calls):
        (out,) = _node().set_backend(model, "fakesage")
        options = out.model_options["transformer_options"]
        q, k, v = _qkv()
        result = attention.optimized_attention(q, k, v, 1, transformer_options=options)
    assert len(calls) == 1
    assert torch.equal(result, torch.full_like(q, 7.0)), "core's default ran instead"


def test_override_does_not_recurse():
    """A backend that falls back to a wrap_attn-decorated function must not
    bounce back into the override. `_inside_attn_wrapper` is what prevents it,
    and it only survives because the override forwards **kwargs untouched."""
    entered = []

    def body(q, k, v, heads, *args, **kwargs):
        entered.append(1)
        # Mirrors attention_sage falling back to attention_pytorch.
        return attention.attention_pytorch(q, k, v, heads, *args, **kwargs)

    model = FakeModelPatcher()
    with registered_backend(body=body):
        (out,) = _node().set_backend(model, "fakesage")
        options = out.model_options["transformer_options"]
        q, k, v = _qkv()
        result = attention.optimized_attention(q, k, v, 1, transformer_options=options)
    assert len(entered) == 1, "override re-entered; _inside_attn_wrapper was lost"
    assert torch.equal(result, torch.full_like(q, 2.0))


def test_low_precision_attention_false_routes_to_fallback():
    """The guide-mask path passes low_precision_attention=False; core's
    attention_sage honours it by handing off to SDPA. Routing through core's
    registered function is what preserves that."""

    def body(q, k, v, heads, *args, **kwargs):
        if kwargs.get("low_precision_attention", True) is False:
            return attention.attention_pytorch(q, k, v, heads, *args, **kwargs)
        return torch.full_like(q, 7.0)

    model = FakeModelPatcher()
    with registered_backend(body=body):
        (out,) = _node().set_backend(model, "fakesage")
        options = out.model_options["transformer_options"]
        q, k, v = _qkv()
        result = attention.optimized_attention(
            q, k, v, 1, transformer_options=options, low_precision_attention=False
        )
    assert torch.equal(result, torch.full_like(q, 2.0)), "should have used the fallback"


# ===========================================================================
# composition with STG
# ===========================================================================
def test_stg_rebind_preserves_the_override():
    """STG rebinds the module global to an undecorated method, then delegates to
    the binding it captured -- which is still wrap_attn-decorated. So the
    override still fires, and only for attentions STG did not skip."""
    model = FakeModelPatcher()
    with registered_backend() as (_, calls):
        (out,) = _node().set_backend(model, "fakesage")
        options = out.model_options["transformer_options"]
        q, k, v = _qkv()
        patch = stg.PatchAttention([99])  # 99 is never reached -> nothing skipped
        with patch:
            result = attention.optimized_attention(
                q, k, v, 1, transformer_options=options
            )
    assert len(calls) == 1, "the backend was bypassed inside a PatchAttention block"
    assert torch.equal(result, torch.full_like(q, 7.0))
    assert patch.current_idx == 0, "STG's attention counter was disturbed"


def test_stg_skip_bypasses_the_backend():
    model = FakeModelPatcher()
    with registered_backend() as (_, calls):
        (out,) = _node().set_backend(model, "fakesage")
        options = out.model_options["transformer_options"]
        q, k, v = _qkv()
        with stg.PatchAttention([0]):  # index 0 IS skipped
            result = attention.optimized_attention(
                q, k, v, 1, transformer_options=options
            )
    assert not calls, "a skipped attention must never reach the kernel"
    assert torch.equal(result, v)


def test_patch_attention_restores_globals():
    original = attention.optimized_attention
    original_masked = attention.optimized_attention_masked
    with stg.PatchAttention([0]):
        assert attention.optimized_attention is not original
    assert attention.optimized_attention is original
    assert attention.optimized_attention_masked is original_masked


# ===========================================================================
# the bypass detector
# ===========================================================================
def test_bypass_detector_fires_on_attn_forward_patch(caplog):
    model = FakeModelPatcher()
    key = "diffusion_model.transformer_blocks.0.attn1.forward"
    model.object_patches[key] = object()
    assert attention_compat.find_attention_object_patches(model) == [key]
    with caplog.at_level(logging.WARNING):
        assert attention_compat.warn_if_attention_bypassed(model, "test") is True
    assert len(caplog.records) == 1
    assert key in caplog.text


def test_bypass_detector_ignores_unrelated_patches(caplog):
    model = FakeModelPatcher()
    model.object_patches.update(
        {
            "manual_cast_dtype": object(),
            "model_sampling": object(),
            # a weight patch on the very module we care about -- the sharp case
            "diffusion_model.transformer_blocks.0.attn1.to_q.weight": object(),
        }
    )
    assert attention_compat.find_attention_object_patches(model) == []
    with caplog.at_level(logging.WARNING):
        assert attention_compat.warn_if_attention_bypassed(model, "test") is False
    assert caplog.records == []


def test_bypass_detector_reads_backup_patches():
    """object_patches_backup is shared by reference across clones, so it catches
    a sibling patcher having already applied a patch to the same model."""
    model = FakeModelPatcher()
    key = "diffusion_model.transformer_blocks.3.attn1.forward"
    model.object_patches_backup[key] = object()
    assert attention_compat.find_attention_object_patches(model) == [key]


def test_bypass_detector_tolerates_a_bare_patcher():
    class Bare:
        pass

    assert attention_compat.find_attention_object_patches(Bare()) == []
    assert attention_compat.warn_if_attention_bypassed(Bare(), "test") is False


def test_detector_runs_at_guider_construction_not_per_step():
    """It must be called from the three guider __init__s and nowhere that runs
    per step -- patch_model and predict_noise both do."""
    found = {}
    for relpath in ("stg.py", "guiders/multimodal_guider.py"):
        tree = ast.parse((REPO / relpath).read_text(encoding="utf-8"))
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
                calls = {
                    node.func.id
                    for node in ast.walk(fn)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                }
                if "warn_if_attention_bypassed" in calls:
                    found.setdefault(fn.name, set()).add(cls.name)

    assert found.keys() == {"__init__"}, f"called outside __init__: {found}"
    assert found["__init__"] == {
        "STGGuider",
        "STGGuiderAdvanced",
        "MultimodalGuider",
    }, found["__init__"]


# ===========================================================================
# degradation
# ===========================================================================
def test_module_degrades_when_core_lacks_get_attention_function(monkeypatch):
    """__init__.py prunes any node exposing NODE_UNAVAILABLE_REASON, so an older
    ComfyUI loses this one node instead of the whole pack."""
    monkeypatch.delattr(attention, "get_attention_function")
    sys.modules.pop("ltxpkg.attention_backend_probe", None)
    reloaded = _load("ltxpkg.attention_backend_probe", "attention_backend.py")
    try:
        assert isinstance(reloaded.NODE_UNAVAILABLE_REASON, str)
        assert reloaded.NODE_UNAVAILABLE_REASON
        assert reloaded.LTXVAttentionBackend.NODE_UNAVAILABLE_REASON
    finally:
        sys.modules.pop("ltxpkg.attention_backend_probe", None)


def test_conftest_wrap_attn_matches_core():
    """The recursion guarantee is only as good as the stub's fidelity."""
    # repo lives at <ComfyUI>/custom_nodes/<repo>, so parents[1] is the ComfyUI root
    core = REPO.parents[1] / "comfy" / "ldm" / "modules" / "attention.py"
    if not core.exists():
        pytest.skip("not installed inside a ComfyUI checkout")

    def extract(path):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "wrap_attn":
                return ast.dump(node)
        return None

    theirs = extract(core)
    ours = extract(pathlib.Path(__file__).parent / "conftest.py")
    assert theirs is not None and ours is not None
    assert ours == theirs, "conftest's wrap_attn has drifted from ComfyUI's"
