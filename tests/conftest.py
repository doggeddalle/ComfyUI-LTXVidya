"""Shared test harness: stub out ComfyUI so the suite runs without an install.

pytest imports conftest before any test module, so installing the stubs here
guarantees they are in ``sys.modules`` by the time a test module imports a
repo module that does ``import comfy.samplers`` at module scope.

Only torch and kornia are real -- that is deliberate, so the kornia
compatibility checks exercise the actually-installed version.
"""

import functools
import sys
import types

import torch


def _module(name):
    module = types.ModuleType(name)
    sys.modules[name] = module
    if "." in name:  # attach to parent so `comfy.samplers` resolves
        parent, _, child = name.rpartition(".")
        setattr(sys.modules[parent], child, module)
    return module


def _install_comfy_stubs():
    comfy = _module("comfy")
    comfy.__path__ = []

    model_management = _module("comfy.model_management")
    model_management.get_torch_device = lambda: torch.device("cpu")
    model_management.intermediate_device = lambda: torch.device("cpu")
    # Large enough that headroom checks pass by default; tests that care about
    # the check monkeypatch this.
    model_management.get_free_memory = lambda device=None: 64 * 2**30
    model_management.soft_empty_cache = lambda *a, **kw: None
    model_management.free_memory = lambda *a, **kw: None
    model_management.dtype_size = lambda dtype: torch.empty(
        (), dtype=dtype
    ).element_size()

    _module("comfy.model_detection")

    utils = _module("comfy.utils")
    utils.load_torch_file = lambda *a, **kw: {}

    # ComfyUI exposes folder_paths as a top-level module.
    folder_paths = _module("folder_paths")
    folder_paths.get_full_path = lambda folder, name: None
    folder_paths.get_full_path_or_raise = lambda folder, name: name
    folder_paths.get_filename_list = lambda folder: []
    folder_paths.folder_names_and_paths = {}
    folder_paths.models_dir = ""

    class ProgressBar:
        def __init__(self, total):
            self.total = total
            self.current = 0

        def update(self, value=1):
            self.current += value

        def update_absolute(self, value, total=None):
            self.current = value

    utils.ProgressBar = ProgressBar

    _module("comfy.ldm").__path__ = []
    _module("comfy.ldm.common_dit").pad_to_patch_size = lambda x, *a, **kw: x

    lightricks = _module("comfy.ldm.lightricks")
    lightricks.__path__ = []
    ltx = _module("comfy.ldm.lightricks.model")
    ltx.BasicTransformerBlock = type("BasicTransformerBlock", (torch.nn.Module,), {})
    ltx.LTXVModel = type("LTXVModel", (torch.nn.Module,), {})
    ltx.apply_rotary_emb = lambda x, pe: x
    patchifier = _module("comfy.ldm.lightricks.symmetric_patchifier")
    patchifier.latent_to_pixel_coords = lambda *a, **kw: None

    _module("comfy.ldm.modules").__path__ = []
    attention = _module("comfy.ldm.modules.attention")

    # wrap_attn is copied verbatim from comfy/ldm/modules/attention.py. The
    # attention-backend override rides on the exact `_inside_attn_wrapper`
    # bookkeeping below -- an approximation here would let a recursion bug pass.
    # tests/test_attention_backend.py AST-compares this against core when a real
    # ComfyUI is present.
    def wrap_attn(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            remove_attn_wrapper_key = False
            try:
                if "_inside_attn_wrapper" not in kwargs:
                    transformer_options = kwargs.get("transformer_options", None)
                    remove_attn_wrapper_key = True
                    kwargs["_inside_attn_wrapper"] = True
                    if transformer_options is not None:
                        if "optimized_attention_override" in transformer_options:
                            return transformer_options["optimized_attention_override"](
                                func, *args, **kwargs
                            )
                return func(*args, **kwargs)
            finally:
                if remove_attn_wrapper_key:
                    del kwargs["_inside_attn_wrapper"]

        return wrapper

    attention.wrap_attn = wrap_attn
    attention.REGISTERED_ATTENTION_FUNCTIONS = {}

    def register_attention_function(name, func):
        if name not in attention.REGISTERED_ATTENTION_FUNCTIONS:
            attention.REGISTERED_ATTENTION_FUNCTIONS[name] = func

    def get_attention_function(name, default=...):
        # Core resolves "optimized" to the *live* module global, which is what
        # makes it unsafe inside an STG PatchAttention block. Read it the same
        # way so that behaviour is reproducible in tests.
        if name == "optimized":
            return attention.optimized_attention
        if name not in attention.REGISTERED_ATTENTION_FUNCTIONS:
            if default is ...:
                raise KeyError(f"Attention function {name} not found.")
            return default
        return attention.REGISTERED_ATTENTION_FUNCTIONS[name]

    attention.register_attention_function = register_attention_function
    attention.get_attention_function = get_attention_function

    @wrap_attn
    def attention_pytorch(q, k, v, heads, *args, **kwargs):
        # A third sentinel value, so a test can tell "fell back to pytorch" from
        # both "ran the default" (zeros) and "was skipped by STG" (v).
        return torch.full_like(q, 2.0)

    attention.attention_pytorch = attention_pytorch
    register_attention_function("pytorch", attention_pytorch)

    @wrap_attn
    def optimized_attention(q, k, v, heads, *args, **kwargs):
        # Distinguishable from `v` so tests can tell "ran" from "skipped".
        return torch.zeros_like(q)

    attention.optimized_attention = optimized_attention
    attention.optimized_attention_masked = optimized_attention

    samplers = _module("comfy.samplers")
    samplers.CFGGuider = type("CFGGuider", (), {})
    samplers.calc_cond_batch = lambda *args, **kwargs: None

    _module("comfy.model_patcher").ModelPatcher = type("ModelPatcher", (), {})
    _module("comfy.patcher_extension").CallbacksMP = type("CallbacksMP", (), {})

    comfy_api = _module("comfy_api")
    comfy_api.__path__ = []

    class _Stub:
        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, item):
            return _Stub()

        def __call__(self, *args, **kwargs):
            return _Stub()

    class _IO:
        ComfyNode = type("ComfyNode", (), {})
        Schema = _Stub
        NodeOutput = _Stub

        class Image:
            Input = staticmethod(lambda *args, **kwargs: None)
            Output = staticmethod(lambda *args, **kwargs: None)

        Mask = Image
        Boolean = Image
        Int = Image

    _module("comfy_api.latest").io = _IO


_install_comfy_stubs()
