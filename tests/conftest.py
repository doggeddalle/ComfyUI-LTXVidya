"""Shared test harness: stub out ComfyUI so the suite runs without an install.

pytest imports conftest before any test module, so installing the stubs here
guarantees they are in ``sys.modules`` by the time a test module imports a
repo module that does ``import comfy.samplers`` at module scope.

Only torch and kornia are real -- that is deliberate, so the kornia
compatibility checks exercise the actually-installed version.
"""

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

    utils = _module("comfy.utils")

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
    _module("comfy.ldm.modules").__path__ = []
    attention = _module("comfy.ldm.modules.attention")

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
