"""Choose the attention kernel for one model, without a CLI flag or a restart.

ComfyUI selects an attention implementation once at startup from the command line
(``--use-sage-attention`` and friends). It also supports a per-model override:
``wrap_attn`` (comfy/ldm/modules/attention.py) consults
``transformer_options["optimized_attention_override"]`` on every attention call.
This node installs that override for the model passing through it.

Why it matters on Ampere: ``q8_kernels`` needs SM 8.9, so the LTXV Q8 path is
permanently unavailable on a 3090 -- but SageAttention 2 supports SM 8.0/8.6.

**What it is actually worth, measured end to end on an RTX 3090** (Q4_0,
960x544x121, 20 steps, cfg 3.0, same seed):

    SDPA  230.8 s   (11.04 s/step)
    sage  218.4 s   (10.50 s/step)   = 1.06x

Do not expect more than that from the isolated-kernel figures. The raw
``sageattn`` call is ~2.2x faster than SDPA at this clip's attention shape, but
attention is only a fraction of a 22B DiT's step time, and ComfyUI reaches the
kernel through reshapes that the isolated benchmark does not pay. 1.06x for a
visually equivalent result is still free speed -- it is just not 2x.

Three caveats worth knowing:

* Sage's INT8 path is an approximation. Over a full 20-step render the latents
  differ from SDPA by relative L2 ~0.038 (~50.6 dB in latent space) -- visually
  equivalent in side-by-side frames, but not bit-identical. That is why this node
  defaults to "default": it is opt-in.
* The installed sageattention 2.2.0's ``sageattn`` takes no ``attn_mask``
  argument, so core sets ``SAGE_ATTENTION_SUPPORTS_MASK = False`` and demotes every
  *masked* call to SDPA. LTX's guide-mask path
  (``comfy/ldm/lightricks/model.py:_attention_with_guide_mask``) additionally
  passes ``low_precision_attention=False`` on each sub-call. So a guide with
  ``strength != 1.0`` or a ``pixel_mask`` gets none of the speedup. That demotion
  is correct, and routing through core's registered function is what preserves it.
* It is deterministic run to run: two sage renders of the same seed produced
  bit-identical latents.
"""

import logging

import comfy.ldm.modules.attention as comfy_attention
import torch

from .attention_compat import warn_if_attention_bypassed
from .nodes_registry import comfy_node

logger = logging.getLogger(__name__)

# Not a `from ... import`: on a ComfyUI without the attention registry that would
# raise at import time and take the whole node pack down. __init__.py prunes any
# node exposing NODE_UNAVAILABLE_REASON instead.
NODE_UNAVAILABLE_REASON = None
if not hasattr(comfy_attention, "get_attention_function"):
    NODE_UNAVAILABLE_REASON = (
        "this ComfyUI build has no comfy.ldm.modules.attention."
        "get_attention_function; update ComfyUI to select an attention backend"
    )

DEFAULT_BACKEND = "default"

# Ordered by how likely they are to be the one you want, so the combo reads well.
_PREFERRED_ORDER = (
    "sage",
    "sage3",
    "flash",
    "xformers",
    "pytorch",
    "sub_quad",
    "split",
)

# get_attention_function("optimized") returns the *live* module global. Inside an
# STG PatchAttention block that global is STG's own stg_attention, so resolving it
# would re-enter STG's call counter and double-advance current_idx -- silently
# shifting which attention gets perturbed. Never offer it.
_EXCLUDED_BACKENDS = frozenset({"optimized"})

_GPU_ONLY_BACKENDS = frozenset({"sage", "sage3", "flash", "xformers"})

# Read off sageattention 2.2.0's dispatch table (core.py), which covers sm75-sm121
# and raises otherwise. A floor on the installed version, not a contract; sage3 is
# deliberately absent because it is not installed here and core's own guard
# handles it.
_MIN_COMPUTE_CAPABILITY = {"sage": (7, 5), "flash": (8, 0)}


def available_backends():
    """Backend names this ComfyUI actually registered, most useful first.

    Read at call time, not import time: another pack may register a backend after
    us, and INPUT_TYPES is re-evaluated on every /object_info request.
    """
    registry = getattr(comfy_attention, "REGISTERED_ATTENTION_FUNCTIONS", None) or {}
    names = {name for name in registry if name not in _EXCLUDED_BACKENDS}
    ordered = [name for name in _PREFERRED_ORDER if name in names]
    ordered += sorted(names.difference(ordered))
    return [DEFAULT_BACKEND] + ordered


def resolve_backend(name):
    """Return the undecorated body of a registered backend, or None for default.

    Raises RuntimeError with an actionable message rather than letting the failure
    surface as a NameError or KeyError halfway through a render.
    """
    if NODE_UNAVAILABLE_REASON:
        raise RuntimeError(NODE_UNAVAILABLE_REASON)

    if name == DEFAULT_BACKEND:
        return None

    if name in _EXCLUDED_BACKENDS:
        raise RuntimeError(
            f"attention backend {name!r} resolves to whatever is currently bound, "
            "which inside an STG block is STG itself. Pick a concrete backend."
        )

    wrapper = comfy_attention.get_attention_function(name, default=None)
    if wrapper is None:
        raise RuntimeError(
            f"attention backend {name!r} is not available in this ComfyUI. "
            f"Available: {', '.join(available_backends())}. "
            "Install the 'sageattention' package to get 'sage'."
        )

    body = getattr(wrapper, "__wrapped__", None)
    if body is None:
        raise RuntimeError(
            f"attention backend {name!r} is registered but is not wrap_attn-"
            "decorated. Refusing to install it, because calling it from the "
            "override would re-enter ComfyUI's dispatch and recurse."
        )

    if name in _GPU_ONLY_BACKENDS and not torch.cuda.is_available():
        raise RuntimeError(
            f"attention backend {name!r} requires a CUDA GPU; none is available."
        )

    floor = _MIN_COMPUTE_CAPABILITY.get(name)
    if floor is not None and torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability < floor:
            raise RuntimeError(
                f"attention backend {name!r} requires compute capability "
                f"{floor[0]}.{floor[1]}+, but {torch.cuda.get_device_name()} "
                f"reports {capability[0]}.{capability[1]}."
            )

    return body


def make_attention_override(backend_name, body):
    """Build the ``optimized_attention_override`` callable.

    ``body`` is the *undecorated* function, so calling it cannot re-enter
    ``wrap_attn``'s dispatch and recurse.

    kwargs is forwarded verbatim, and that is load-bearing. It carries
    ``_inside_attn_wrapper=True``, which ``wrap_attn`` sets before calling us.
    Core's backends fall back internally -- ``attention_sage`` calls
    ``attention_pytorch`` for masked or low-precision-refused inputs -- and those
    fallbacks *are* wrap_attn-decorated. The flag is the only thing stopping them
    bouncing straight back into this override. Never strip it.
    """

    def optimized_attention_override(func, *args, **kwargs):
        # If core already selected this backend, run exactly what it would have.
        if func is body:
            return func(*args, **kwargs)
        return body(*args, **kwargs)

    optimized_attention_override.ltxv_attention_backend = backend_name
    return optimized_attention_override


@comfy_node(name="LTXVAttentionBackend")
class LTXVAttentionBackend:
    CATEGORY = "Lightricks/performance"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "set_backend"
    NODE_UNAVAILABLE_REASON = NODE_UNAVAILABLE_REASON

    DESCRIPTION = (
        "Select the attention kernel for this model only, with no CLI flag and no "
        "restart. 'sage' is the one kernel speedup available on a 3090 "
        "(q8_kernels needs SM 8.9), and measured 1.06x end to end on a 20-step "
        "960x544x121 render -- visually equivalent, but well short of the ~2.2x "
        "the isolated kernel shows, because attention is only part of a step. "
        "Masked attention (any guide with strength != 1.0 or a pixel mask) "
        "correctly falls back to SDPA and sees no speedup at all."
    )

    @classmethod
    def INPUT_TYPES(cls):
        backends = available_backends()
        return {
            "required": {
                "model": ("MODEL",),
                "backend": (
                    backends,
                    {
                        "default": DEFAULT_BACKEND,
                        "tooltip": (
                            "'default' leaves ComfyUI's startup choice alone. "
                            "Anything else installs a per-model override. Only "
                            "backends this install actually registered are listed."
                        ),
                    },
                ),
            },
            "optional": {
                "remove_existing_override": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Clear an override installed earlier in the graph (by "
                            "this node or another pack). Use with backend="
                            "'default' to get back to ComfyUI's startup choice."
                        ),
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, backend):
        # Naming `backend` here replaces ComfyUI's combo membership check, so a
        # workflow saved on a machine with sageattention opens on one without it
        # and reports why instead of "value not in list".
        try:
            resolve_backend(backend)
        except RuntimeError as exc:
            return str(exc)
        return True

    def set_backend(self, model, backend, remove_existing_override=False):
        patched = model.clone()
        transformer_options = patched.model_options["transformer_options"]

        if remove_existing_override:
            removed = transformer_options.pop("optimized_attention_override", None)
            if removed is not None:
                logger.info(
                    "LTXV: removed attention override (%s)",
                    getattr(removed, "ltxv_attention_backend", "installed elsewhere"),
                )

        # Raises here, at node execution, rather than deep inside sampling.
        body = resolve_backend(backend)
        existing = transformer_options.get("optimized_attention_override")

        if body is None:
            if existing is not None:
                logger.info(
                    "LTXV: backend='default' leaves the existing attention "
                    "override (%s) in place; set remove_existing_override to "
                    "clear it.",
                    getattr(existing, "ltxv_attention_backend", "installed elsewhere"),
                )
            return (patched,)

        if existing is not None and not hasattr(existing, "ltxv_attention_backend"):
            logger.warning(
                "LTXV: replacing an attention override installed by another node."
            )

        transformer_options["optimized_attention_override"] = make_attention_override(
            backend, body
        )
        # An object-patched attention forward bypasses this node too, not just STG.
        warn_if_attention_bypassed(patched, "LTXVAttentionBackend")
        logger.info("LTXV: attention backend -> %s", backend)
        return (patched,)
