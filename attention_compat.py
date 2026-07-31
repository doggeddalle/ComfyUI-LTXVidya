"""Detect attention implementations that bypass ComfyUI's attention dispatch.

This pack perturbs and re-routes attention in two ways, both of which go through
``comfy.ldm.modules.attention.optimized_attention``:

* STG (``stg.py``) rebinds that module global for the blocks it skips, and counts
  the calls the model makes to decide which attention to perturb.
* ``LTXVAttentionBackend`` installs
  ``transformer_options["optimized_attention_override"]``, which core consults
  inside ``wrap_attn``.

A node that instead replaces an attention module's ``forward`` outright -- via
``ModelPatcher.add_object_patch("diffusion_model....attn1.forward", fn)`` -- never
issues those calls. Both mechanisms then silently do nothing for that attention.
For STG that is worse than a no-op: with ``attn1`` fused, the first
``optimized_attention`` a block issues is ``attn2``'s, so index 0 selects the
*cross*-attention and STG perturbs something other than what was configured.

This module is deliberately free of optional dependencies so ``stg.py`` can import
it unconditionally.
"""

import logging

logger = logging.getLogger(__name__)

# ComfyUI resolves object patches by dotted attribute path, so an attention
# replacement always ends in one of these. Weight patches (`...attn1.to_q.weight`)
# and unrelated overrides (`manual_cast_dtype`, `model_sampling`) do not.
_ATTENTION_FORWARD_SUFFIXES = (".attn1.forward", ".attn2.forward", ".attn.forward")
_DIFFUSION_MODEL_PREFIX = "diffusion_model."


def find_attention_object_patches(model):
    """Return the object-patch keys that replace an attention ``forward``.

    Both dicts are scanned. ``object_patches`` holds what this patcher will apply;
    ``object_patches_backup`` holds what is *currently* applied to the underlying
    module, and ComfyUI shares that dict by reference across clones, so it also
    catches a sibling patcher having patched the same model.
    """
    keys = set()
    for attribute in ("object_patches", "object_patches_backup"):
        # getattr rather than attribute access: third-party ModelPatcher
        # subclasses exist, and the test stub is a bare class.
        patches = getattr(model, attribute, None)
        if not patches:
            continue
        try:
            candidates = list(patches)
        except TypeError:
            continue
        keys.update(
            key
            for key in candidates
            if isinstance(key, str)
            and key.startswith(_DIFFUSION_MODEL_PREFIX)
            and key.endswith(_ATTENTION_FORWARD_SUFFIXES)
        )
    return sorted(keys)


def warn_if_attention_bypassed(model, context):
    """Log one actionable warning if attention ``forward`` has been object-patched.

    Returns True when something was found, so a caller can react; nothing in this
    pack refuses to run on account of it.
    """
    offending = find_attention_object_patches(model)
    if not offending:
        return False

    logger.warning(
        "LTXV (%s): %d attention module(s) have their forward() replaced by an "
        "object patch (e.g. %s), typically by a fused sage/flash attention node. "
        "Such a replacement never calls ComfyUI's optimized_attention, so "
        "LTXVAttentionBackend has no effect on those attentions and STG perturbs "
        "the wrong one -- with attn1 fused, STG index 0 lands on the cross- "
        "attention instead of the self-attention. Use LTXVAttentionBackend for "
        "the backend and remove the fused patch, or drop STG.",
        context,
        len(offending),
        offending[0],
    )
    return True
