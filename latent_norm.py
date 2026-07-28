import logging
from typing import Callable

import torch
from comfy.model_patcher import ModelPatcher

from .nodes_registry import comfy_node

logger = logging.getLogger(__name__)

# torch.quantile refuses inputs above 2**24 elements. Chunk over channels so a
# single large clip still takes the vectorized path rather than falling back to
# a per-channel Python loop.
_QUANTILE_MAX_ELEMENTS = 2**24


def _batched_quantile(flat: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """torch.quantile over the last dim of a (B, C, N) tensor, chunked if needed.

    Returns shape (len(quantiles), B, C).
    """
    if flat.numel() <= _QUANTILE_MAX_ELEMENTS:
        return torch.quantile(flat, quantiles, dim=-1)

    per_channel = max(1, flat.shape[-1])
    chunk = max(1, _QUANTILE_MAX_ELEMENTS // per_channel)
    results = [
        torch.quantile(flat[:, start : start + chunk], quantiles, dim=-1)
        for start in range(0, flat.shape[1], chunk)
    ]
    return torch.cat(results, dim=-1)


@comfy_node(name="LTXVAdainLatent")
class LTXVAdainLatent:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latents": ("LATENT",),
                "reference": ("LATENT",),
                "factor": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.01,
                        "round": 0.01,
                    },
                ),
            },
            "optional": {
                "per_frame": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "batch_normalize"

    CATEGORY = "Lightricks/latents"

    def batch_normalize(self, latents, reference, factor, per_frame=False):
        t = latents["samples"]  #  B x C x F x H x W
        ref = reference["samples"]

        # Match the statistics the per-element loop used to gather: per
        # (batch, channel) over the whole clip, or per (batch, channel, frame)
        # over that frame's spatial extent.
        if per_frame:
            if ref.size(2) == 1:
                logger.info("Reference has only one frame, using it for all frames")
                # expand, not repeat: this is a read-only broadcast, and the
                # old repeat() wrote back into the caller's reference dict.
                ref = ref.expand(-1, -1, t.size(2), -1, -1)
            elif t.size(2) > ref.size(2):
                raise ValueError("Latents have more frames than reference")
            else:
                ref = ref[:, :, : t.size(2)]
            dims = (-2, -1)
        else:
            dims = (-3, -2, -1)

        ref = ref[: t.size(0), : t.size(1)]

        r_sd, r_mean = torch.std_mean(ref, dim=dims, keepdim=True)
        i_sd, i_mean = torch.std_mean(t, dim=dims, keepdim=True)

        # A constant channel/frame gave 0/0 -> NaN before, which then poisoned
        # the whole latent through the lerp below. Falling back to the
        # reference mean is the meaningful AdaIn answer for that case.
        normalized = ((t - i_mean) / i_sd.clamp_min(1e-8)) * r_sd + r_mean

        return ({**latents, "samples": torch.lerp(t, normalized, factor)},)


@comfy_node(name="LTXVStatNormLatent")
class LTXVStatNormLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latents": ("LATENT",),
                "target_mean": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.01,
                        "round": 0.01,
                    },
                ),
                "target_std": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 10.0,
                        "step": 0.01,
                        "round": 0.01,
                    },
                ),
                "percentile": (
                    "FLOAT",
                    {
                        "default": 95.0,
                        "min": 50.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.1,
                        "tooltip": "Percentile of distribution to use for statistics calculation",
                    },
                ),
                "factor": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.01,
                        "round": 0.01,
                    },
                ),
                "clip_outliers": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "statistical_normalize"

    CATEGORY = "Lightricks/latents"

    def statistical_normalize(
        self, latents, target_mean, target_std, percentile, factor, clip_outliers
    ):
        t = latents["samples"]  # B x C x F x H x W

        # For 95% of distribution, we want to exclude 2.5% from each tail
        lower_percentile = (100 - percentile) / 2
        upper_percentile = 100 - lower_percentile

        batch, channels = t.shape[0], t.shape[1]
        # Statistics are per (batch, channel) over the flattened F*H*W extent.
        # torch.quantile only accepts float32/float64, and the masked
        # reductions below are more accurate in fp32 regardless.
        flat = t.reshape(batch, channels, -1).float()

        quantiles = torch.tensor(
            [lower_percentile / 100, upper_percentile / 100],
            device=flat.device,
            dtype=flat.dtype,
        )
        bounds = _batched_quantile(flat, quantiles)  # (2, batch, channels)
        lower_bound = bounds[0].unsqueeze(-1)
        upper_bound = bounds[1].unsqueeze(-1)

        in_range = (flat >= lower_bound) & (flat <= upper_bound)
        count = in_range.sum(-1, keepdim=True)

        # Mean/std of the in-range values only. Dividing the variance by
        # (count - 1) matches Tensor.std()'s default Bessel correction, which
        # the per-channel loop relied on.
        weighted = flat * in_range
        current_mean = weighted.sum(-1, keepdim=True) / count.clamp(min=1)
        variance = (((flat - current_mean) ** 2) * in_range).sum(-1, keepdim=True) / (
            count - 1
        ).clamp(min=1)
        current_std = variance.sqrt()

        normalized = ((flat - current_mean) / current_std) * target_std + target_mean

        if clip_outliers:
            normalized_lower = (
                (lower_bound - current_mean) / current_std
            ) * target_std + target_mean
            normalized_upper = (
                (upper_bound - current_mean) / current_std
            ) * target_std + target_mean
            normalized = torch.where(flat < lower_bound, normalized_lower, normalized)
            normalized = torch.where(flat > upper_bound, normalized_upper, normalized)

        # Degenerate channels, matching the original branches exactly:
        # near-zero spread -> shift by mean only; nothing in range -> untouched.
        normalized = torch.where(
            current_std > 1e-8, normalized, flat - current_mean + target_mean
        )
        normalized = torch.where(count > 0, normalized, flat)

        normalized = normalized.reshape(t.shape).to(t.dtype)
        return ({**latents, "samples": torch.lerp(t, normalized, factor)},)


class PerStepNormPatcher:
    """
    Base class for per-step normalization nodes.
    """

    @classmethod
    def required(s):
        return {
            "model": ("MODEL",),
            "factors": (
                "STRING",
                {
                    "default": "0.9, 0.75, 0.0",
                    "tooltip": "Comma-separated list of factors, each factor will be used for one step.",
                },
            ),
        }

    @classmethod
    def optional(s):
        return {}

    @staticmethod
    def patch_smooth_norm(
        model: ModelPatcher, factors: str, normalize_fn: Callable
    ) -> ModelPatcher:
        model = model.clone()
        factors = [float(x) for x in factors.split(",")]
        step = 0

        def norm_fn(args):
            nonlocal step
            latent = args["denoised"]
            factor = factors[min(step, len(factors) - 1)]
            step += 1
            out_latent = normalize_fn(latent={"samples": latent}, factor=factor)
            return out_latent[0]["samples"]

        model.set_model_sampler_post_cfg_function(norm_fn)
        return (model,)

    @classmethod
    def IS_CHANGED(s, *args, **kwargs):
        return float("NaN")


@comfy_node(name="LTXVPerStepAdainPatcher")
class LTXVPerStepAdainPatcher(PerStepNormPatcher):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": super().required()
            | {
                "reference": ("LATENT",),
            },
            "optional": super().optional()
            | {
                "per_frame": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_model"

    CATEGORY = "Lightricks/latents"

    def patch_model(self, model, factors, reference, per_frame=False):
        def cfg_adain(latent, factor):
            return LTXVAdainLatent().batch_normalize(
                latent, reference, factor, per_frame
            )

        return self.patch_smooth_norm(model, factors, cfg_adain)


@comfy_node(name="LTXVPerStepStatNormPatcher")
class LTXVPerStepStatNormPatcher(PerStepNormPatcher):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": super().required()
            | {
                "target_mean": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.01,
                        "round": 0.01,
                    },
                ),
                "target_std": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 10.0,
                        "step": 0.01,
                        "round": 0.01,
                    },
                ),
                "percentile": (
                    "FLOAT",
                    {
                        "default": 95.0,
                        "min": 50.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.1,
                        "tooltip": "Percentile of distribution to use for statistics calculation",
                    },
                ),
                "clip_outliers": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_model"

    CATEGORY = "Lightricks/latents"

    def patch_model(
        self, model, factors, target_mean, target_std, percentile, clip_outliers
    ):
        def cfg_stat_norm(latent, factor):
            return LTXVStatNormLatent().statistical_normalize(
                latent, target_mean, target_std, percentile, factor, clip_outliers
            )

        return self.patch_smooth_norm(model, factors, cfg_stat_norm)
