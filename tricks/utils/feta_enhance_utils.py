import torch
from einops import rearrange


def _feta_score(query_image, key_image, head_dim, num_frames, enhance_weight):
    # A single frame has no off-diagonal attention to measure; the original
    # arithmetic divided by zero here and returned NaN, which then scaled the
    # whole block output.
    num_off_diag = num_frames * num_frames - num_frames
    if num_off_diag == 0:
        return torch.ones((), device=query_image.device, dtype=torch.float32)

    scale = head_dim**-0.5
    attn_temp = (query_image * scale) @ key_image.transpose(-2, -1)
    # softmax(dtype=...) fuses the float32 upcast. Casting first materialized a
    # second full (B*S, heads, T, T) tensor before the softmax even ran.
    attn_temp = torch.softmax(attn_temp, dim=-1, dtype=torch.float32)

    # Reshape to [batch_size * num_tokens, num_frames, num_frames]
    attn_temp = attn_temp.reshape(-1, num_frames, num_frames)

    # Sum the off-diagonal entries directly. The previous version built a
    # boolean eye mask, expanded it to the full attention shape, and produced a
    # masked *copy* of the attention matrix -- three full-size allocations per
    # call, on every FETA-enabled block of every sampling step.
    total = attn_temp.sum(dim=(1, 2))
    diagonal = attn_temp.diagonal(dim1=1, dim2=2).sum(dim=-1)
    mean_scores = (total - diagonal) / num_off_diag

    enhance_scores = mean_scores.mean() * (num_frames + enhance_weight)
    return enhance_scores.clamp(min=1)


def get_feta_scores(img_q, img_k, num_heads, transformer_options):
    num_frames = transformer_options["original_shape"][2]
    _, ST, dim = img_q.shape
    head_dim = dim // num_heads
    spatial_dim = ST // num_frames

    query_image = rearrange(
        img_q,
        "B (T S) (N C) -> (B S) N T C",
        T=num_frames,
        S=spatial_dim,
        N=num_heads,
        C=head_dim,
    )
    key_image = rearrange(
        img_k,
        "B (T S) (N C) -> (B S) N T C",
        T=num_frames,
        S=spatial_dim,
        N=num_heads,
        C=head_dim,
    )
    weight = transformer_options.get("feta_weight", 0)
    return _feta_score(query_image, key_image, head_dim, num_frames, weight)
