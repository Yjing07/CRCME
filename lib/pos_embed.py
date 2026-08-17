"""
3D Sin-Cos positional embedding utilities for volumetric ViT.

Fixes vs original:
  - Replaced deprecated np.float with np.float64 (NumPy 1.24+)
  - Rewrote get_3d_sincos_pos_embed to handle H, W, T axes explicitly
    instead of the ambiguous grid-reshape trick that only worked for num_c=2
  - Kept get_2d_sincos_pos_embed as a thin wrapper for backward compat
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_3d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int,
                            grid_t: int, cls_token: bool = False) -> np.ndarray:
    """Generate 3D sin-cos positional embeddings for (T, H, W) volumes.

    Args:
        embed_dim: Total embedding dimension (must be divisible by 6 for even
                   split across three axes; falls back to 2:2:2 ratio).
        grid_h:    Number of patch rows (image_size // patch_size).
        grid_w:    Number of patch cols (image_size // patch_size).
        grid_t:    Number of temporal tokens (frames // frame_patch_size).
        cls_token: Prepend a zero embedding for the CLS token.

    Returns:
        pos_embed: np.ndarray of shape
                   [grid_t*grid_h*grid_w, embed_dim] (no cls)
                   [1 + grid_t*grid_h*grid_w, embed_dim] (with cls)
    """
    assert embed_dim % 4 == 0, (
        f"embed_dim must be divisible by 4, got {embed_dim}"
    )
    # Dimension split: T gets D/2, H and W each get D/4
    # Requires only embed_dim % 4 == 0 — works for 1024, 768, 512, 256 …
    dim_hw = embed_dim // 4   # per spatial axis
    dim_t  = embed_dim // 2   # temporal axis

    pos_h = np.arange(grid_h, dtype=np.float64)
    pos_w = np.arange(grid_w, dtype=np.float64)
    pos_t = np.arange(grid_t, dtype=np.float64)

    emb_h = _sincos_1d(dim_hw, pos_h)  # (H, D/4)
    emb_w = _sincos_1d(dim_hw, pos_w)  # (W, D/4)
    emb_t = _sincos_1d(dim_t,  pos_t)  # (T, D/2)

    # Broadcast over all T*H*W positions (T varies slowest)
    grid_t_idx, grid_h_idx, grid_w_idx = np.meshgrid(
        np.arange(grid_t), np.arange(grid_h), np.arange(grid_w), indexing='ij'
    )
    flat_t = grid_t_idx.reshape(-1)
    flat_h = grid_h_idx.reshape(-1)
    flat_w = grid_w_idx.reshape(-1)

    # dim_t + dim_hw + dim_hw = D/2 + D/4 + D/4 = D
    pos_embed = np.concatenate(
        [emb_t[flat_t], emb_h[flat_h], emb_w[flat_w]], axis=1
    )

    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros((1, embed_dim), dtype=np.float64), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, num_c: int,
                            cls_token: bool = False) -> np.ndarray:
    """Backward-compatible wrapper around get_3d_sincos_pos_embed.

    Maps the old (grid_size, num_c) signature to the new 3D interface,
    treating the image as square (grid_h == grid_w == grid_size) with
    num_c temporal tokens.
    """
    return get_3d_sincos_pos_embed(
        embed_dim=embed_dim,
        grid_h=grid_size,
        grid_w=grid_size,
        grid_t=num_c,
        cls_token=cls_token,
    )


def interpolate_pos_embed(model, checkpoint_model: dict) -> None:
    """Interpolate positional embeddings when input resolution changes.

    Operates in-place on checkpoint_model['pos_embed'].
    """
    if 'pos_embed' not in checkpoint_model:
        return

    pos_embed_ckpt = checkpoint_model['pos_embed']
    embed_dim = pos_embed_ckpt.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra = model.pos_embedding.shape[-2] - num_patches

    orig_size = int((pos_embed_ckpt.shape[-2] - num_extra) ** 0.5)
    new_size = int(num_patches ** 0.5)

    if orig_size != new_size:
        print(f"Interpolating pos_embed: {orig_size}x{orig_size} -> {new_size}x{new_size}")
        extra = pos_embed_ckpt[:, :num_extra]
        tokens = pos_embed_ckpt[:, num_extra:]
        tokens = tokens.reshape(-1, orig_size, orig_size, embed_dim).permute(0, 3, 1, 2)
        tokens = torch.nn.functional.interpolate(
            tokens, size=(new_size, new_size), mode='bicubic', align_corners=False
        )
        tokens = tokens.permute(0, 2, 3, 1).flatten(1, 2)
        checkpoint_model['pos_embed'] = torch.cat([extra, tokens], dim=1)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sincos_1d(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """1D sin-cos embedding.

    Args:
        embed_dim: Output dimension (must be even).
        pos:       1-D array of positions, shape (M,).

    Returns:
        emb: np.ndarray of shape (M, embed_dim).
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000 ** omega)          # (D/2,)

    pos = pos.reshape(-1)                   # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)  # (M, D)