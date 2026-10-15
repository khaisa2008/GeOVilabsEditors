"""Generic boundary refinement (products, food, vehicles, fallback).

Applies a gentle, band-limited treatment: only the transition area around
the object boundary is touched — interiors keep every pixel exactly as
segmentation produced them. Feathering is gated by the colour edge map so
flat background areas never turn into a blurred skirt.

The unsharp sharpening is *edge-aware* (bilateral-style): it removes the
softening a previous blur introduced without creating ringing on crisp
silhouettes, and the amount scales with the local colour-edge strength so
flat rims stay untouched.
"""

import numpy as np
from PIL import ImageFilter

from ..utils import gaussian, dilate, erode, normalize01, bilateral_filter


def boundary_band(alpha, lo=0.05, hi=0.95):
    return (alpha > lo) & (alpha < hi)


def edge_density(image):
    """0..1 edge-strength map of a PIL image."""
    g = image.convert('L')
    e = np.asarray(g.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    return normalize01(e)


def unsharp_alpha(alpha, radius=2, amount=0.35, adaptive=True):
    """Edge-aware unsharp of the alpha.

    When ``adaptive`` the detail component (alpha − blurred) is pulled only
    where the local colour edge is strong; flat rims are left exactly as
    measured, so the mask never gains ringing artefacts on smooth gradients.
    """
    blur = gaussian(alpha, radius)
    detail = alpha - blur
    if adaptive:
        grad = np.hypot(*np.gradient(alpha))
        gate = np.clip(gaussian(grad, 1.5) * 4.0, 0.0, 1.0)
        detail = detail * gate
    return np.clip(alpha + detail * amount, 0.0, 1.0)


def edge_blur(alpha, radius, edge=None):
    """Gaussian blend that only moves pixels near actual colour detail."""
    g = gaussian(alpha, radius)
    if edge is None:
        return g
    w = np.clip(np.asarray(edge, dtype=np.float32), 0.0, 1.0)
    return g * w + alpha * (1.0 - w)


def edge_refine(alpha, image, strategy):
    band = boundary_band(alpha)
    cfg = strategy or {}
    bias = cfg.get('bias', 'neutral')
    feather = float(cfg.get('feather', 0.0))
    out = alpha.astype(np.float32).copy()

    # Products / vehicles want crisp geometry: sharpen the transition only
    # where the colour image actually has a strong edge, using the
    # edge-aware unsharp (no ringing, no invented detail).
    if cfg.get('refine') == 'edge' and bias != 'erode':
        ed = edge_density(image)
        sharp = unsharp_alpha(out, radius=2, amount=0.45, adaptive=True)
        out = np.where(band & (ed > 0.3), sharp, out)

    if bias == 'dilate':
        biased = dilate(alpha, 3)
        out = np.where(band, np.maximum(out, biased), out)
    elif bias == 'erode':
        biased = erode(alpha, 3)
        out[band] = biased[band]

    if feather > 0:
        ed = edge_density(image)
        feathered = edge_blur(out, feather, ed)
        out = np.where(band, feathered, out)

    if bias == 'erode':
        out[~band] = alpha[~band]
    return np.clip(out, 0.0, 1.0)