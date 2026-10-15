"""Human hair refinement.

Hair strands are often semi-transparent, fine and noisy. Strategy:

1. Find **strand regions** with a *local high-frequency test*: a fine
   Laplacian-of-Gaussian (sigma ≈ 1) response that stays strong where the
   broad response (sigma ≈ 3.5) is weak. Hair strands light up the fine
   channel, gentle skin/background gradients do not.
2. In those regions bias toward DILATION so fly-away strands are kept — but
   the dilation is **colour-gated in Lab space**: it only spreads into pixels
   whose colour matches the local foreground. Background-coloured halo pixels
   are never dilated into (the anti-halo requirement).
3. Light smoothing only there; broad skin/clothing edges get the generic
   treatment. Output is monotonic — we never erode a strand (requirement 5).
"""

import numpy as np

from ..utils import gaussian, dilate, normalize01
from ..matting.edge_detection import fg_similarity_lab
from .edge import boundary_band, edge_density


def refine(alpha, image, strategy, fallback=None):
    cfg = strategy or {}
    alpha = np.asarray(alpha, dtype=np.float32)
    rgb = np.asarray(image.convert('RGB'), dtype=np.float32)

    _lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    fine = _laplacian_mag(_lum, 1.0)
    coarse = _laplacian_mag(_lum, 3.5)
    strand = normalize01(np.clip(fine - 0.6 * coarse, 0.0, None))

    band = boundary_band(alpha)
    hf = strand * band.astype(np.float32)
    hf = gaussian(hf, 4.0)
    w = hf / (hf.max() + 1e-6)

    dilated = dilate(alpha, 5)          # keep thin strands
    if cfg.get('edge_gate', True):
        sim = fg_similarity_lab(rgb, alpha, radius=5)
        gate = np.clip(gaussian(sim, 1.5), 0.0, 1.0)
        dilated = alpha + (dilated - alpha) * gate

    out = alpha * (1.0 - w) + dilated * w

    # Gentle smoothing only inside the hair region; never erode strands.
    smooth = gaussian(out, 1.1)
    out = np.where(w > 0.2, smooth, out)
    out = np.maximum(out, alpha)        # monotonic — never cut hair

    return np.clip(out, 0.0, 1.0)


def _laplacian_mag(lum, sigma):
    """Absolute Laplacian magnitude after a Gaussian pre-smooth."""
    g = gaussian(lum, sigma)
    gy, gx = np.gradient(g)
    gyy, gyx = np.gradient(gy, axis=0), np.gradient(gx, axis=0)
    gxy, gxx = np.gradient(gy, axis=1), np.gradient(gx, axis=1)
    return np.abs(gyy) + np.abs(gyx + gxy) + np.abs(gxx)