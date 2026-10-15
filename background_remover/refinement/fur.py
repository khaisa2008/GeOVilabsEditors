"""Animal fur / feather / whisker refinement.

Fur and feathers produce dense, chaotic high-frequency edges. We use a
stronger dilation bias than human hair and avoid any erosion so fine hairs
and whiskers survive. The dilation is **colour-gated in Lab space** (like
hair) so it broadens into genuine fur-coloured pixels and never into
background halo — but the influence radius scales locally: denser fur gets a
stronger pull, sparse whiskers keep their original thin footprint.
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
    fine = _laplacian_mag(_lum, 1.2)
    coarse = _laplacian_mag(_lum, 4.0)
    strand = normalize01(np.clip(fine - 0.7 * coarse, 0.0, None))

    band = boundary_band(alpha)
    hf = strand * band.astype(np.float32)
    hf = gaussian(hf, 5.0)
    w = hf / (hf.max() + 1e-6)

    # Dilate more where fur is dense and fine (high strand energy), less on
    # broad flat areas so whiskers are not swallowed into a block.
    radius_map = np.clip(3 + 4 * w, 3, 7).round().astype(int)
    dilated = alpha.copy()
    for k in (3, 5, 7):
        pull = (radius_map >= k).astype(np.float32)
        if pull.mean() > 0.001:
            d = dilate(alpha, k)
            dilated = np.where(pull > 0, np.maximum(dilated, d), dilated)

    if cfg.get('edge_gate', True):
        sim = fg_similarity_lab(rgb, alpha, radius=6)
        gate = np.clip(gaussian(sim, 2.0), 0.0, 1.0)
        dilated = alpha + (dilated - alpha) * gate

    out = alpha * (1.0 - w * 0.85) + dilated * (w * 0.85)

    smooth = gaussian(out, 1.3)
    out = np.where(w > 0.15, smooth, out)
    out = np.maximum(out, alpha)        # preserve every whisker

    # Feather the outermost fringe slightly so the transition is natural.
    fringe = (alpha > 0.02) & (alpha < 0.85)
    out = np.where(fringe, gaussian(out, 0.8), out)
    return np.clip(out, 0.0, 1.0)


def _laplacian_mag(lum, sigma):
    """Absolute Laplacian magnitude after a Gaussian pre-smooth."""
    g = gaussian(lum, sigma)
    gy, gx = np.gradient(g)
    gyy, gyx = np.gradient(gy, axis=0), np.gradient(gx, axis=0)
    gxy, gxx = np.gradient(gy, axis=1), np.gradient(gx, axis=1)
    return np.abs(gyy) + np.abs(gyx + gxy) + np.abs(gxx)