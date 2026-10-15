"""Plant / nature refinement (thin branches, grass, stems, flowers).

Plants are full of delicate linear structures. We bias toward keeping thin
silhouettes by dilating in high-frequency regions — colour-gated so stems
survive but background sky/ground halos don't spread — plus a mild blur so
single-pixel stems render smoothly instead of disappearing.
"""

import numpy as np

from ..utils import gaussian, dilate
from ..matting.edge_detection import fg_similarity
from .edge import boundary_band, edge_density


def refine(alpha, image, strategy, fallback=None):
    cfg = strategy or {}
    edges = edge_density(image)
    band = boundary_band(alpha)
    hf = edges * band.astype(np.float32)
    hf = gaussian(hf, 3.5)
    w = hf / (hf.max() + 1e-6)

    dilated = dilate(alpha, 3)
    if cfg.get('edge_gate', True):
        rgb = np.asarray(image.convert('RGB'), dtype=np.float32)
        sim = fg_similarity(rgb, alpha, radius=5)
        gate = np.clip(gaussian(sim, 1.5), 0.0, 1.0)
        dilated = alpha + (dilated - alpha) * gate

    out = alpha * (1.0 - w * 0.7) + dilated * (w * 0.7)
    smooth = gaussian(out, 0.9)
    out = np.where(w > 0.25, smooth, out)
    out = np.maximum(out, alpha)

    return np.clip(out, 0.0, 1.0)