"""Soft contact shadow for transparent cutouts.

Blurs the alpha into a grey map, offsets it slightly toward the bottom and
composites it underneath the cutout. Cheap and deterministic (no physics),
keeps the drop shadow toggleable per request.
"""

import numpy as np
from PIL import Image

from ..utils import gaussian


def soft_shadow(cutout, blur=18, opacity=0.35, offset=(0, 6)):
    """Return a NEW RGBA image: ``cutout`` plus a soft shadow layer."""
    cutout = cutout.convert('RGBA')
    a = np.asarray(cutout, dtype=np.float32)
    alpha = a[..., 3]
    if alpha.max() <= 0:
        return cutout

    body = (alpha >= 200).astype(np.float32)
    shadow = gaussian(body, max(1, int(blur))) * opacity
    shadow[shadow > 255] = 255

    h, w = shadow.shape
    ox, oy = offset
    shadow_map = np.zeros((h, w), dtype=np.float32)
    sx0, sx1 = max(0, ox), max(0, w - ox)
    sy0, sy1 = max(0, oy), max(0, h - oy)
    shadow_map[sy0:sy1, sx0:sx1] = shadow[:sy1 - sy0, :sx1 - sx0]

    canvas = np.zeros((h, w, 4), dtype=np.float32)
    canvas[..., 3] = shadow_map          # grey alpha-layer shadow
    canvas[..., 0:3] = 24.0              # very dark grey

    top = np.asarray(cutout, dtype=np.float32)
    composite = np.maximum(canvas, top)
    return Image.fromarray(np.clip(composite, 0, 255).astype(np.uint8), 'RGBA')