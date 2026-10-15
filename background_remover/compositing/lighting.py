"""Lighting / color harmonisation between cutout and new background.

Minimal and safe: matches the cutout's global brightness toward the new
background so a white-cutout-blue-bg case doesn't look pasted. Applied
only to pixels whose alpha > 0.9 (the solid body), so edges stay intact.
"""

import numpy as np
from PIL import Image


def harmonize(cutout, background_mode=None, background_param=None):
    """Return a copy of ``cutout`` (RGBA) with light tonal adjustment.

    With no background info we simply return the input unchanged — this is
    a no-op by default and only activates for colour backgrounds.
    """
    if background_mode != 'color':
        return cutout

    arr = np.asarray(cutout.convert('RGBA'), dtype=np.float32)
    alpha = arr[..., 3:4]
    body = alpha > 230

    rgb = background_param.lstrip('#') if background_param else 'ffffff'
    bg = np.array([int(rgb[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)
    bg_lum = float(np.dot(bg, np.array([0.299, 0.587, 0.114])))

    fg_body = arr.copy()
    fg_body[body[..., 0]] = arr[body[..., 0]]
    fg_lum = float(np.mean(0.299 * fg_body[body[..., 0], 0]
                           + 0.587 * fg_body[body[..., 0], 1]
                           + 0.114 * fg_body[body[..., 0], 2])) \
        if body.any() else 128.0

    # Mild match: pull the body toward a middle brightness relative to bg.
    target = (bg_lum + fg_lum) / 2.0
    if fg_lum < 1.0:
        return cutout
    gain = target / fg_lum
    gain = float(np.clip(gain, 0.85, 1.18))

    out = arr.copy()
    out[..., 0:3] = np.where(body, arr[..., 0:3] * gain, arr[..., 0:3])
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), 'RGBA')