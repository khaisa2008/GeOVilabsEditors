"""Cartoon / illustration refinement.

Illustrations have bold outlines and cell-shaded areas. We contrast-stretch
the alpha to make outlines pop, then keep a hair of anti-aliasing so the
result doesn't get jagged. No feathering across the drawing.
"""

import numpy as np

from ..utils import gaussian


def refine(alpha, image, strategy, fallback=None):
    out = np.clip((alpha - 0.5) * 1.7 + 0.5, 0.0, 1.0)
    out = gaussian(out, 0.6)
    return np.clip(out, 0.0, 1.0)