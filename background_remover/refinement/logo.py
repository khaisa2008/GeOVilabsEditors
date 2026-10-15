"""Logo / graphic refinement — razor-sharp, keeps internal transparency.

Logos must keep crisp vector edges, small text and holes. Instead of a
soft matte we binarise the alpha (anything > 0.5 is foreground) and re-add
only a tiny anti-alias fringe so edges stay smooth but never mushy.
"""

import numpy as np

from ..utils import gaussian


def refine(alpha, image, strategy, fallback=None):
    a = np.where(alpha > 0.47, 1.0, 0.0)
    # Minimal AA only — sub-pixel edge, no feathering across the object.
    out = gaussian(a, 0.8)
    return np.clip(out, 0.0, 1.0)