"""Colour-space helpers used across the refinement chain.

Pure numpy implementations (no OpenCV / no skimage) so the engine keeps
its minimal dependency footprint:

- ``rgb_to_lab`` / ``lab_to_rgb`` — perceptually uniform Lab (D65,
  standard 0..255 L*, a*, b* encoding like OpenCV's cv2.cvtColor).
- ``lab_distance`` — perceptually meaningful colour difference, used to
  judge spill / fringes where plain Euclidean RGB is too crude.
"""

import numpy as np

# OpenCV-compatible Lab encoding:
#   L  -> 0..255 (L* = L / 255 * 100)
#   a  -> 0..255 (a* = a - 128)
#   b  -> 0..255 (b* = b - 128)


def rgb_to_lab(rgb):
    """Convert RGB (any numeric array, 0..255) to Lab (float, 0..255)."""
    rgb = np.asarray(rgb, dtype=np.float32)
    r = np.clip(rgb[..., 0], 0.0, 255.0) / 255.0
    g = np.clip(rgb[..., 1], 0.0, 255.0) / 255.0
    b = np.clip(rgb[..., 2], 0.0, 255.0) / 255.0

    def _linearize(c):
        c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
        return c

    r, g, b = (_linearize(x) for x in (r, g, b))

    # sRGB D65
    X = 0.412453 * r + 0.357580 * g + 0.180423 * b
    Y = 0.212671 * r + 0.715160 * g + 0.072169 * b
    Z = 0.019334 * r + 0.119193 * g + 0.950227 * b

    def _f(t):
        delta = 6.0 / 29.0
        return np.where(t > delta ** 3, np.cbrt(t),
                        t / (3.0 * delta ** 2) + 4.0 / 29.0)

    fx, fy, fz = _f(X / 0.95047), _f(Y / 1.0), _f(Z / 1.08883)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    bb = 200.0 * (fy - fz)

    # Encode to 0..255 like OpenCV (L*100 -> 255; a*/b* +128)
    Lc = L / 100.0 * 255.0
    ac = a + 128.0
    bc = bb + 128.0
    return np.stack([Lc, ac, bc], axis=-1)


def lab_to_rgb(lab):
    """Convert Lab (float, 0..255 OpenCV-style) back to RGB (float, 0..255)."""
    lab = np.asarray(lab, dtype=np.float32)
    L = lab[..., 0] / 255.0 * 100.0
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0

    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    delta = 6.0 / 29.0

    def _finv(t):
        return np.where(t > delta, t ** 3,
                        3.0 * delta ** 2 * (t - 4.0 / 29.0))

    X = 0.95047 * _finv(fx)
    Y = _finv(fy)
    Z = 1.08883 * _finv(fz)

    r = 3.240454 * X + (-1.537138) * Y + (-0.498531) * Z
    g = (-0.969266) * X + 1.876010 * Y + 0.041556 * Z
    b = 0.055643 * X + (-0.204026) * Y + 1.057225 * Z

    def _gamma(c):
        c = np.maximum(c, 0.0)
        c = np.where(c > 0.0031308, 1.055 * np.power(c, 1.0 / 2.4) - 0.055,
                     12.92 * c)
        return c

    r, g, b = (_gamma(x) for x in (r, g, b))
    out = np.stack([r, g, b], axis=-1)
    return np.clip(out, 0.0, 1.0) * 255.0


def lab_distance(rgb_a, rgb_b):
    """Perceptually meaningful colour difference between two RGB arrays."""
    la = rgb_to_lab(rgb_a)
    lb = rgb_to_lab(rgb_b)
    return np.sqrt(((la - lb) ** 2).sum(axis=-1))