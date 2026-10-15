"""Flat-colour segmentation — perfect for logos, documents and product
shots on clean backgrounds (no AI model required).

Algorithm: Euclidean RGB distance to a target colour with a soft falloff
band so edges stay anti-aliased instead of giving a hard staircase.
"""

import numpy as np

from .base import MaskSegmenter


class ColorSegmenter(MaskSegmenter):
    name = 'color'

    def __init__(self, target_color, tolerance=30, smooth=None):
        self.target = np.asarray(target_color, dtype=np.float32)
        self.tolerance = max(1, int(tolerance))
        self.smooth = smooth if smooth is not None else max(10.0, tolerance * 0.6)

    def segment(self, image):
        arr = np.asarray(image.convert('RGB'), dtype=np.float32)
        dist = np.sqrt(((arr - self.target) ** 2).sum(axis=2))
        alpha = np.clip((dist - self.tolerance) / self.smooth * 255.0,
                        0.0, 255.0)
        return (alpha / 255.0).astype(np.float32)


def border_color_stats(image, band_ratio=0.05):
    """Return the dominant border colour and its occupancy fraction."""
    rgb = np.asarray(image.convert('RGB'), dtype=np.float32)
    h, w, _ = rgb.shape
    bw = max(2, int(min(h, w) * band_ratio))
    border = np.concatenate([
        rgb[:bw].reshape(-1, 3),
        rgb[-bw:].reshape(-1, 3),
        rgb[:, :bw].reshape(-1, 3),
        rgb[:, -bw:].reshape(-1, 3),
    ])
    bq = (border // 40).astype(np.int32)
    codes = bq[:, 0] * 4096 + bq[:, 1] * 64 + bq[:, 2]
    codes, counts = np.unique(codes, return_counts=True)
    best = codes[int(np.argmax(counts))]
    frac = int(counts.max()) / counts.sum()
    b, g, r = best % 64, (best // 64) % 64, best // 4096
    center = 20
    color = [int(r * 40 + center), int(g * 40 + center), int(b * 40 + center)]
    return [min(255, c) for c in color], float(frac)


def auto_tolerance(image, target_color):
    """Infer a sensible colour tolerance from the border region variance.

    A tiny variance means the backdrop is solid → we can afford a small
    tolerance (safer for text/logo lines); a noisy backdrop needs a wider
    net.
    """
    rgb = np.asarray(image.convert('RGB'), dtype=np.float32)
    h, w, _ = rgb.shape
    bw = max(2, int(min(h, w) * 0.05))
    border = np.concatenate([
        rgb[:bw].reshape(-1, 3), rgb[-bw:].reshape(-1, 3),
        rgb[:, :bw].reshape(-1, 3), rgb[:, -bw:].reshape(-1, 3),
    ]).reshape(-1, 3)
    target = np.asarray(target_color, dtype=np.float32)
    dist = np.sqrt(((border - target) ** 2).sum(axis=1))
    std = float(dist.std())
    mean = float(dist.mean())
    return int(np.clip(mean + 3.0 * std, 8, 120))