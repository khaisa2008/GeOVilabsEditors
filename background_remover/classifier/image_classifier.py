"""Adaptive Image Classifier (no heavy model required).

A multi-signal heuristic classifier that inspects color, edge, texture,
saturation and skin-tone statistics on a downscaled copy of the image and
maps it to one of the GeOViLabs content categories.

Design decisions
----------------
- Runs in <150ms on a 512px preview with pure numpy + Pillow.
- Returns a confidence score; if confidence is low the pipeline treats the
  image as ``MIXED`` and falls back to the safest general strategy.
- Never guesses a category aggressively: rules are ordered so that only
  strongly-supported predictions win.
"""

import numpy as np
from PIL import Image, ImageFilter

from ..categories import (
    HUMAN, ANIMAL, PLANT, PRODUCT, LOGO, CARTOON,
    DOCUMENT, FOOD, VEHICLE, MIXED,
)
from ..utils import normalize01

PREVIEW = 512


class Classification:
    __slots__ = ('category', 'confidence', 'features')

    def __init__(self, category, confidence, features=None):
        self.category = category
        self.confidence = float(confidence)
        self.features = features or {}

    def as_dict(self):
        return {
            'image_type': self.category,
            'confidence': round(self.confidence, 3),
            'features': {k: round(float(v), 4) for k, v in self.features.items()},
        }


def _skin_ratio(rgb):
    """Fraction of pixels that look like human skin (RGB heuristic)."""
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    skin = (r > 60) & (g > 30) & (b > 10) & (r > g) & (g > b) \
        & ((mx - mn) > 12) & (r - g >= 10) & (mx > 90)
    return float(skin.mean())


def _quantized_colors(img):
    """Estimate palette size with PIL quantization on a small copy."""
    try:
        q = img.convert('RGB').quantize(colors=24, method=Image.Quantize.MEDIANCUT)
        return len(q.getpalette()) // 3
    except Exception:
        arr = np.asarray(img.resize((96, 96), Image.BILINEAR)).reshape(-1, 3)
        codes = arr[:, 0] * 65_536 + arr[:, 1] * 256 + arr[:, 2]
        return int(np.unique(codes).size)


def _edge_stats(img_gray):
    edges = np.asarray(img_gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    edges = normalize01(edges)
    return {
        'edge_density': float((edges > 0.18).mean()),
        'edge_mean': float(edges.mean()),
    }


def _background_uniformity(rgb):
    """Fraction the dominant border colour occupies in the image."""
    h, w, _ = rgb.shape
    bw = max(2, int(w * 0.04))
    border = np.concatenate([
        rgb[:bw].reshape(-1, 3),
        rgb[-bw:].reshape(-1, 3),
        rgb[:, :bw].reshape(-1, 3),
        rgb[:, -bw:].reshape(-1, 3),
    ])
    bq = (border // 40).astype(np.int32)
    codes = bq[:, 0] * 4096 + bq[:, 1] * 64 + bq[:, 2]
    codes, counts = np.unique(codes, return_counts=True)
    top = counts.max() / counts.sum()
    return float(top), int(codes.size)


def _green_ratio(hsv_h, s):
    green = ((hsv_h >= 60) & (hsv_h <= 165) & (s > 0.25))
    return float(green.mean())


def _colorfulness(rgb):
    """Hasler-Süsstrunk colorfulness metric."""
    r, g, b = (rgb[..., i].astype(np.float32) for i in range(3))
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(
        rg.mean() ** 2 + yb.mean() ** 2))


def _noise_level(img_gray):
    """Estimated sensor noise = std of a blurred-difference on a small copy."""
    small = img_gray.resize((128, 128), Image.BILINEAR)
    blur = small.filter(ImageFilter.GaussianBlur(radius=1.4))
    diff = np.asarray(small, dtype=np.float32) - np.asarray(blur, dtype=np.float32)
    return float(np.sqrt((diff ** 2).mean()))


def classify(image):
    """Classify a PIL image. Returns :class:`Classification`."""
    rgb = np.asarray(_resize(image).convert('RGB'), dtype=np.uint8)
    h, w, _ = rgb.shape
    feat = {}

    g = image.convert('L').resize((max(1, w), max(1, h)), Image.BILINEAR)
    es = _edge_stats(g)
    feat.update(es)

    feat['palette'] = _quantized_colors(_resize(image))
    feat['noise'] = _noise_level(g)
    feat['colorful'] = _colorfulness(rgb)

    hsv = np.asarray(_resize(image).convert('HSV'), dtype=np.float32)
    hsv_h = hsv[..., 0] / 255.0 * 360.0
    hsv_s = hsv[..., 1] / 255.0
    hsv_v = hsv[..., 2] / 255.0
    feat['sat_mean'] = float(hsv_s.mean())
    feat['val_mean'] = float(hsv_v.mean())
    feat['green_ratio'] = _green_ratio(hsv_h, hsv_s)

    skin = _skin_ratio(rgb)
    feat['skin_ratio'] = skin

    bg_top, bg_kinds = _background_uniformity(rgb)
    feat['bg_uniform'] = bg_top
    feat['bg_kinds'] = bg_kinds

    return _decide(feat)


def _resize(image):
    w, h = image.size
    if max(w, h) <= PREVIEW:
        return image
    r = PREVIEW / max(w, h)
    return image.resize((max(1, int(w * r)), max(1, int(h * r))), Image.BILINEAR)


def _decide(f):
    flat = f['palette'] <= 16
    skin = f['skin_ratio']
    edge_d = f['edge_density']
    colorful = f['colorful']
    sat = f['sat_mean']
    noise = f['noise']
    green = f['green_ratio']
    bg_uni = f['bg_uniform']
    photo = noise >= 1.6 and colorful >= 12  # photographic, not vector/flat

    # 1) Transparent-alpha logos / graphics handled upstream; here we look
    #    for flat-colour high-contrast graphics (low noise + few colours).
    graphic = (flat and noise < 3.0 and edge_d > 0.05)

    # 2) Human: strong skin-tone presence on a photograph.
    if photo and skin > 0.18 and colorful >= 18:
        conf = min(0.99, 0.55 + skin * 1.2)
        return Classification(HUMAN, conf, f)

    # 3) Plant: dominant green on a photo.
    if photo and green > 0.45:
        conf = 0.5 + min(0.45, green)
        return Classification(PLANT, conf, f)

    # 4) Cartoon / illustration: flat, low-noise, saturated, posterised.
    if flat and sat > 0.35 and noise < 2.4 and edge_d > 0.04:
        return Classification(CARTOON, 0.78, f)

    # 5) Document / text: extremely flat (2-6 colours), low saturation.
    if flat and sat < 0.22 and f['palette'] <= 8 and noise < 2.2 and edge_d > 0.03:
        return Classification(DOCUMENT, 0.85, f)

    # 6) Logo / graphic: flat few colours, high-contrast edges.
    if flat and (f['palette'] <= 12) and edge_d > 0.06:
        return Classification(LOGO, 0.72, f)

    # 7) Strong uniform backdrop → product or logo.
    if bg_uni > 0.55 and photo:
        conf = 0.5 + (bg_uni - 0.55)
        return Classification(PRODUCT, min(0.9, conf), f)

    # 8) Photograph of an animal: textured, colourful, non-skin, non-green.
    if photo and colorful >= 14 and edge_d > 0.05:
        return Classification(ANIMAL, 0.6, f)

    # 9) Colourful, moderately flat photo food-ish / general.
    if photo and colorful >= 20:
        return Classification(FOOD, 0.55, f)

    # 10) Not enough evidence → safe fallback.
    return Classification(MIXED, 0.45, f)