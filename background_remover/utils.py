"""Shared numpy / Pillow helpers for the background remover.

Everything is written to work with only ``numpy`` + ``Pillow`` (no OpenCV,
no scipy) so the app keeps running on a bare install.
"""

import numpy as np
from PIL import Image, ImageFilter


def to_pil(a, mode='L'):
    """Convert a numpy (0..255) array to a PIL image."""
    arr = np.clip(np.round(a), 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode=mode)


def to_np(img):
    """Convert a PIL image to a float32 numpy array (0..255)."""
    return np.asarray(img, dtype=np.float32)


def erode(alpha, k):
    """Greyscale morphological erosion (MinFilter), square kernel k."""
    if k <= 1:
        return alpha
    return to_np(to_pil(alpha).filter(ImageFilter.MinFilter(k)))


def dilate(alpha, k):
    """Greyscale morphological dilation (MaxFilter), square kernel k."""
    if k <= 1:
        return alpha
    return to_np(to_pil(alpha).filter(ImageFilter.MaxFilter(k)))


def gaussian(alpha, r):
    """Gaussian blur of an alpha-like array."""
    if r <= 0 or r is None:
        return alpha
    return to_np(to_pil(alpha).filter(ImageFilter.GaussianBlur(radius=r)))


def open_(alpha, k):
    """Erode then dilate — removes small bright specks (noise)."""
    return dilate(erode(alpha, k), k)


def close_(alpha, k):
    """Dilate then erode — fills small dark holes."""
    return erode(dilate(alpha, k), k)


def box_blur(a, r, mode='edge'):
    """Vectorized separable box blur (fast, approximates Gaussian).

    ``a`` must be a 2D float32 array. Blur radius ``r`` applied twice
    (once per axis) with edge padding.

    Implemented with the integral-image (cumsum) trick so cost is O(1)
    per pixel regardless of ``r`` — around 10x faster than the old
    rolling-window loop for radii used by matting / decontamination.
    """
    if r <= 0:
        return a
    k = 2 * r + 1

    a = np.asarray(a, dtype=np.float32)

    def _blur_axis(arr, axis):
        n = arr.shape[axis]
        pad = np.pad(arr, [(r, r) if i == axis else (0, 0)
                           for i in range(arr.ndim)], mode=mode)
        csum = np.cumsum(pad, axis=axis, dtype=np.float64)
        csum = np.concatenate(
            [np.zeros_like(np.take(csum, [0], axis=axis)), csum], axis=axis)
        hi = np.take(csum, np.arange(k, n + k), axis=axis)
        lo = np.take(csum, np.arange(0, n), axis=axis)
        return ((hi - lo) / k).astype(np.float32)

    out = _blur_axis(a, 0)
    out = _blur_axis(out, 1)
    return out


def box_blur_3d(image, r, mode='edge'):
    """Blur an HxWxC float image channel by channel."""
    h, w, c = image.shape
    out = np.empty_like(image, dtype=np.float32)
    for ch in range(c):
        out[..., ch] = box_blur(image[..., ch], r, mode=mode)
    return out


def guided_filter(src, guide, radius=4, eps=1e-3):
    """Edge-aware filter (He et al. guided image filter).

    ``src``   : HxW float32 signal to filter (1 channel).
    ``guide`` : HxW float32 guiding (structure) image.
    ``radius``: local window radius.
    ``eps``   : regularisation — higher = smoother (less edge kept).

    Keeps the edges/structures of ``guide`` while smoothing flat areas, the
    basis for higher-quality matting and background estimation.
    """
    r = max(1, int(radius))
    mean_I = box_blur(guide, r)
    mean_II = box_blur(guide * guide, r)
    mean_p = box_blur(src, r)
    mean_Ip = box_blur(guide * src, r)

    var_I = np.clip(mean_II - mean_I * mean_I, 0.0, None)
    a = (mean_Ip - mean_I * mean_p) / (var_I + float(eps))
    b = mean_p - a * mean_I

    a = box_blur(a, r)
    b = box_blur(b, r)
    return a * guide + b


def bilateral_filter(src, spatial=3, sigma_color=32.0):
    """Edge-preserving bilateral smoothing without scipy.

    Fast approximation: two passes of box-blur following the guided-filter
    decomposition, which is far cheaper than a true O(r^2) bilateral and
    adequate for edge refinement stages.
    """
    r = max(1, int(spatial))
    b = guided_filter(src, src, radius=r, eps=float(sigma_color) ** 2)
    return b


def bounding_box(mask):
    """Return (x0, y0, x1, y1) of the True region or None if empty."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def main_bbox(mask, keep=0.98):
    """Return the bounding box enclosing ~``keep`` of the mask mass."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    # Rank by mass along each axis, keep the densest span.
    order_y = np.argsort(ys)
    order_x = np.argsort(xs)
    n = len(ys)
    y_vals = ys[order_y]
    x_vals = xs[order_x]
    cut = int(n * (1.0 - keep))
    y0, y1 = int(y_vals[cut]), int(y_vals[n - 1 - cut])
    x0, x1 = int(x_vals[cut]), int(x_vals[n - 1 - cut])
    return x0, y0, x1, y1


def resize_cover(img, long_side):
    """Resize an image so the longer side == long_side (LANCZOS)."""
    w, h = img.size
    if max(w, h) <= long_side:
        return img
    ratio = long_side / max(w, h)
    return img.resize((max(1, int(w * ratio)), max(1, int(h * ratio))),
                      Image.LANCZOS)


def grayscale(img):
    if img.mode == 'L':
        return img
    return img.convert('L')


def edge_map(img):
    """0..255 edge-strength image using PIL's FIND_EDGES kernel."""
    g = grayscale(img)
    return g.filter(ImageFilter.FIND_EDGES)


def normalize01(a):
    a = np.asarray(a, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-6:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def dominant_color(img, region_mask=None, k=4):
    """Most frequent colour (list [r, g, b]) in the image / region."""
    arr = np.asarray(img.convert('RGB'), dtype=np.float32).reshape(-1, 3)
    if region_mask is not None:
        m = np.asarray(region_mask, dtype=bool).reshape(-1)
        if m.any():
            arr = arr[m]
    # Simple k-bin histogram per channel merge — good enough for surveys.
    quant = (arr // (256 // k)).astype(np.int32)
    codes = (quant[:, 0] * k + quant[:, 1]) * k + quant[:, 2]
    codes, counts = np.unique(codes, return_counts=True)
    best = codes[int(np.argmax(counts))]
    b, g, r = (best // k) % k, best % k, (best // (k * k))
    center = int((256 // k) / 2)
    return [(r * (256 // k) + center),
            (g * (256 // k) + center),
            (b * (256 // k) + center)]