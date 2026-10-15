"""Mask cleanup — speckle removal and hole filling done *safely*.

The tricky part is not destroying hair/fur/thin-logo-lines. We solve that
by labelling connected components on a decimated copy of the mask and
removing only *isolated* blobs by area. The object boundary and long thin
structures always survive because their area is large.
"""

import numpy as np
from PIL import Image, ImageFilter

# Per-cleanup-level speckle / hole areas (as a fraction of image pixels).
LEVELS = {
    'light':  dict(speck=0.0015, hole=0.0008),   # hair / fur / plant
    'normal': dict(speck=0.0040, hole=0.0015),   # product / food / vehicle
    'strong': dict(speck=0.0100, hole=0.0030),   # document scans
    'sharp':  dict(speck=0.0020, hole=0.0),      # logo — never fill holes
}


def _label(mask):
    """Two-pass connected-component labelling with union-find.

    ``mask``: 2D bool array → returns int label array (0 = background).
    Pure Python, so callers pass a decimated mask.
    """
    rows, cols = mask.shape
    labels = np.zeros((rows, cols), dtype=np.int32)
    parent = [0, 1]  # 1-indexed union-find: parent[lbl] == lbl at a root.

    def find(x):
        p = parent
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:
            p[x], x = root, p[x]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    next_label = 1
    for i in range(rows):
        for j in range(cols):
            if not mask[i, j]:
                continue
            left = labels[i, j - 1] if j > 0 else 0
            up = labels[i - 1, j] if i > 0 else 0
            if left and up:
                lo = left if left < up else up
                labels[i, j] = lo
                union(left, up)
            elif left:
                labels[i, j] = left
            elif up:
                labels[i, j] = up
            else:
                parent.append(next_label)
                labels[i, j] = next_label
                next_label += 1

    # Relabel every slot through its component root.
    for i in range(rows):
        row = labels[i]
        for j in range(cols):
            if row[j]:
                row[j] = find(int(row[j]))
    return labels


def _decimate(mask, long_side=512):
    h, w = mask.shape
    if max(h, w) <= long_side:
        return mask.astype(np.uint8) * 255, (w, h)
    r = long_side / max(h, w)
    nw, nh = max(1, int(w * r)), max(1, int(h * r))
    img = Image.fromarray((mask.astype(np.uint8)) * 255, mode='L')
    img = img.resize((nw, nh), Image.NEAREST)
    return np.asarray(img, dtype=bool), (w, h)


def _restore(small_mask, orig_size):
    w, h = orig_size
    if (small_mask.shape[1], small_mask.shape[0]) == (w, h):
        return small_mask
    img = Image.fromarray((small_mask.astype(np.uint8)) * 255, mode='L')
    return np.asarray(img.resize((w, h), Image.NEAREST), dtype=bool)


def _component_areas(mask):
    """Return (labels, {label: area})."""
    labels = _label(mask)
    counts = np.bincount(labels.ravel())
    areas = {int(i): int(c) for i, c in enumerate(counts) if c}
    return labels, areas


def remove_small_specks(alpha, min_area_ratio=0.004):
    """Zero out floating fg blobs smaller than ``min_area_ratio`` of image."""
    fg = alpha > 128
    if fg.sum() < 4:
        return alpha
    total = fg.size
    dec, osize = _decimate(fg)
    labels, areas = _component_areas(dec)
    keep = np.zeros(dec.shape, dtype=bool)
    thr = max(4, min_area_ratio * dec.size)
    for lbl, area in areas.items():
        if area >= thr:
            keep |= labels == lbl
    small = _restore(keep, osize)
    out = np.where(small, alpha, 0.0).astype(np.float32)
    return _smooth_zeroed(out)


def fill_small_holes(alpha, max_area_ratio=0.0015):
    """Fill background components that are fully surrounded (holes)."""
    fg = alpha > 128
    if not fg.any() or fg.all():
        return alpha
    dec, osize = _decimate(fg)
    bg = ~dec
    labels, areas = _component_areas(bg)
    thr = max(4, max_area_ratio * dec.size)

    edge_labels = set(labels[0, :]) | set(labels[-1, :]) \
        | set(labels[:, 0]) | set(labels[:, -1])

    filled = np.zeros(dec.shape, dtype=bool)
    for lbl, area in areas.items():
        if area == 0 or lbl == 0:
            continue
        if area < thr and lbl not in edge_labels:
            filled |= labels == lbl
    if not filled.any():
        return alpha
    keep = _restore(filled, osize)
    out = np.where(keep, 255.0, alpha).astype(np.float32)
    return _smooth_zeroed(out)


def cleanup(alpha, category=None, strategy=None):
    """Category-aware mask cleanup. Returns updated float alpha (0..255)."""
    cfg = strategy or {}
    level = cfg.get('cleanup', 'normal')
    speck = LEVELS.get(level, LEVELS['normal'])['speck']
    hole = LEVELS.get(level, LEVELS['normal'])['hole']

    alpha = np.asarray(alpha, dtype=np.float32)
    out = alpha.copy()
    if speck > 0 and out.max() > 0:
        out = remove_small_specks(out, speck)
    if hole > 0:
        out = fill_small_holes(out, hole)
    return out


def _smooth_zeroed(alpha):
    """Lightly blur only the freshly-zeroed pixels so removal isn't jaggy."""
    arr = np.asarray(np.clip(alpha, 0, 255), dtype=np.uint8)
    blur = np.asarray(Image.fromarray(arr).filter(
        ImageFilter.GaussianBlur(radius=0.7)), dtype=np.float32)
    # Keep solid pixels, blend the new-zero band toward a faint blur so the
    # cut boundary isn't a staircase.
    solid = alpha > 200
    out = np.where(solid, np.maximum(alpha, blur), alpha)
    return out.astype(np.float32)