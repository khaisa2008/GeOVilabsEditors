"""Foreground colour decontamination.

When a foreground edge is semi-transparent the displayed colour is a mix of
the true foreground colour F and the background colour B:

    I = a * F + (1 - a) * B            →   F = (I - (1 - a) * B) / a

After background removal we want F. Given a local estimate of the background
colour field ``B`` we can un-mix. This module:

- applies the un-mix **only** where the alpha is meaningful (>= ``min_a``) so
  transparent pixels never blow up (requirement 6 / 7),
- **iterates** so the background field keeps improving as the mask is
  refined,
- weights its strength **edge-aware** so the body stays untouched while the
  fringe is fully de-contaminated,
- keeps the high-confidence body (``a > 0.98``) pixel-identical.

Enhancements:

- **Guided-filter background estimation** — the background field is estimated
  with an edge-aware filter (guided by the colour image) so it stops leaking
  across object edges, which is what left colour fringes before.
- **Lab-space chroma correction** — after RGB un-mixing, a chroma correction
  in the perceptually uniform Lab space removes residual colour spill while
  leaving luminance (hence hair-strand opacity) intact.
"""

import numpy as np

from ..utils import box_blur_3d, box_blur, guided_filter


def estimate_bg_field(image_rgb, alpha, radius=8, edge_weight=None,
                      use_guided=True):
    """Weighted-average background colour near every pixel.

    With ``use_guided`` (default) the estimate is filtered edge-aware using
    the colour image as guide, so the background estimate is accurate right
    up to the object contour — this is the key to clean decontamination.
    """
    a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    w = (1.0 - a)
    if edge_weight is not None:
        w = w * (1.0 + edge_weight)
    bg_w = box_blur(w, radius) + 1e-4
    B = box_blur_3d(np.asarray(image_rgb, dtype=np.float32)
                    * w[..., None], radius) / bg_w[..., None]
    rgb = np.clip(B, 0.0, 255.0)
    if use_guided and rgb.shape[0] > 4 and rgb.shape[1] > 4:
        lum = np.clip(image_rgb[..., 0] * 0.299 + image_rgb[..., 1] * 0.587
                      + image_rgb[..., 2] * 0.114, 0.0, 255.0)
        gr = max(2, radius)
        out = np.empty_like(rgb, dtype=np.float32)
        for c in range(3):
            out[..., c] = guided_filter(rgb[..., c], lum / 255.0,
                                        radius=gr, eps=0.02)
        return np.clip(out, 0.0, 255.0)
    return np.clip(B, 0.0, 255.0)


def decontaminate(image_rgb, alpha, strength=0.6, radius=8, min_a=0.12,
                  iterations=2, edge_field=None):
    """Return a decontaminated float RGB copy (0..255)."""
    rgb = np.asarray(image_rgb, dtype=np.float32).copy()
    try:
        from ..color_ops import rgb_to_lab, lab_to_rgb
    except Exception:
        rgb_to_lab = lab_to_rgb = None

    # Adaptive min-alpha: hair/fur benefit from a lower floor so thin strands
    # also get un-mixed; hard products can use the higher default.
    a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    if (a > 0.03).sum() and (a < 0.15).sum():
        edge_frac = float(((a > 0.03) & (a < 0.15)).mean())
        if edge_frac > 0.05:
            min_a = min(min_a, 0.06)
    band = (a >= min_a) & (a < 0.98)

    if edge_field is not None and edge_field.shape == a.shape:
        band = band & (edge_field > 0.02)

    iters = max(1, int(iterations))
    B = None
    for _ in range(iters):
        B = estimate_bg_field(rgb, a, radius)
        a_safe = np.maximum(a, min_a)[..., None]
        F = (rgb - (1.0 - a)[..., None] * B) / a_safe
        F = np.clip(F, 0.0, 255.0)

        strength_arr = strength * band.astype(np.float32)
        out = rgb * (1.0 - strength_arr[..., None]) \
            + F * strength_arr[..., None]
        body = a > 0.98
        out = np.where(body[..., None], rgb, out)
        rgb = np.clip(out, 0.0, 255.0).astype(np.float32)

    # Lab-space chroma correction stage — kills colour spill that survives
    # RGB-domain un-mixing. Only chrominance (a*, b*) is pulled toward the
    # recovered foreground; luminance is preserved so hair stays honest.
    # Reuses the background field from the last iteration instead of
    # re-estimating it (a full guided estimate costs ~1s on large images).
    if rgb_to_lab is not None and B is not None:
        try:
            lab = rgb_to_lab(np.clip(rgb, 0, 255).astype(np.uint8))
            src_lab = rgb_to_lab(np.clip(np.asarray(image_rgb, dtype=np.float32),
                                         0, 255).astype(np.uint8))
            B2 = B
            lum_lab = rgb_to_lab(np.clip(B2, 0, 255).astype(np.uint8))
            La, Aa, Ba = lab[..., 0], lab[..., 1], lab[..., 2]
            Ls, As, Bs = src_lab[..., 0], src_lab[..., 1], src_lab[..., 2]
            Lb, Ab, Bb = lum_lab[..., 0], lum_lab[..., 1], lum_lab[..., 2]
            w2 = strength * 0.6 * band.astype(np.float32)
            A_corr = Aa + (As - Aa) * w2
            B_corr = Ba + (Bs - Ba) * w2
            lab2 = np.stack([La, A_corr, B_corr], axis=-1)
            rgb_corr = lab_to_rgb(lab2)
            rgb = rgb * (1.0 - w2[..., None]) + rgb_corr * w2[..., None]
            rgb = np.clip(rgb, 0.0, 255.0).astype(np.float32)
        except Exception:
            pass

    return np.clip(rgb, 0.0, 255.0).astype(np.float32)


def decontaminate_fringe(image_rgb, alpha, fringe, strength=0.75, radius=6):
    """Decontaminate only a given fringe mask, keeping everything else."""
    a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    rgb = np.asarray(image_rgb, dtype=np.float32).copy()
    fringe = np.asarray(fringe, dtype=bool)
    if not fringe.any():
        return rgb
    B = estimate_bg_field(rgb, a, radius)
    a_safe = np.maximum(a, 0.12)[..., None]
    F = (rgb - (1.0 - a)[..., None] * B) / a_safe
    F = np.clip(F, 0.0, 255.0)
    wgt = (strength * fringe.astype(np.float32))[..., None]
    rgb = rgb * (1.0 - wgt) + F * wgt
    return np.clip(rgb, 0.0, 255.0).astype(np.float32)


def _lum(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]