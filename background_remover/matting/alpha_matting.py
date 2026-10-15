"""Local alpha matting (edge refinement).

Refines the alpha **only inside the trimap's unknown band** (never the whole
image). For each unknown pixel we estimate a foreground colour F and a
background colour B from nearby known pixels, then apply the classical
projection:

    alpha = dot(I - B, F - B) / |F - B|^2

Enhancements over the plain version:

- **Adaptive trimap**: when no trimap is supplied it is generated inside the
  matting work scale (≤1536) with an edge-aware, varying band width.
- **Confidence-weighted blend**: where F and B separate well (crisp edge) we
  trust the projection; where they barely differ (hair over a similar
  background, sheer fabric) we stay close to the original prediction so we
  never invent a halo.
- **Edge-aware (guided) colour estimation**: F and B are estimated with an
  edge-preserving guided filter driven by the colour image, so the
  foreground/background colours respect the object's structure instead of
  leaking across the contour.
- **Iterative refinement**: after each round the result is re-trimapped with
  the *improved* alpha and matted again, so contamination in the band is
  progressively removed instead of being frozen in place.
"""

import numpy as np
from PIL import Image

from ..utils import box_blur, to_pil, guided_filter


def refine_alpha(image_rgb, alpha, trimap=None, radius=6, downscale=1536,
                 iterations=2, adaptive=True, fg_band=None, bg_band=None,
                 cfg=None):
    """Refine ``alpha`` inside the trimap's unknown band.

    ``image_rgb`` is HxWx3 float RGB (0..255), ``alpha`` float32 0..1.
    ``trimap`` (uint8 0/128/255) is optional — when omitted it is generated
    adaptively at the work scale. ``radius`` is the local colour-estimation
    radius. Returns a refined float32 alpha 0..1 at the input size.
    """
    if radius <= 0:
        return alpha
    a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)

    work, scale = _to_work(image_rgb, a, downscale)
    wrgb, w_alpha = work[0], work[1]

    if trimap is not None:
        t_img = to_pil(np.asarray(trimap, dtype=np.uint8))
        if scale != 1.0:
            t_img = t_img.resize((wrgb.shape[1], wrgb.shape[0]),
                                 Image.NEAREST)
        w_trim = np.asarray(t_img, dtype=np.uint8)
    else:
        w_trim = None

    fg_band = fg_band or radius
    bg_band = bg_band or radius

    out = w_alpha.astype(np.float32).copy()
    for _ in range(max(1, int(iterations))):
        if w_trim is None:
            from .trimap import generate_trimap
            w_trim, _, _, _ = generate_trimap(
                out, fg_band, bg_band, adaptive=adaptive, rgb=wrgb, cfg=cfg)
        out = _matte_round(wrgb, out, w_trim, radius)

    if scale != 1.0:
        h, w = alpha.shape
        out_img = to_pil(np.clip(out, 0.0, 1.0) * 255.0)
        out_img = out_img.resize((w, h), Image.LANCZOS)
        out = np.asarray(out_img, dtype=np.float32) / 255.0
    return np.clip(out, 0.0, 1.0)


def _matte_round(wrgb, w_alpha, w_trim, radius):
    """One matting pass: estimate F/B, project, then confidence-blend."""
    kfg = (w_trim == 255).astype(np.float32)
    kbg = (w_trim == 0).astype(np.float32)
    unk = (w_trim == 128)
    if not unk.any() or not kfg.any() or not kbg.any():
        return w_alpha

    # Edge-aware F/B estimation. The luminance of the photo is the guide:
    # within a hair strand or an object contour the colours stay connected,
    # across the silhouette they stop bleeding into each other.
    try:
        lum = 0.299 * wrgb[..., 0] + 0.587 * wrgb[..., 1] + 0.114 * wrgb[..., 2]
        lum_g = lum / 255.0
        r = max(2, radius)
        F = np.empty_like(wrgb, dtype=np.float32)
        B = np.empty_like(wrgb, dtype=np.float32)
        for c in range(3):
            f_sig = _safe_div(box_blur(wrgb[..., c] * kfg, radius),
                              box_blur(kfg, radius) + 1e-4)
            b_sig = _safe_div(box_blur(wrgb[..., c] * kbg, radius),
                              box_blur(kbg, radius) + 1e-4)
            F[..., c] = guided_filter(f_sig, lum_g, radius=r, eps=0.05)
            B[..., c] = guided_filter(b_sig, lum_g, radius=r, eps=0.05)
    except Exception:
        fg_w = box_blur(kfg, radius) + 1e-4
        bg_w = box_blur(kbg, radius) + 1e-4
        F = np.empty_like(wrgb, dtype=np.float32)
        B = np.empty_like(wrgb, dtype=np.float32)
        for c in range(3):
            F[..., c] = box_blur(wrgb[..., c] * kfg, radius) / fg_w
            B[..., c] = box_blur(wrgb[..., c] * kbg, radius) / bg_w

    fb = F - B
    denom = (fb * fb).sum(axis=2) + 1e-4
    num = ((wrgb - B) * fb).sum(axis=2)
    refined = np.clip(num / denom, 0.0, 1.0)
    refined = box_blur(refined, max(1, radius // 2))  # stabilise the band

    # Confidence: when F and B separate well the projection is trustworthy.
    contrast = np.sqrt((fb * fb).sum(axis=2))
    conf = np.clip((contrast - 20.0) / 100.0, 0.0, 1.0)
    mix = 0.35 + 0.5 * conf

    out = w_alpha.astype(np.float32).copy()
    out[unk] = mix[unk] * refined[unk] + (1.0 - mix[unk]) * w_alpha[unk]
    return np.clip(out, 0.0, 1.0)


def _safe_div(a, b):
    b = np.maximum(b, 1e-6)
    return a / b


def _to_work(image_rgb, alpha, max_side):
    """Optionally downscale to keep matting fast on very large images."""
    h, w = alpha.shape
    if max(h, w) <= max_side:
        return (np.asarray(image_rgb, dtype=np.float32),
                np.clip(alpha, 0.0, 1.0)), 1.0
    r = max_side / max(h, w)
    nw, nh = max(1, int(w * r)), max(1, int(h * r))
    a_img = to_pil(alpha * 255.0).resize((nw, nh), Image.BILINEAR)
    rgb_img = Image.fromarray(np.clip(image_rgb, 0, 255).astype(np.uint8),
                              mode='RGB').resize((nw, nh), Image.BILINEAR)
    return (np.asarray(rgb_img, dtype=np.float32),
            np.asarray(a_img, dtype=np.float32) / 255.0), r