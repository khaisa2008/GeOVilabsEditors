"""Alpha–colour spill correction.

The hardest halo case is a pixel whose *colour* says "this is ~30% object"
but whose predicted alpha says "this is ~80% object". Colour un-mixing alone
can't fix it (it trusts the alpha). We instead re-project the alpha from
colour evidence:

    a_implied = dot(I - B, F - B) / |F - B|^2

where F is a *robust* foreground colour sampled from the confident interior
(eroded to avoid boundary contamination) and B a robust background colour
from confident background pixels. Where the implied alpha is clearly lower
than the prediction, the edge is background-dominated (spill) → we pull the
alpha toward the implied value and re-run decontamination at the corrected
alpha, which then produces the true object colour (not white).

Genuine hair/fur is untouched: strand pixels whose colour is consistent with
their alpha give a_implied ≈ a_predicted.

Enhancements over the original:

- **Multi-sample foreground** — F is sampled with several EROSION radii and
  averaged, so a single contaminated rim sample cannot skew the estimate.
- **Lab-aware evidence** — the F/B separation evidence is judged in the
  perceptually uniform Lab space, making the pull much more reliable on
  skin tones and other colours where RGB distance misleads.
"""

import numpy as np

from ..utils import box_blur, erode
from ..matting.edge_detection import edge_zone
from ..color_ops import lab_distance


def fg_estimate(rgb, mask, radius=6):
    """Per-channel blurred colour where ``mask`` is set (0..255 RGB)."""
    m = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    if m.sum() <= 0:
        return np.zeros_like(rgb, dtype=np.float32)
    w = box_blur(m, radius) + 1e-4
    rgb = np.asarray(rgb, dtype=np.float32)
    out = np.empty_like(rgb, dtype=np.float32)
    for c in range(3):
        out[..., c] = box_blur(rgb[..., c] * m, radius) / w
    return out


def _robust_fg(rgb, a):
    """Foreground colour estimate robust to rim contamination: averaged over
    several erosion radii so one bad sample cannot dominate."""
    masks = []
    base = (a > 0.9).astype(np.float32)
    if base.mean() > 0.01:
        for k in (3, 5, 7):
            m = (erode(base * 255.0, k) > 0).astype(np.float32)
            if m.mean() > 0.0:
                masks.append(m)
    if not masks:
        if base.mean() > 0.0:
            masks.append(base)
        else:
            masks.append((a > 0.9).astype(np.float32))
    acc = None
    for m in masks:
        f = fg_estimate(rgb, m)
        acc = f if acc is None else acc + f
    return acc / len(masks)


def correct_spill_alpha(image_rgb, alpha, zone=None, radius=7, strength=0.85,
                        min_a=0.15):
    """Return a corrected float32 alpha (0..1).

    Only pixels inside the boundary zone with a confident background sample
    within reach are considered; interiors and solid background are left
    untouched. Alpha never increases here — spill suppression is a *lowering*
    operation, strand preservation is handled elsewhere.
    """
    a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    rgb = np.asarray(image_rgb, dtype=np.float32)
    if zone is None:
        zone = edge_zone(a, thresh=0.03, dilation=4)

    # Robust foreground colour: confident interior, morphed inwards so the
    # bright contaminated rim never leaks into the estimate.
    fg_mask = _robust_fg(rgb, a)
    F = fg_mask

    # Robust background colour from confident background pixels.
    bg_mask = (a < 0.35).astype(np.float32)
    B = fg_estimate(rgb, bg_mask)

    fb = F - B
    denom = (fb * fb).sum(axis=2) + 1e-4
    num = ((rgb - B) * fb).sum(axis=2)
    a_imp = np.clip(num / denom, 0.0, 1.0)

    # Evidence of F/B separation judged in Lab space (perceptually reliable
    # for skin, fabric, etc. where an RGB gap can mislead).
    evidence_rgb = np.sqrt((fb * fb).sum(axis=2))
    ev_n = np.clip((evidence_rgb - 30.0) / 90.0, 0.0, 1.0)
    try:
        lab_d = lab_distance(rgb, F)
        lab_ev = np.clip((lab_d - 40.0) / 140.0, 0.0, 1.0)
        ev_n = np.maximum(ev_n, lab_ev)
    except Exception:
        pass

    # Pull alpha DOWN toward the colour-implied value wherever the prediction
    # is clearly too high — continuous (not binary) so marginal pixels get a
    # partial correction. Only where a real background sample is within reach.
    support = box_blur(bg_mask, radius) > 0.02
    # Never touch the confident body: halo pixels are background-dominated, so
    # their colour-implied alpha sits well below the prediction. The gate must
    # reach a == 1.0 because a hard / binary mask leaves the anti-aliased
    # boundary pixel fully opaque while its colour still says "background".
    # Genuine edges keep a_imp ≈ a and drop out of the condition.
    cond = zone & (a > min_a) & support & (a_imp < a - 0.08)
    if not cond.any():
        return a

    pull = np.clip((a - a_imp) / 0.25, 0.0, 1.0)
    w = strength * ev_n * pull * 0.8
    out = a.copy()
    out[cond] = a[cond] * (1.0 - w[cond]) + a_imp[cond] * w[cond]
    # Never raise alpha in this stage.
    out = np.minimum(out, a)
    return np.clip(out, 0.0, 1.0)