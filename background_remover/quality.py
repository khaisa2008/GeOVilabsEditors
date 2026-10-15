"""Quality scoring for segmentation / matting results.

Computes a 0..1 score from cheap, deterministic metrics:

- coverage          : foreground compactness inside the frame
- boundary_quality  : how cleanly foreground and background separate at
                      the boundary (colour contrast weighted by alpha)
- isolated_pixels   : speckle ratio (before cleanup)
- boundary_smooth   : compactness of the boundary (perimeter vs area)
- foreground_conf   : average alpha of the confident interior
- halo              : 0..1 severity of white/grey/colour fringe around the
                      edge (spill ratio in the semi-transparent band)

Used to decide whether the pipeline should retry with a stronger strategy
and whether an extra halo-refinement round is still needed.
"""

import numpy as np

from .utils import gaussian, to_pil, box_blur
from .postprocess.decontamination import estimate_bg_field, _lum
from .postprocess.spill_correction import fg_estimate
from .matting.edge_detection import edge_zone


def score(alpha, image_rgb, category='mixed'):
    """alpha: float32 0..1; image_rgb: HxWx3 float 0..255."""
    a = np.clip(alpha, 0.0, 1.0)
    if not a.any():
        return 0.0, {'coverage': 0, 'boundary': 0, 'isolated': 1,
                     'smooth': 0, 'confidence': 0, 'halo': 1.0}

    fg = a > 0.5
    total = a.size

    coverage = float(fg.mean()) if total else 0.0
    # Bias: an object should occupy a reasonable middle range, not a sliver.
    coverage_score = np.clip(1.0 - abs(coverage - 0.55) / 0.55, 0.0, 1.0)

    # Boundary quality: sample a band around the mask edge, compare the
    # average colour outside vs inside. Strong local separation = clean cut.
    band = (a > 0.1) & (a < 0.9)
    if band.any():
        inside = (a > 0.7)
        outside = (a <= 0.35)
        if inside.any() and outside.any():
            in_c = image_rgb[inside].mean(axis=0)
            out_c = image_rgb[outside].mean(axis=0)
            diff = float(np.sqrt(((in_c - out_c) ** 2).sum())) / 360.0
            boundary = float(np.clip(diff, 0.0, 1.0))
        else:
            boundary = 0.5
    else:
        boundary = 1.0

    # Isolated-pixel ratio: pixels of fg surrounded by bg (rough speckle).
    iso = _isolated_ratio(fg)

    # Boundary smoothness: compactness edge/(edge min) proxy. (Cheap)
    smooth = _edge_smoothness(a)

    # Edge alignment: how well the alpha contour follows the true colour
    # edges. A clean cut sits exactly on the silhouette; a floating mask
    # (halo, wrong matte) drifts into flat regions and scores low.
    alignment = edge_alignment(image_rgb, a)

    # Foreground confidence: interior mean alpha.
    interior = gaussian(a, 6.0) > 0.8
    confidence = float(a[interior].mean()) if interior.any() \
        else float(a[fg].mean())

    # Halo severity: fraction of the transition band that still looks like
    # background spill (light fringes on dark edges etc.).
    halo = halo_score(image_rgb, a)

    q = (coverage_score * 0.25 + boundary * 0.25 + alignment * 0.1 +
         (1.0 - iso) * 0.18 + smooth * 0.12 + confidence * 0.1) * (1.0 - halo * 0.3)
    details = {
        'coverage': round(coverage, 4),
        'boundary_quality': round(boundary, 4),
        'edge_alignment': round(float(alignment), 4),
        'isolated_ratio': round(iso, 4),
        'smoothness': round(smooth, 4),
        'foreground_confidence': round(float(confidence), 4),
        'hair_fineness': round(float(hair_fineness(a, image_rgb)), 4),
        'halo': round(float(halo), 4),
    }
    return float(np.clip(q, 0.0, 1.0)), details


def halo_score(image_rgb, alpha, lo=0.08, hi=0.95, work_cap=1024):
    """0..1 spill/halo severity in the semi-transparent edge band.

    0 → clean transition, 1 → a heavy white/grey/colour fringe. Used both
    inside ``score`` and by the pipeline to trigger an extra halo pass.

    Scoring is statistical (fractions of the transition band), so it runs on
    a downscaled working copy for large images — the full-resolution guided
    estimate costs seconds per call and adds nothing to the ratio.
    """
    a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    rgb = np.asarray(image_rgb, dtype=np.float32)
    if max(rgb.shape[0], rgb.shape[1]) > work_cap:
        s = work_cap / max(rgb.shape[0], rgb.shape[1])
        nw, nh = max(1, int(rgb.shape[1] * s)), max(1, int(rgb.shape[0] * s))
        from .utils import to_pil
        from PIL import Image
        a = np.asarray(to_pil(a * 255.0).resize((nw, nh), Image.BILINEAR),
                       dtype=np.float32) / 255.0
        rgb = np.asarray(
            Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), 'RGB')
            .resize((nw, nh), Image.BILINEAR), dtype=np.float32)
    band = (a > lo) & (a < hi)
    if not band.any():
        return 0.0
    if band[band].size < 64:
        return 0.0

    # Boundary-confine: only the contour area can harbour a halo; the object
    # interior must never count as spill.
    band &= edge_zone(a, thresh=0.03, dilation=3)
    if not band.any():
        return 0.0

    B = estimate_bg_field(rgb, a, radius=6)
    a_safe = np.maximum(a, 1e-3)[..., None]
    F = (rgb - (1.0 - a)[..., None] * B) / a_safe
    diff = np.sqrt(((F - B) ** 2).sum(axis=2))

    # Background-dominated spill: recovered fg almost equals the local bg.
    spill = diff < 40.0

    # White/grey fringe: pixel substantially whitened relative to the local
    # solid foreground (a coloured object rim aurally washed toward white), or
    # a clearly achromatic bright pixel against a darker subject.
    lum = np.clip(_lum(rgb), 0.0, 255.0)
    from .utils import erode as _erode
    kfg = (a > 0.7).astype(np.float32)
    if kfg.mean() > 0.01:
        kfg = (_erode(kfg * 255.0, 3) > 0).astype(np.float32)
    fw = box_blur(kfg, 6) + 1e-4
    fg_lum = box_blur(lum * kfg, 6) / fw
    uplift = (lum - fg_lum) / np.maximum(255.0 - fg_lum, 1.0)   # 1 = white
    mx = np.maximum(np.abs(rgb[..., 0] - rgb[..., 1]),
                    np.abs(rgb[..., 1] - rgb[..., 2]))
    grey = mx < 35.0
    gap = lum - fg_lum
    # A real fringe is *clearly* brighter than the solid foreground near it —
    # a modest +20 lum against an already-bright region is rim light, not halo.
    lum_halo = (gap > 45.0) & ((uplift > 0.35) | grey)

    # Alpha–colour inconsistency: pixels whose colour implies a much lower
    # alpha than the mask predicts are background-dominated rims (the colour
    # of many "fine white fibres" on a previously bad cut).
    incoh = np.zeros(a.shape, dtype=bool)
    if (a > 0.9).sum() > 32 and (a < 0.35).sum() > 32:
        F = fg_estimate(rgb, _fg_mask_for_score(a))
        B4 = estimate_bg_field(rgb, a, radius=6)
        fb = F - B4
        denom = (fb * fb).sum(axis=2) + 1e-4
        a_imp = np.clip(((rgb - B4) * fb).sum(axis=2) / denom, 0.0, 1.0)
        incoh = band & (a_imp < a - 0.3)

    bad = (spill | lum_halo | incoh) & band
    n = int(band.sum())
    if n <= 0:
        return 0.0
    # Fraction of the *transition band* that is contaminated — this is what
    # drives the pipeline's "extra halo round" gate.
    return float(np.clip(bad.sum() / n, 0.0, 1.0))


def edge_alignment(image_rgb, alpha, lo=0.03):
    """0..1 how well the alpha contour follows the true colour edges.

    An SSIM-style *structural* comparison: the normalised cross-correlation
    between the alpha-gradient magnitude and the colour edge magnitude inside
    the boundary band. A clean cut peaks exactly on the silhouette (→ near 1);
    a mask that floats in a flat interior or halos outward drifts from the
    colour edges (→ near 0.5 and below).
    """
    from .matting.edge_detection import alpha_gradient, color_edge_map
    a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    rgb = np.asarray(image_rgb, dtype=np.float32)
    ag = alpha_gradient(a, sigma=1.5)
    cg = color_edge_map(rgb, sigma=1.0)
    band = ag > lo
    if not band.any():
        return 1.0
    a_s = ag[band]
    c_s = cg[band]
    if a_s.std() < 1e-6 or c_s.std() < 1e-6:
        return 0.5
    corr = float((((a_s - a_s.mean()) * (c_s - c_s.mean())).mean())
                 / (a_s.std() * c_s.std() + 1e-6))
    return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))


def hair_fineness(alpha, image_rgb, lo=0.05, hi=0.95):
    """0..1 fraction of the transition band that is fine, filamentary detail.

    Hair / fur / feathers are high-frequency and semi-transparent; a good
    matting result keeps them. Measures the share of boundary pixels that sit
    on strong local high-frequency colour detail. For a clean logo this is
    near 0, for a portrait of hair it is meaningfully > 0 — purely a
    diagnostic, not part of the weighted score.
    """
    a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    rgb = np.asarray(image_rgb, dtype=np.float32)
    band = (a > lo) & (a < hi)
    n = int(band.sum())
    if n < 64:
        return 0.0
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    gy, gx = np.gradient(lum)
    mag = np.abs(gy) + np.abs(gx)
    from .utils import gaussian as _g
    mag_c = mag - _g(mag, 3.0)          # high-frequency (strand) component
    thr = float(np.percentile(mag_c[band], 65))
    if thr <= 0:
        return 0.0
    fine = int((mag_c[band] > thr).sum())
    return float(np.clip(fine / n, 0.0, 1.0))


def should_retry(score_value, passes):
    """Low score + few passes → try a stronger pipeline pass."""
    if passes >= 2:
        return False
    return score_value < 0.55


def halo_pass_needed(halo, threshold=0.35, passes=1, max_halo_passes=2):
    """Whether an extra halo-refinement round is warranted."""
    if passes >= max_halo_passes:
        return False
    return halo >= threshold


def _fg_mask_for_score(a):
    """Confident foreground mask, eroded so the rim doesn't leak in."""
    from .utils import erode
    m = (a > 0.9).astype(np.float32)
    if m.mean() > 0.01:
        m = (erode(m * 255.0, 5) > 0).astype(np.float32)
    if m.mean() <= 0.0:
        m = (a > 0.9).astype(np.float32)
    return m


def _isolated_ratio(fg):
    if not fg.any():
        return 1.0
    from PIL import Image, ImageFilter
    f = (fg.astype(np.uint8)) * 255
    small = Image.fromarray(f).resize((128, 128), Image.NEAREST)
    opened = small.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    opened = np.asarray(opened, dtype=bool) > 0
    orig = (np.asarray(small) > 0)
    removed = orig & ~opened
    return float(removed.mean())


def _edge_smoothness(a):
    """Compactness: 4*pi*area/perimeter^2 for the coarse boundary."""
    fg = a > 0.5
    if not fg.any():
        return 0.0
    from .utils import dilate, erode
    bound = np.logical_xor(fg, dilate(erode(fg.astype(np.float32), 3), 3) > 0.5)
    perim = float(bound.sum())
    area = float(fg.sum())
    if perim <= 0 or area <= 0:
        return 0.5
    compact = (4.0 * np.pi * area) / (perim * perim + 1e-6)
    return float(np.clip(compact, 0.0, 1.0))