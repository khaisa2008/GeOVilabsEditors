"""Multi-stage halo removal (white / grey / colour fringes).

Combines foreground decontamination with adaptive spill suppression and a
dedicated white/grey luminance-halo pass. This is the primary defensive
layer against the classic "white rim" on hair, fur, clothing and complex
edges.

Stages:

1. **Decontamination** — iterative, edge-aware foreground colour un-mixing
   (removes background colour clinging to semi-transparent edge pixels).
2. **Adaptive defringe** — for each semi-transparent pixel in the *boundary
   zone*, un-mix the raw RGB and compare with the local background. If the
   recovered foreground barely differs from the background the pixel is
   bg-dominated (spill) → alpha is pulled down; if it clearly contrasts (a
   real object edge such as hair or a hard-surface contour) the alpha is
   kept. Thresholds derive from the local band distribution (adaptive) and
   the whole operation is confined to the contour zone so the object
   INTERIOR is never mistaken for spill.
3. **Luminance halo suppression** — bright, low-chroma fringe pixels that
   are noticeably lighter than the surrounding foreground are exactly the
   white halo users complain about; they get a gentler alpha pull. Genuinely
   white objects (white dress, snow, white fur) have no lightness gap →
   untouched.

Logo / cartoon / document categories keep their hard contours: they receive
a lighter rim decontamination and defringe but never the spatial feathering.

This is deliberately conservative: alpha is only ever *reduced* inside the
contour zone, the object body and genuinely-white subjects are never eroded.
"""

import numpy as np
from PIL import Image

from ..matting.edge_detection import edge_zone
from ..utils import box_blur, erode, to_pil
from .decontamination import decontaminate, estimate_bg_field

# Decontamination / halo work cap. The pipeline's guided-filter background
# estimation is the most expensive numpy stage: a 3000x4000 photo at full
# resolution costs ~10s per estimate. Running the whole halo chain on a
# capped working copy (radii scaled to match) is ~5x faster with negligible
# quality difference, because the analysis windows are relative.
_DECONT_CAP = 1536


def remove_halo(image_rgb, alpha, strategy):
    """Return (decontaminated_rgb, refined_alpha) for the given strategy."""
    cfg = strategy or {}

    rgb = np.asarray(image_rgb, dtype=np.float32)
    alpha_f = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 255.0) / 255.0

    h, w = alpha_f.shape
    if max(h, w) > _DECONT_CAP:
        s = _DECONT_CAP / max(h, w)
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        rgb_w = np.asarray(
            Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), 'RGB')
            .resize((nw, nh), Image.BILINEAR), dtype=np.float32)
        a_w = np.asarray(to_pil(alpha_f * 255.0).resize((nw, nh),
                                                        Image.BILINEAR),
                         dtype=np.float32) / 255.0
        radius_w = max(1, int(round((int(cfg.get('spill_radius', 6) or 6))
                                    * s)))
        rgb_w, a_w = _chain(rgb_w, a_w, cfg, radius_w)
        rgb_full = np.asarray(
            Image.fromarray(np.clip(rgb_w, 0, 255).astype(np.uint8), 'RGB')
            .resize((w, h), Image.BILINEAR), dtype=np.float32)
        a_full = np.asarray(to_pil(a_w * 255.0).resize((w, h), Image.LANCZOS),
                            dtype=np.float32) / 255.0
        return rgb_full, a_full

    return _chain(rgb, alpha_f, cfg,
                  int(cfg.get('spill_radius', 6) or 6))


def _chain(rgb, alpha_f, cfg, radius):
    """Run the halo stages on a single (possibly downscaled) working copy."""
    strength = float(cfg.get('halo_strength', 0.6))
    iterations = int(cfg.get('decontam_iterations', 2) or 2)
    if not cfg.get('halo', True):
        strength *= 0.5

    rgb_out = rgb.astype(np.float32).copy()

    if strength > 0 and (alpha_f > 0.03).any() and (alpha_f < 0.98).any():
        zone = edge_zone(alpha_f, thresh=0.03, dilation=4)
        # Hair / fur residue lives DEEP inside the strand mass, not just at the
        # 4px contour — give the decontamination / spill passes a wider reach
        # there so the background colour between strands is actually un-mixed
        # (the tight zone kept it bright grey before).
        wide_dilate = int(cfg.get('decontam_zone', 4) or 4)
        work_zone = edge_zone(alpha_f, thresh=0.03, dilation=wide_dilate)
        # Hair / fur: decontaminate *every* semi-transparent pixel (gaps
        # between strands can sit far from the contour). It self-regulates —
        # where the local background estimate equals the foreground the un-mix
        # is a no-op, so solid interiors are never altered.
        decont_field = None if cfg.get('decontam_all') \
            else work_zone.astype(np.float32)

        # Stage 1 — decontaminate the fringe (confined to the contour zone).
        if cfg.get('halo', True):
            rgb_out = decontaminate(rgb_out, alpha_f, strength=strength,
                                    radius=radius, iterations=iterations,
                                    edge_field=decont_field)

        # Stage 2 — adaptive spill defringe (alpha pull on bg-dominated rim).
        # Must see the RAW (pre-decontamination) colours: it un-mixes the
        # observed mix I = a·F + (1−a)·B, which decontamination already
        # removed — feeding it the cleaned copy would double-unmix.
        alpha_f = defringe_alpha(rgb, alpha_f, strength=strength,
                                 radius=radius, zone=work_zone,
                                 adaptive=cfg.get('adaptive_defringe', True))

        # Stage 2b — alpha←colour spill correction (only when configured).
        if cfg.get('correct_spill'):
            from .spill_correction import correct_spill_alpha
            alpha_f = correct_spill_alpha(rgb_out, alpha_f, zone=work_zone,
                                          radius=radius,
                                          strength=strength * 1.1)

        # Stage 3 — white / grey luminance halo pull (tight contour only so
        # genuine highlights inside the subject are never shaved).
        alpha_f = suppress_luminance_halo(rgb_out, alpha_f,
                                          strength=strength, radius=radius,
                                          zone=zone)

    return rgb_out, alpha_f


def defringe_alpha(image_rgb, alpha, strength=0.7, radius=6, min_a=0.06,
                   adaptive=True, zone=None):
    """Lower alpha of spill-fringe pixels (background-colour rim)."""
    a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    rgb = np.asarray(image_rgb, dtype=np.float32)
    # Include the solid shell (a == 1): a hard / binary mask leaves the
    # anti-aliased boundary pixel fully opaque but still carrying the mixed
    # background colour — the coloured outline. Its |F−B| evidence decides.
    band = (a > min_a)
    if zone is not None:
        band &= zone
    if not band.any():
        return a

    B = estimate_bg_field(rgb, a, radius)
    a_safe = np.maximum(a, 1e-3)[..., None]
    F = (rgb - (1.0 - a)[..., None] * B) / a_safe
    diff = np.sqrt(((F - B) ** 2).sum(axis=2))

    # Multi-radius support check: a real background must be within reach at
    # several neighbourhood sizes, otherwise the "background" estimate is
    # actually the object's own interior and we must not pull alpha.
    support = np.zeros_like(a, dtype=np.float32)
    for r in (max(2, radius // 2), radius, min(radius * 2, 14)):
        support = np.maximum(
            support, (box_blur((a < 0.4).astype(np.float32), r) > 0.05))
    band = band & support.astype(bool)

    if adaptive:
        lo, hi = _adaptive_thresholds(diff, band)
    else:
        lo, hi = 25.0, 125.0
    conf = np.clip((diff - lo) / (hi - lo), 0.0, 1.0)
    scale = 1.0 - strength * (1.0 - conf)
    out = a.copy()
    out[band] = a[band] * scale[band]
    return out


def suppress_luminance_halo(rgb, alpha, strength=0.6, radius=6, min_a=0.1,
                            zone=None):
    """Pull alpha down on bright low-chroma fringe (the white halo).

    ``rgb`` should already be decontaminated so the comparison is against the
    recovered foreground colour, not the raw mixed colour.

    The reference foreground luminance is measured from the **solid interior
    only** (alpha ~1, eroded away from the rim). Measuring it from the whole
    alpha>0.85 population lets a wide semi-transparent grey wash count as
    "foreground" and cancel its own brightness gap — exactly the detached
    halo ring you see around upscaled cutouts.
    """
    a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    rgb = np.asarray(rgb, dtype=np.float32)
    band = (a > min_a) & (a < 0.985)
    if zone is not None:
        band &= zone
    if not band.any() or not (a > 0.9).any():
        return a

    lum = np.clip(_lum(rgb), 0.0, 255.0)

    # Solid interior reference, eroded so the wash / hair rim can never pollute
    # the "how bright is the subject" measurement.
    solid = (a >= 0.985).astype(np.float32)
    interior = box_blur(solid, max(1, radius // 2)) > 0.999
    k_erode = max(3, 2 * (radius // 2) + 1)
    interior = erode((interior.astype(np.uint8)) * 255, k_erode) > 0
    if not interior.any():
        interior = a > 0.9
    interior_f = interior.astype(np.float32)

    fw = box_blur(interior_f, radius) + 1e-4
    fg_lum = _blur(lum * interior_f, radius) / fw

    lift = lum - fg_lum                            # how much brighter than local fg

    # Adaptive interpretation of "bright fringe": the pull threshold scales
    # with the local foreground luminance, so a genuinely bright subject is
    # judged relative to its own brightness instead of a fixed 10-60 band.
    fg_lum_lo = np.percentile(fg_lum[interior], 20) if interior.any() else 40.0
    base = np.clip(float(fg_lum_lo) * 0.12, 6.0, 20.0)
    lift_n = np.clip((lift - base) / (base * 5.0), 0.0, 1.0)

    mx = np.maximum.reduce([  # chroma-ish: |r-g| (0=neutral)
        np.abs(rgb[..., 0] - rgb[..., 1]),
        np.abs(rgb[..., 1] - rgb[..., 2]),
    ])
    grey = 1.0 - np.clip(mx / 70.0, 0.0, 1.0)     # 1 = grey/white fringe

    # Chroma gate: only genuinely *neutral* halo gets suppressed. Bright,
    # warm hair strands (chroma ~40-80) and coloured spills must survive;
    # the white/background-coloured skirt is near-neutral (chroma < 10).
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    chroma_w = np.clip((40.0 - chroma) / 25.0, 0.0, 1.0)

    white_halo = lift_n * grey * chroma_w * band.astype(np.float32)
    scale = 1.0 - strength * 0.9 * white_halo
    out = np.clip(a * scale, 0.0, 1.0)
    # A *confirmed* skirt (pixel is massively brighter than the surrounding
    # solid subject AND near-neutral) has no foreground content — delete it
    # instead of leaving a faint haze.
    hard = white_halo > 0.8
    if hard.any():
        out[hard] = 0.0
    return out


def _adaptive_thresholds(diff, band, lo_pct=18, hi_pct=62, hard_lo=8.0,
                         hard_hi=220.0):
    """Per-image spill thresholds from the |F−B| distribution in the band."""
    d = diff[band]
    if d.size < 32:
        return 25.0, 125.0
    lo = float(np.percentile(d, lo_pct))
    hi = float(np.percentile(d, hi_pct))
    lo = float(np.clip(lo, hard_lo, hard_hi))
    hi = float(np.clip(hi, lo + 20.0, hard_hi))
    return lo, hi


def _blur(a, r):
    return box_blur(np.asarray(a, dtype=np.float32), r)


def _lum(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def trim_detached_fringe(alpha_f, floor=0.65, reach=6, min_a=0.03):
    """Zero semi-transparent fragments far from the solid body.

    The halo chain's upscale (and the model itself) can leave a *detached*
    skirt / ring around the silhouette — pixels that have alpha but are
    separated from the object by a transparent gap. These have no business in
    a cutout. Connected low-alpha hair and fur survive because they stay
    within ``reach`` of a >=0.5 pixel.
    """
    a = np.clip(np.asarray(alpha_f, dtype=np.float32), 0.0, 1.0)
    if not (a > min_a).any():
        return a
    strong = (a >= 0.5).astype(np.float32)
    near = box_blur(strong, reach) > 0.0
    drop = (a > min_a) & (a < floor) & (~near)
    if not drop.any():
        return a
    out = a.copy()
    out[drop] = 0.0
    return out


def deskirt_fullres(image_rgb, alpha, cfg, radius=8, min_a=0.04):
    """Full-resolution skirt trim for hair / fur cutouts.

    ``image_rgb`` is the (already decontaminated) full-res float RGB, ``alpha``
    a float 0..1 at the same size. The halo chain runs on a downscaled working
    copy and its final upscale *remixes* edge pixels back toward the
    background colour, regrowing a faded band on the silhouette. Three
    full-res passes fix that:

    1. **Colour un-mix at the real size** — semi-transparent pixels recover
       their true foreground colour against the local background field, so a
       hair edge becomes hair-coloured instead of wall-coloured.
    2. **Background-colour match** — whatever is *still* background-coloured
       after un-mixing (the detached ring, pure-wall gaps inside the hair) is
       deleted outright, no matter how faint its alpha histogram looks.
    3. **Detached-fragment removal** — leftover low-alpha regions separated
       from the solid body are deleted.
    """
    rgb = np.asarray(image_rgb, dtype=np.float32)
    a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    if not (a > 0).any() or not (a < 0.985).any():
        return np.clip(rgb, 0.0, 255.0), a

    # 1 — Local background field (transparent pixels dominate the estimate,
    # edge-aware so it never leaks across the contour).
    B = estimate_bg_field(rgb, a, max(2, radius // 2))

    # 2 — Full-resolution un-mix. The low-res chain already decontaminated,
    # but the final upscale interpolated the edges back toward B.
    band = (a > 0.03) & (a < 0.98)
    a_safe = np.maximum(a, 0.08)[..., None]
    F = (rgb - (1.0 - a)[..., None] * B) / a_safe
    F = np.clip(F, 0.0, 255.0)
    wgt = (0.85 * band.astype(np.float32))[..., None]
    rgb = rgb * (1.0 - wgt) + F * wgt
    rgb = np.where((a > 0.98)[..., None], np.asarray(image_rgb,
                                                     dtype=np.float32), rgb)

    # 2b — Background-colour match: after un-mixing, a semi pixel that is
    # still essentially the local background has no foreground content.
    d = np.sqrt(((rgb - B) ** 2).mean(axis=2))
    bg_band = (a > 0.005) & (a < 0.92) & (d < 20.0)
    if bg_band.any():
        a = np.where(bg_band, 0.0, a)

    # 3 — Bright neutral halo suppression (grey fringe that is not itself the
    # background colour, e.g. grey spill on a light object; warm strands are
    # chroma-gated out inside).
    strength = float(cfg.get('halo_strength', 0.0) or 0.0)
    if strength > 0 and (a > min_a).any() and (a < 0.985).any():
        a = suppress_luminance_halo(rgb, a, strength=strength, radius=radius,
                                    zone=None, min_a=min_a)

    # 4 — Geometrically detached fragments.
    a = trim_detached_fringe(a)
    return np.clip(rgb, 0.0, 255.0), np.clip(a, 0.0, 1.0)