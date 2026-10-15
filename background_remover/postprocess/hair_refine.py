"""Hair-aware edge refinement for model mattes.

Why this stage exists
---------------------
The segmentation model (BiRefNet / ISNet) already outputs a soft alpha that
carries most of the hair / fur / feather structure, but three hair-specific
artifacts survive — worst on curly / wavy hair and on thin fly-away strands:

- **background spill** — semi-transparent pixels whose colour is essentially
  the local background (the wall showing through curly hair, or a dark/light
  tint bleeding onto hair at the contour),
- **the white / grey fringe (halo)** — a 1–3 px shell that is much lighter
  and more neutral than the solid foreground around it,
- **mixed strand colours** — thin hair pixels whose RGB is a mix of hair and
  background (grey wash on dark strands, colour tint on light strands).

What it does
------------
A dedicated, hair-aware refinement chain confined to the transition band:

1. **hair / fine-structure detection** — the Laplacian texture map and the
   soft-alpha band identify where hair actually lives;
2. **narrow transition band** — ``edge_band_width`` localises every operation
   around the silhouette so the solid interior never changes;
3. **alpha re-projection** — the colour-implied alpha (classical F/B
   projection, dot(I−B, F−B) / |F−B|²) is blended with the prediction and
   gated by Lab foreground/background separation and adaptive thresholds;
4. **Lab-based spill reduction** — background-coloured rim pixels are pulled
   toward alpha 0 using perceptual (Lab) similarity to the local background;
5. **halo removal** — bright / low-chroma fringe is detected by luminance
   lift against the solid interior and pulled down without touching genuine
   hair colour (warm, structured strands are chroma- and strand-gated);
6. **strand preservation** — pixels on fine structure whose colour matches
   the local foreground (``strand_preserve_threshold``) are exempt from every
   alpha-lowering pass, so low-alpha thin strands survive;
7. **curl-aware gap handling** — only *narrow*, confidently-background holes
   hugging the solid body are plugged; alpha is never raised, so the negative
   space between curls stays transparent;
8. **edge-aware weighting** — every strength map is edge-preserved with a
   guided filter driven by the photo, so refinement follows strand structure
   instead of blurring across it;
9. **decontamination** — recovered strand colours are un-mixed against an
   edge-aware background field (``decontamination_strength``), removing the
   background wash from light and dark hair alike;
10. **adaptive thresholds** — spill / strand / gap cutoffs are derived from
    the local band distribution instead of hard global values.

Hard guarantees: solid interior pixels (alpha ``>= solid_floor``) are
bit-identical in both alpha and RGB, pure background stays 0, alpha is never
raised, and coherent flat surfaces (a white shirt on a bright backdrop) are
protected by a flat-interior gate, so clothing / face / skin are never
damaged.
"""

import numpy as np
from PIL import Image

from ..utils import box_blur, erode, guided_filter, to_pil
from ..matting.edge_detection import edge_zone, fg_similarity_lab
from ..color_ops import lab_distance
from .alpha_refine import _bg_field_robust, _reach, _texture_map
from .edge_quality import (
    _robust_fg, _bg_similarity, _flat_surface, _to_work,
    _up_f, _up_b, _up3,
)

ANALYSIS_CAP = 2048  # Increased from 1280 for better high-res hair detail
SOLID_FLOOR = 0.985


def refine_hair(image_rgb, alpha, strategy):
    """Return ``(decontaminated_rgb, refined_alpha)`` — hair-aware refinement.

    ``image_rgb``: HxWx3 float32 RGB (0..255, ideally already decontaminated),
    ``alpha``: float32 0..1 at the same size. Gated by ``hair_refine``
    (default True — off for razor-edge categories). Only the transition band
    is touched: solid interior pixels are bit-identical on output.
    """
    cfg = strategy or {}
    rgb0 = np.asarray(image_rgb, dtype=np.float32)
    a0 = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)

    if not cfg.get('hair_refine', True) or cfg.get('color_mode'):
        return np.clip(rgb0, 0.0, 255.0), a0
    if not (a0 > 0.02).any() or not (a0 < 0.985).any():
        return np.clip(rgb0, 0.0, 255.0), a0

    m = _analysis(rgb0, a0, int(cfg.get('analysis_cap', ANALYSIS_CAP)), cfg)
    band = np.asarray(m['band'], dtype=bool)

    # Solid flat interior, away from the transition band → absolute lock.
    body = (a0 >= SOLID_FLOOR) & ~band

    rgb, a = rgb0.copy(), a0.copy()

    # 3+4+6+8+10 — Lab-gated, edge-aware alpha re-projection (rim spill).
    a = _rim_alpha(rgb, a, m, cfg)
    # 7 — curl-aware background-gap cleanup between fine strands.
    a = _gap_cleanup(a, m, cfg)
    # 5 — white / grey halo suppression + clean-up of the halo colours.
    rgb, a = _halo_cleanup(rgb, a, m, cfg)
    # 9 — strand colour decontamination (edge-aware un-mix).
    rgb = _strand_decon(rgb, a, m, cfg)

    # Absolute protection: the solid interior is never touched.
    rgb = np.where(body[..., None], rgb0, rgb)
    a = np.where(body, a0, a)
    return np.clip(rgb, 0.0, 255.0), np.clip(a, 0.0, 1.0)


# ---------------------------------------------------------------------- #
# analysis maps (computed on a capped work copy, applied at full res)
# ---------------------------------------------------------------------- #
def _analysis(rgb, a, cap, cfg):
    """Per-pixel maps needed by the refinement stages.

    Heavy colour / structure analysis runs at ``cap`` resolution; maps are
    then upscaled to the full size so every *decision* lands on the true
    pixels. Booleans upscale NEAREST (structure flags), continuous maps with
    LANCZOS / BILINEAR.
    """
    rgb_w, a_w, s = _to_work(rgb, a, cap)
    full = (h, w) = a.shape
    work = s != 1.0

    lum_w = _lum(rgb_w)
    tex_w = np.clip(_texture_map(lum_w), 0.0, 1.0)

    # Robust foreground colour + edge-aware background field.
    F_w = _robust_fg(rgb_w, a_w, s if work else 1.0)
    B_w, reach_w = _bg_field_robust(rgb_w, a_w, radius=7)
    B_w = _guided_color(B_w, rgb_w, radius=max(4, min(9, 7)))

    # Lab similarity of every pixel to the local fg and bg colour fields.
    fg_sim_w = np.clip(fg_similarity_lab(rgb_w, a_w, radius=7), 0.0, 1.0)
    bg_sim_w = np.clip(_bg_similarity(rgb_w, B_w), 0.0, 1.0)

    # How strongly the local foreground / background colours separate in Lab.
    # The projection is only trusted where this is high.
    try:
        sep_w = np.clip((lab_distance(F_w, B_w) - 14.0) / 80.0, 0.0, 1.0)
    except Exception:
        sep_w = np.zeros_like(bg_sim_w)

    # Self-adaptive fine-structure threshold on the transition band.
    band_w = _transition_band(a_w, cfg)
    if band_w.any():
        fine_thr = float(np.clip(np.percentile(tex_w[band_w], 55), 0.16, 0.70))
    else:
        fine_thr = 0.42
    fine_thr = float(cfg.get('fine_thr', fine_thr))
    fine_w = tex_w > fine_thr

    # Strand protection: real fine structure whose colour matches the local
    # foreground — these are physical strands, never spill.
    sp_thr = float(cfg.get('strand_preserve_threshold', 0.50))
    strand_w = fine_w & (fg_sim_w >= sp_thr)

    # Anti-regression: flat *interior* surfaces (garment, fabric, smooth
    # product faces) are never treated as a background gap, even when their
    # colour resembles the backdrop (white shirt on a bright wall).
    fr = int(float(cfg.get('flat_radius', 8)) * s) if work \
        else int(float(cfg.get('flat_radius', 8)))
    fc = int(float(cfg.get('flat_clearance', 6)) * s) if work \
        else int(float(cfg.get('flat_clearance', 6)))
    flat_w = _flat_surface(tex_w, max(1, fr))
    interior_flat_w = flat_w & ~_reach((a_w < 0.35), max(1, fc))

    # How close every pixel is to the solid body (gap handling is only legal
    # inside a small reach of the foreground silhouette).
    gr = float(cfg.get('gap_reach', 7))
    gr = int(round(gr * s)) if work else int(round(gr))
    near_solid_w = _reach(a_w >= 0.8, max(1, gr))

    if not work:
        return dict(tex=tex_w, fine=fine_w, strand=strand_w, band=band_w,
                    fg_sim=fg_sim_w, bg_sim=bg_sim_w, sep=sep_w,
                    bg_reach=reach_w, near_solid=near_solid_w,
                    interior_flat=interior_flat_w, F=F_w, B=B_w)
    return dict(
        tex=_up_f(tex_w, full), fine=_up_b(fine_w, full),
        strand=_up_b(strand_w, full), band=_up_b(band_w, full),
        fg_sim=_up_f(fg_sim_w, full), bg_sim=_up_f(bg_sim_w, full),
        sep=_up_f(sep_w, full), bg_reach=_up_b(reach_w, full),
        near_solid=_up_b(near_solid_w, full),
        interior_flat=_up_b(interior_flat_w, full),
        F=_up3(F_w, full), B=_up3(B_w, full),
    )


def _transition_band(a, cfg):
    """Narrow transition band around the mask contour (edge_band_width)."""
    half = float(cfg.get('edge_band_width', 10.0) or 10.0)  # Increased from 6.0 to 10.0 for curly hair
    half = max(0.0, min(half, 32.0))  # Increased max from 24 to 32
    if half <= 0:
        half = 1.0
    return edge_zone(a, thresh=float(cfg.get('band_thresh', 0.02)),  # Lowered from 0.03
                     dilation=int(round(half)))


# ---------------------------------------------------------------------- #
# stage 1 — Lab-gated alpha re-projection (rim spill)
# ---------------------------------------------------------------------- #
def _rim_alpha(rgb, a, m, cfg):
    """Pull background-dominated rim alpha toward its colour-implied value.

    The observed colour I is a mix of foreground F and background B:
    I = a·F + (1−a)·B, so the colour implies a = dot(I−B, F−B)/|F−B|².
    Where the implied alpha is clearly below the prediction AND the pixel
    looks like the local background (adaptive Lab gates) AND F/B really
    separate (evidence), the prediction was inflated by background spill and
    is pulled toward the implied value. Fine strands and the solid body are
    exempt. Alpha is never raised and the pull map is edge-aware (guided by
    the photo).
    """
    a = np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0)
    rgb = np.asarray(rgb, dtype=np.float32)
    band = np.asarray(m['band'], dtype=bool)
    if not band.any():
        return a

    F = np.asarray(m['F'], dtype=np.float32)
    B = np.asarray(m['B'], dtype=np.float32)
    fb = F - B
    denom = (fb * fb).sum(axis=2) + 1e-4
    a_imp = np.clip(((rgb - B) * fb).sum(axis=2) / denom, 0.0, 1.0)

    fg_sim = np.clip(np.asarray(m['fg_sim'], dtype=np.float32), 0.0, 1.0)
    bg_sim = np.clip(np.asarray(m['bg_sim'], dtype=np.float32), 0.0, 1.0)
    sep = np.clip(np.asarray(m['sep'], dtype=np.float32), 0.0, 1.0)

    sel = band & (a > float(cfg.get('spill_min_a', 0.04)))
    if not sel.any():
        return a

    # Adaptive thresholds (req. 10): from the band distribution, clamped.
    bg_lo = cfg.get('spill_bg_lo')
    if bg_lo is None:
        bg_lo = _pct(bg_sim, sel, 40, 0.40, 0.78)
    fg_hi = cfg.get('spill_fg_hi')
    if fg_hi is None:
        fg_hi = _pct(fg_sim, sel, 62, 0.28, 0.80)

    # Confidence / evidence for "this pixel is background spill" (req. 10).
    # In dense hair the local foreground/background fields barely separate
    # (every F/B pair near the strand mass is dark), so the F/B separation
    # alone is not enough. The decisive signal is *dominance*: the pixel
    # matches the local background much better than the local foreground
    # (bg_sim − fg_sim), which stays high exactly where separation collapses.
    disc = np.clip((bg_sim - fg_sim - 0.05) / 0.30, 0.0, 1.0)
    ev = np.maximum(np.clip(sep / 0.5, 0.0, 1.0), disc)

    cond = sel & (bg_sim > float(bg_lo)) & (fg_sim < float(fg_hi)) \
        & (a_imp < a - float(cfg.get('spill_delta', 0.07))) \
        & (ev > float(cfg.get('spill_ev', 0.10))) \
        & np.asarray(m['bg_reach'], dtype=bool)
    cond &= ~np.asarray(m['strand'], dtype=bool)
    cond &= ~(fg_sim > float(cfg.get('strand_preserve_threshold', 0.50)) + 0.25)
    if not cond.any():
        return a

    pull = np.clip((a - a_imp) / 0.25, 0.0, 1.0)
    st = float(cfg.get('spill_strength', 0.85))
    ar = float(cfg.get('alpha_refine_strength', 0.85))
    w = np.clip(st * ar * ev * pull, 0.0, 1.0)
    w = _edge_smooth(w, rgb, 3)

    out = a * (1.0 - w) + a_imp * w

    # Firm pull: overwhelmingly background-dominant colour + weak implied
    # alpha → land on the colour-implied value (never above the prediction).
    firm = cond & (ev > float(cfg.get('spill_firm_ev', 0.55))) \
        & ((bg_sim - fg_sim) > float(cfg.get('spill_firm_disc', 0.35))) \
        & (pull > float(cfg.get('spill_firm_pull', 0.85)))
    if firm.any():
        out = np.where(firm, a_imp, out)
    out = np.minimum(out, a)
    out = np.where(cond, out, a)
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------- #
# stage 2 — curl-aware background-gap cleanup
# ---------------------------------------------------------------------- #
def _gap_cleanup(a, m, cfg):
    """Plug genuinely background-coloured holes between fine strands.

    A hole must (a) hug the solid body, (b) have a real background sample
    within reach, (c) be Lab-similar to the local background and clearly
    *different* from the local foreground, and (d) not sit on fine structure
    that could be a real strand. Flat interiors are excluded outright. Alpha
    is never raised, so the negative space between curls stays transparent —
    we only remove holes that are demonstrably background.
    """
    a = np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0)
    zone = (a > float(cfg.get('gap_min_a', 0.02))) & (a < float(cfg.get('gap_a_cap', 0.75)))  # Lowered cap from 0.80
    if not zone.any():
        return a

    bg_sim = np.clip(np.asarray(m['bg_sim'], dtype=np.float32), 0.0, 1.0)
    fg_sim = np.clip(np.asarray(m['fg_sim'], dtype=np.float32), 0.0, 1.0)

    cand0 = zone & np.asarray(m['near_solid'], dtype=bool) \
        & np.asarray(m['bg_reach'], dtype=bool) \
        & ~np.asarray(m['interior_flat'], dtype=bool) \
        & ~np.asarray(m['strand'], dtype=bool)
    if not cand0.any():
        return a

    bg_lo = cfg.get('gap_bg_lo')
    if bg_lo is None:
        bg_lo = _pct(bg_sim, cand0, 50, 0.45, 0.75)  # Increased from 45/0.42/0.72
    fg_hi = cfg.get('gap_fg_hi')
    if fg_hi is None:
        fg_hi = _pct(fg_sim, cand0, 50, 0.30, 0.65)  # Lowered from 55/0.35/0.68

    cand = cand0 & (bg_sim > float(bg_lo)) & (fg_sim < float(fg_hi))
    if not cand.any():
        return a

    s = np.clip((bg_sim - 0.45) / 0.35, 0.0, 1.0) \
        * np.clip((0.60 - fg_sim) / 0.45, 0.0, 1.0)  # Adjusted thresholds
    out = a * (1.0 - 0.8 * s * cand)  # Reduced from 0.9

    # Confident, narrow, background-coloured holes are removed outright.
    # But be more conservative - require VERY high bg_sim and low fg_sim
    hard = cand & (bg_sim > 0.78) & (fg_sim < 0.18) & (a < 0.65)  # Stricter
    if hard.any():
        out = np.where(hard, 0.0, out)
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------- #
# stage 3 — white / grey halo suppression
# ---------------------------------------------------------------------- #
def _halo_cleanup(rgb, a, m, cfg):
    """Pull bright / neutral fringe down without fading genuine hair colour.

    The halo is a bright, low-chroma shell around the silhouette. It is
    detected by luminance-lift against the *solid interior* reference (eroded
    so the halo cannot count itself) gated by a neutral-chroma test. Warm /
    structured strands and foreground-matching pixels escape. In the same
    region the colour is decontaminated so removing the halo never leaves a
    faded strand behind.
    """
    a = np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0)
    rgb = np.asarray(rgb, dtype=np.float32)
    band = np.asarray(m['band'], dtype=bool)
    if not band.any() or not (a > 0.9).any():
        return rgb, a

    semi = band & (a > float(cfg.get('halo_min_a', 0.05))) & (a < 0.999)  # Lowered from 0.10
    if not semi.any():
        return rgb, a

    solid = (a >= SOLID_FLOOR).astype(np.float32)
    rad = max(2, int(cfg.get('halo_radius', 8)))  # Increased from 6
    interior = box_blur(solid, max(1, rad // 2)) > 0.999
    k = max(3, 2 * (rad // 2) + 1)
    interior = erode((interior.astype(np.uint8)) * 255, k) > 0
    if not interior.any():
        return rgb, a

    lum = np.clip(_lum(rgb), 0.0, 255.0)
    fw = box_blur(interior.astype(np.float32), rad) + 1e-4
    fg_lum = box_blur(lum * interior.astype(np.float32), rad) / fw

    lift = lum - fg_lum
    base = np.clip(float(np.percentile(fg_lum[interior], 20)) * 0.08, 4.0, 15.0)  # More sensitive
    lift_n = np.clip((lift - base) / (base * 4.0), 0.0, 1.0)  # More sensitive

    mx = np.maximum(np.abs(rgb[..., 0] - rgb[..., 1]),
                    np.abs(rgb[..., 1] - rgb[..., 2]))
    grey = 1.0 - np.clip(mx / 50.0, 0.0, 1.0)  # More sensitive (was 70)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    chroma_w = np.clip((30.0 - chroma) / 20.0, 0.0, 1.0)  # More sensitive (was 40/25)

    # Also detect DARK halos
    dark_lift = fg_lum - lum
    dark_base = np.clip(float(np.percentile(fg_lum[interior], 20)) * 0.08, 4.0, 15.0)
    dark_lift_n = np.clip((dark_lift - dark_base) / (dark_base * 4.0), 0.0, 1.0)

    halo_w = (lift_n + dark_lift_n) * grey * chroma_w

    # Strands, foreground-matching pixels and the solid body never get pulled.
    protect = (np.asarray(m['strand'], dtype=bool)
               | (np.asarray(m['fg_sim'], dtype=np.float32) > 0.55))  # Lowered from 0.60

    # Colour-implied alpha: how much real foreground sits under this pixel.
    F = np.asarray(m['F'], dtype=np.float32)
    B = np.asarray(m['B'], dtype=np.float32)
    fb = F - B
    denom = (fb * fb).sum(axis=2) + 1e-4
    a_imp = np.clip(((rgb - B) * fb).sum(axis=2) / denom, 0.0, 1.0)

    hm = float(cfg.get('halo_min', 0.18))  # Lowered from 0.22
    cond = semi & ~protect & (halo_w > hm)
    if not cond.any():
        return rgb, a

    hw = float(cfg.get('halo_strength', 0.85)) * 0.9 * halo_w  # Increased from 0.70
    soft = a * (1.0 - hw)
    floor_a = np.clip(a_imp, 0.02, 1.0)  # Lowered from 0.03
    out = np.maximum(soft, floor_a)
    out = np.minimum(out, a)
    a = np.where(cond, out, a)
    a = np.clip(a, 0.0, 1.0)

    # Decontaminate the (still visible) fringe colours against the edge-aware
    # background field so the darker/true colour shows through instead of the
    # pale halo mix.
    focus = cond & (a > float(cfg.get('halo_min_a', 0.05)))  # Lowered from 0.10
    if focus.any():
        rgb = _decon(rgb, a, focus, B,
                     strength=float(cfg.get('decontamination_strength', 0.90)))  # Increased from 0.80
    return rgb, a


# ---------------------------------------------------------------------- #
# stage 4 — strand colour decontamination (edge-aware un-mix)
# ---------------------------------------------------------------------- #
def _strand_decon(rgb, a, m, cfg):
    """Recover the true strand colour inside fine structures.

    Re-un-mix fine-structure pixels against the precomputed edge-aware
    background field: dark strands lose their grey wall wash, light strands
    lose the dark background tint. Only pixels on real structure with a real
    background sample within reach are touched, and the blend weight is
    edge-aware (follows strand structure).
    """
    a = np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0)
    rgb = np.asarray(rgb, dtype=np.float32)
    band = np.asarray(m['band'], dtype=bool)
    strand_min = float(cfg.get('strand_min_a', 0.04))  # Lowered from 0.06

    focus = band & (a > strand_min) & (a < 0.98) \
        & np.asarray(m['bg_reach'], dtype=bool) \
        & np.asarray(m['fine'], dtype=bool)
    # Skip pixels that are confidently background (holes already cleaned).
    focus &= ~((np.asarray(m['bg_sim'], dtype=np.float32) > 0.65)  # Increased from 0.62
               & (np.asarray(m['fg_sim'], dtype=np.float32) < 0.25))  # Lowered from 0.30
    if not focus.any():
        return rgb

    strength = float(cfg.get('decontamination_strength', 0.90))  # Increased from 0.80
    w = focus.astype(np.float32)
    if w.any():
        w = _edge_smooth(w, rgb, 4) * strength

    B = np.asarray(m['B'], dtype=np.float32)
    a_safe = np.maximum(a, max(0.06, strand_min))[..., None]  # Lowered from 0.08
    F = np.clip((rgb - (1.0 - a)[..., None] * B) / a_safe, 0.0, 255.0)
    out = rgb * (1.0 - w[..., None]) + F * w[..., None]
    body = (a >= SOLID_FLOOR)[..., None]
    out = np.where(body, rgb, out)
    return np.clip(out, 0.0, 255.0)


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def _decon(rgb, a, focus, B, strength=0.8, min_a=0.08):
    """Un-mix foreground colour in ``focus`` regions against field ``B``."""
    a_safe = np.maximum(a, min_a)[..., None]
    F = np.clip((rgb - (1.0 - a)[..., None] * B) / a_safe, 0.0, 255.0)
    w = (strength * np.asarray(focus, dtype=np.float32))[..., None]
    out = rgb * (1.0 - w) + F * w
    body = (a >= SOLID_FLOOR)[..., None]
    out = np.where(body, rgb, out)
    return np.clip(out, 0.0, 255.0)


def _edge_smooth(w, rgb, radius=3):
    """Guided (edge-aware) smoothing of a 0..1 map using the photo as guide."""
    try:
        lum = np.clip(_lum(rgb), 0.0, 255.0) / 255.0
        return np.clip(guided_filter(np.asarray(w, dtype=np.float32), lum,
                                     radius=max(1, int(radius)), eps=0.06),
                       0.0, 1.0)
    except Exception:
        return np.clip(w, 0.0, 1.0)


def _guided_color(src, guide_rgb, radius=7):
    """Edge-aware smoothing of a colour field (background estimate) so it
    never leaks across hair strands or object contours."""
    try:
        lum = np.clip(_lum(guide_rgb), 0.0, 255.0) / 255.0
        out = np.empty_like(src, dtype=np.float32)
        for c in range(3):
            out[..., c] = guided_filter(src[..., c], lum, radius=max(2, int(radius)),
                                        eps=0.06)
        return np.clip(out, 0.0, 255.0)
    except Exception:
        return src


def _pct(v, sel, pct, lo, hi):
    """Adaptive percentile threshold (req. 10) with safe clamps."""
    vals = v[sel]
    if vals.size < 64:
        return float(lo)
    return float(np.clip(float(np.percentile(vals, float(pct))), lo, hi))


def _lum(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


__all__ = ['refine_hair']