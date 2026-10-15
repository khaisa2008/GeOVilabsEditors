"""Final edge-aware alpha matte refinement + mask validation.

Runs *after* the halo / decontamination chain as the last quality stage and
fixes the residue those stages cannot remove in one shot:

1. Background "gaps" between fine strands — semi-transparent pixels whose
   colour is indistinguishable from the local background (the wall/white
   showing through curly hair).
2. Detached / far-away semi-transparent noise — the *validation* pass: faint
   alpha pixels that are geometrically separated from the foreground body.
3. The white / grey halo and the colour-contaminated 1–2 px shell (white
   clothing on a coloured background, shoulder / arm spill) — a final soft
   spill pull followed by an edge-only decontamination pass.
4. Jagged / staircase alpha edges — very light feathering (~1–2 px) confined
   to the transition band.

It is **edge-aware** (every stage is confined to the contour zone and the
colour fields it uses are edge-preserving) and **region-aware**: strands /
fur / feathers are recognised by their fine Laplacian texture and handled
with their own thresholds, flat skin / clothing surfaces with stricter ones.
Pixels that sit on strong real structure, or whose colour matches the local
foreground, are always protected. The solid body (alpha >= floor) and every
interior RGB value are never modified, and foreground colours are never
pushed away from the original photo.

The heavy analysis maps are computed on a capped working copy; every *alpha
decision* is applied to the full-resolution mask, so the final alpha is never
re-interpolated.
"""

import numpy as np
from PIL import Image

from ..utils import box_blur, erode, gaussian, normalize01, to_pil
from ..matting.edge_detection import edge_zone, fg_similarity_lab
from .decontamination import estimate_bg_field

ANALYSIS_CAP = 2048        # Increased from 1792 for better high-res hair detail
FEATHER_MAX = 2.0          # guaranteed "very light" 1-2px feather cap


def refine_final_alpha(image_rgb, alpha, strategy):
    """Return (decontaminated_rgb, refined_alpha) — the last matte stage.

    ``image_rgb``: HxWx3 float RGB (0..255, ideally already decontaminated),
    ``alpha``: float32 0..1 at the same size. Gated by
    ``strategy['alpha_refine']`` (default True). Interior pixels (a >= 0.98)
    are bit-identical on output, both in alpha and colour.
    """
    cfg = strategy or {}
    if not cfg.get('alpha_refine', True):
        return image_rgb, _clip01(alpha)

    rgb = np.asarray(image_rgb, dtype=np.float32)
    a = _clip01(np.asarray(alpha, dtype=np.float32))

    if not (a > 0.02).any():
        return np.clip(rgb, 0.0, 255.0), a

    # ---- Analysis maps (capped; all alpha decisions happen at full res). ----
    w_rgb, w_a, s = _to_work(rgb, a, int(cfg.get('refine_cap', ANALYSIS_CAP)))
    maps = _analysis(w_rgb, w_a, cfg)
    if s != 1.0:
        maps = _maps_to_full(maps, a.shape)

    # 1 — Region-aware background-gap cleanup (wall showing through hair,
    #     background residue on clothing / skin edges).
    a = _clean_semitrans(rgb, a, maps, cfg)

    # 2 — Validation: no semi-transparent pixel detached from the foreground
    #     body survives (noise, floaters, skirt remnants).
    a = validate_alpha(a, cfg, structure=maps['texture'],
                       fg_sim=maps['fg_sim'], bg_sim=maps['bg_sim'])

    # 3 — Final soft spill + colour decontamination on the outermost rim
    #     (white / grey halo, shoulder / clothing colour contamination).
    rgb, a = _rim_pass(rgb, a, cfg)

    # 4 — Very light feather (1-2 px) on the final transition band only.
    a = _feather(a, cfg)

    return np.clip(rgb, 0.0, 255.0), np.clip(a, 0.0, 1.0)


def validate_alpha(alpha, strategy=None, structure=None, fg_sim=None,
                   bg_sim=None):
    """Validation: zero faint disconnected semi-transparent pixels.

    Only *faint* (``alpha < validate_mass_floor``) pixels are ever dropped:
    a coherent semi-transparent mass (a soft shirt, a sheer fabric, a wide
    strand group at ~0.4-0.6) is real foreground and is NEVER treated as
    noise — the previous "any pixel under solid_floor must be near the body"
    rule could cut the middle of a garment whose prediction sat around 0.5.

    The drop decision is geometric: the pixel must not be within
    ``validate_mass_reach`` of any pixel ``>= mass_floor`` (a semi-mass) or
    of the solid body. That removes faint haze / floaters / background specks
    far outside the cutout without ever removing interior objects.

    Escapes preserve genuine detail:

    - strong fine structure (a thin strand / fur filament keeps itself),
    - a colour that clearly matches the local foreground,
    - nothing is touched if the confidence would not justify it.

    Pixels ``>= mass_floor`` and fully-transparent pixels are never altered.
    This is a *spatial* filter, not a global erode/dilate.
    """
    cfg = strategy or {}
    a = _clip01(np.asarray(alpha, dtype=np.float32))
    min_keep = float(cfg.get('validate_min_keep', 0.03))
    mass_floor = float(cfg.get('validate_mass_floor', 0.35))
    mass_reach = int(cfg.get('validate_mass_reach', 12))

    cand = (a > min_keep) & (a < mass_floor)
    if not cand.any():
        return a

    # Near any semi-mass or the solid body → part of the subject, keep.
    near = _reach(a >= mass_floor, mass_reach) | _reach(a >= 0.8, mass_reach)

    if not near.any():
        # Nothing reachable at all — the whole foreground is a thin filament
        # network; be conservative and keep it rather than destroy it.
        return a

    escape = np.zeros(a.shape, dtype=bool)
    if structure is not None:
        escape |= np.asarray(structure) > 0.5
    if fg_sim is not None and bg_sim is not None:
        escape |= (np.asarray(fg_sim) > 0.6) & (np.asarray(bg_sim) < 0.5)

    out = a.copy()
    drop = cand & ~near & ~escape
    if drop.any():
        out[drop] = 0.0
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------- #
# analysis maps
# ---------------------------------------------------------------------- #
def _analysis(rgb, a, cfg):
    """Per-pixel region / colour-similarity maps (0..1 floats + bool masks)."""
    lum = _lum(rgb)
    texture = _texture_map(lum)

    band = (a > 0.03) & (a < 0.92)
    if band.any():
        thr = float(np.clip(np.percentile(texture[band], 62), 0.10, 0.75))
    else:
        thr = 0.4
    hair = band & (texture >= thr)
    flat = band & (texture < thr)

    fg_sim = np.clip(fg_similarity_lab(rgb, a, radius=7), 0.0, 1.0)

    # Robust local background: only confident-background pixels (a<0.35) feed
    # the estimate. The naive (1-a)-weighted field lets semi-transparent
    # strands pollute B, making strand pixels look "background-like".
    B, bg_reach = _bg_field_robust(rgb, a, radius=7)
    try:
        from ..color_ops import lab_distance
        d = lab_distance(rgb, B)
    except Exception:
        d = np.sqrt(((rgb - B) ** 2).sum(axis=2))
    bg_sim = 1.0 - np.clip((d - 10.0) / 50.0, 0.0, 1.0)

    # Near the solid body only. The whole point is to stop the background
    # gap / halo *around* the silhouette (between strands, on the skirt),
    # never to sweep a flat coherent garment that merely *looks* like the
    # background (e.g. a white shirt on a light studio backdrop). Requiring
    # the candidate to sit within ``gap_reach`` of solid foreground makes
    # every cleanup decision structural instead of purely colour-based.
    gap_reach = int(cfg.get('gap_reach', 8))
    near_solid = _reach(a >= 0.8, gap_reach)

    return {
        'texture': texture,
        'hair': hair,
        'flat': flat,
        'fg_sim': fg_sim,
        'bg_sim': bg_sim,
        'bg_reach': bg_reach,
        'near_solid': near_solid,
    }


def _bg_field_robust(rgb, a, radius=7):
    """Background colour from confident background pixels only (no guided
    filter, cheap). ``reach`` tells where a real background sample exists
    within ``radius`` — cleanup is only legal where this is True."""
    bg_mask = (a < 0.35).astype(np.float32)
    reach = box_blur(bg_mask, radius) > 0.0
    w = box_blur(bg_mask, radius) + 1e-4
    B = np.empty_like(rgb, dtype=np.float32)
    for c in range(3):
        B[..., c] = box_blur(rgb[..., c] * bg_mask, radius) / w
    return np.clip(B, 0.0, 255.0), reach


def _texture_map(lum):
    """0..1 fine-structure map: strong where thin strands / fur / feathers
    live, ~0 on flat skin, clothing and smooth gradients."""
    fine = _lapmag(lum, 1.1)
    coarse = _lapmag(lum, 4.0)
    t = normalize01(np.clip(fine - 0.6 * coarse, 0.0, None))
    return normalize01(gaussian(t, 1.2))


def _lapmag(lum, sigma):
    """Absolute Laplacian magnitude after a Gaussian pre-smooth."""
    g = gaussian(lum, sigma)
    gy, gx = np.gradient(g)
    gyy, gyx = np.gradient(gy, axis=0), np.gradient(gx, axis=0)
    gxy, gxx = np.gradient(gy, axis=1), np.gradient(gx, axis=1)
    return np.abs(gyy) + np.abs(gyx + gxy) + np.abs(gxx)


# ---------------------------------------------------------------------- #
# stage 1 — background-gap cleanup
# ---------------------------------------------------------------------- #
def _clean_semitrans(rgb, a, m, cfg):
    """Region-aware cleanup of semi-transparent background residue.

    A semi-transparent pixel whose colour is *indistinguishable from the
    local background* (bg_sim high) while differing from the local
    foreground (fg_sim low) is background showing through — a gap between
    strands or residue on a flat edge. Its alpha is pulled toward zero
    continuously; confident gaps are killed outright. The pull is gentler in
    hair / fur (fine strands live in soft alpha) and never touches pixels on
    protected structure, strongly foreground-coloured pixels, or the body.
    """
    min_keep = float(cfg.get('min_keep', 0.03))
    gap_zone = (a > min_keep) & (a < 0.70)
    if not gap_zone.any():
        return a

    fg_sim = np.clip(np.asarray(m['fg_sim']), 0.0, 1.0)
    bg_sim = np.clip(np.asarray(m['bg_sim']), 0.0, 1.0)
    texture = np.clip(np.asarray(m['texture']), 0.0, 1.0)

    bg_lo = float(cfg.get('gap_bg_lo', 0.42))
    fg_hi = float(cfg.get('gap_fg_hi', 0.58))
    bg_reach = np.asarray(m['bg_reach'], dtype=bool)
    near_solid = np.asarray(m['near_solid'], dtype=bool)
    # Only act around the silhouette: the candidate must sit within
    # ``gap_reach`` of solid foreground. A large flat garment whose colour
    # matches the background (white shirt on light studio backdrop) is NOT
    # erased — it does not hug the solid body as a thin gap does.
    cand = gap_zone & (bg_sim > bg_lo) & (fg_sim < fg_hi) & bg_reach \
        & near_solid

    s_flat = (np.clip((bg_sim - 0.35) / 0.40, 0.0, 1.0)
              * np.clip((0.62 - fg_sim) / 0.45, 0.0, 1.0))
    s = np.where(np.asarray(m['flat']), s_flat, s_flat * 0.85)

    # Real fine structure (a physical strand, not a flat gap) must survive:
    # where the pixel sits on strong high-frequency texture the pull is
    # nearly disabled, otherwise sparse strands whose local fg average is
    # dominated by the background would be wiped as "background gaps".
    fine = np.clip(np.asarray(m['texture']), 0.0, 1.0) > 0.5
    s = s * (1.0 - 0.8 * fine)

    out = a * (1.0 - cand * s)

    protected = ((texture > 0.5) | (fg_sim > 0.62) | (a >= 0.5))
    hard = cand & (bg_sim > 0.70) & (fg_sim < 0.30) & ~protected
    if hard.any():
        out = np.where(hard, 0.0, out)

    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------- #
# stage 3 — rim spill + decontamination
# ---------------------------------------------------------------------- #
def _rim_pass(rgb, a, cfg):
    """Soft spill pull + colour decontamination on the outermost rim.

    The previous chain's upscale re-mixes a 1-2 px shell back toward the
    background colour (grey fringe on white clothing, colour contamination
    on shoulders / arms). ``correct_spill_alpha`` re-projects alpha from the
    colour evidence at the real size, then a narrow un-mix recovers the true
    foreground colour. Pure-geometry interior / background pixels untouched.
    """
    zone = edge_zone(a, thresh=0.03, dilation=1)
    if not zone.any():
        return rgb, a

    strength = min(1.2, float(cfg.get('halo_strength', 0.8) or 0.8) * 0.8)
    if cfg.get('rim_spill', True):
        a = _rim_spill(rgb, a, zone,
                       radius=int(cfg.get('spill_radius', 6) or 6),
                       strength=strength, min_a=0.10)

    # Narrow decontamination on the fresh rim (after the pull the edges carry
    # a < 1 so the un-mix can recover the true colour).
    rim = edge_zone(a, thresh=0.03, dilation=2)
    if rim.any():
        rgb = _decontam_rim(rgb, a, rim, radius=4, strength=0.85)

    return rgb, a


def _rim_spill(rgb, a, zone, radius=6, strength=0.8, min_a=0.10):
    """Lean, full-resolution alpha spill pull for the outermost shell.

    Re-projects alpha from the colour evidence: a pixel whose implied alpha
    (colour un-mix against the local foreground / background) is far below
    its predicted alpha is background-dominated (grey fringe on white
    clothing, shoulder spill) and gets pulled down. Cheap box-blur fields
    only — the halo chain already did the accurate (guided/Lab) version, this
    pass mops up the shell that its downscale + LANCZOS upscale re-mixed.
    Confined to ``zone``; interior and pure background untouched.
    """
    a = _clip01(a)
    band = zone & (a > min_a) & (a < 0.99)
    shell = zone & (a >= 0.99)          # hard-mask boundary pixel carrying
    band = band | shell                 # the anti-aliased mixed colour
    if not band.any():
        return a

    fg_mask = (a > 0.9).astype(np.float32)
    if fg_mask.mean() > 0.01:
        fg_mask = (erode(fg_mask * 255.0, 5) > 0).astype(np.float32)
    if fg_mask.sum() <= 0:
        return a
    bg_mask = (a < 0.35).astype(np.float32)

    fw = box_blur(fg_mask, radius) + 1e-4
    bw = box_blur(bg_mask, radius) + 1e-4
    F = np.empty_like(rgb, dtype=np.float32)
    B = np.empty_like(rgb, dtype=np.float32)
    for c in range(3):
        F[..., c] = box_blur(rgb[..., c] * fg_mask, radius) / fw
        B[..., c] = box_blur(rgb[..., c] * bg_mask, radius) / bw

    fb = F - B
    denom = (fb * fb).sum(axis=2) + 1e-4
    a_imp = np.clip(((rgb - B) * fb).sum(axis=2) / denom, 0.0, 1.0)

    ev = np.sqrt((fb * fb).sum(axis=2))
    ev_n = np.clip((ev - 30.0) / 90.0, 0.0, 1.0)
    pull = np.clip((a - a_imp) / 0.25, 0.0, 1.0)
    w = strength * ev_n * pull * 0.8

    cond = band & (a_imp < a - 0.08)
    out = a.copy()
    if cond.any():
        out[cond] = a[cond] * (1.0 - w[cond]) + a_imp[cond] * w[cond]
    return np.clip(np.minimum(out, a), 0.0, 1.0)


def _decontam_rim(rgb, a, rim, radius=4, strength=0.85):
    """Recover the foreground colour of rim pixels against the local
    background field. No guided filter (cheap), confined to ``rim``, body
    pixels (a>=0.98) left pixel-identical."""
    B = estimate_bg_field(rgb, a, max(2, radius), use_guided=False)
    a_safe = np.maximum(a, 0.15)[..., None]
    F = np.clip((rgb - (1.0 - a)[..., None] * B) / a_safe, 0.0, 255.0)
    wgt = (strength * np.asarray(rim, dtype=np.float32))[..., None]
    out = rgb * (1.0 - wgt) + F * wgt
    body = (a >= 0.98)[..., None]
    out = np.where(body, rgb, out)
    return np.clip(out, 0.0, 255.0)


# ---------------------------------------------------------------------- #
# stage 4 — feather
# ---------------------------------------------------------------------- #
def _feather(a, cfg):
    """1-2 px feather confined to the transition band (anti-staircase).

    Uses a partial blend so the semi band is smoothed without raising the
    true background (which stays 0 away from the silhouette) and without
    ever eroding the solid body.
    """
    fe = float(cfg.get('feather', 1.0) or 1.0)
    fe = max(0.5, min(fe, FEATHER_MAX))
    if fe <= 0:
        return a
    band = (a > 0.03) & (a < 0.97)
    if not band.any():
        return a

    blur = _blur01(a, fe)
    w = _blur01(band.astype(np.float32), fe * 1.5)
    out = a * (1.0 - w) + blur * w
    solid = a >= 0.98
    out = np.maximum(out, np.where(solid, a, 0.0))
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def _reach(mask, r):
    """Chebyshev reach: True where any True pixel of ``mask`` is within ``r``."""
    r = int(r)
    if r <= 0:
        return np.asarray(mask, dtype=bool)
    return box_blur(np.asarray(mask, dtype=np.float32), r) > 0.0


def _to_work(rgb, a, cap):
    """Downscale the analysis copies (maps only, never the output mask)."""
    h, w = a.shape
    if max(h, w) <= cap:
        return rgb, a, 1.0
    s = cap / max(h, w)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    rgb_w = np.asarray(
        Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), 'RGB')
        .resize((nw, nh), Image.BILINEAR), dtype=np.float32)
    a_w = np.asarray(to_pil(a * 255.0).resize((nw, nh), Image.BILINEAR),
                     dtype=np.float32) / 255.0
    return rgb_w, a_w, s


def _maps_to_full(maps, size):
    """Upscale every map to the full resolution (NEAREST for masks)."""
    h, w = size
    out = {}
    for k, v in maps.items():
        if v.dtype == bool:
            small = to_pil(v.astype(np.uint8) * 255)
            out[k] = np.asarray(small.resize((w, h), Image.NEAREST)) > 0
        else:
            small = to_pil(np.clip(v, 0.0, 1.0) * 255.0)
            out[k] = np.asarray(small.resize((w, h), Image.LANCZOS),
                                dtype=np.float32) / 255.0
    return out


def _blur01(a, r):
    """Gaussian blur of a 0..1 array, back in the 0..1 range.

    PIL's GaussianBlur operates on uint8, so the input must be pre-scaled to
    the 0..255 range before blurring — otherwise a 0..1 alpha would be
    quantised to 0/1 and the feather would destroy the smooth transition.
    """
    return np.clip(gaussian(np.clip(a, 0.0, 1.0) * 255.0, r), 0.0, 255.0) / 255.0


def _lum(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _clip01(a):
    return np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0)


__all__ = ['refine_final_alpha', 'validate_alpha']