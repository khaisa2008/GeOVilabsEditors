"""Conservative fine-edge refinement for model mattes.

The model matte (BiRefNet / ISNet) already carries soft alpha for hair and
fur, and the pipeline's decontamination stage already un-mixes the edge
colours. What it does not do is touch *alpha* — so two classic hair artifacts
survive:

- the wall showing through curly hair (semi-transparent, background-coloured
  pixels *between* strands), and
- the 1-3px outer shell whose colour says "background" while the mask says
  "opaque" (the white / grey rim).

This module fixes exactly those two, plus a deeper un-mix of the strand
colours, using only spatially-gated, per-pixel adjustments:

1. **gap plug** — background-coloured semi-transparent holes *between* fine
   structures are pulled toward 0. A candidate must (a) live within a small
   reach of solid foreground, (b) be near a real background sample, (c) look
   like the local background **and** unlike the local foreground in Lab
   space. That colour disagreement is what keeps real strands and garments
   safe: hair strands match the local foreground estimate (high ``fg_sim``),
   white clothing on a bright backdrop matches it too, while a wall hole
   between strands matches only the background. Pixels sitting on strong fine
   structure need an even stronger background signal before they are touched
   (a pale fly-away strand has ``fg_sim`` high and escapes). Confident
   candidates are removed outright; the rest are pulled continuously.

2. **rim spill pull** — inside the narrow contour band the colour evidence
   re-projects the alpha; where the implied alpha is much lower than
   predicted the pixel is background-dominated and pulled down. Fine strands
   are exempted, and the pull is automatically weak where foreground /
   background colours barely separate (so pale hair on a pale wall, and white
   seams on white fabric, are not erased).

3. **strand decontamination** — after gaps are plugged the background field
   is cleaner, so a wider un-mix recovers the true colour inside fine
   structures (dark / coloured strands lose their grey wall wash).

Every stage leaves the solid body and the pure background bit-identical,
alpha is never raised, and the object interior is never eroded. General
purpose: hair, fur, feathers, foliage and thin object detail all benefit;
categories with razor edges (logo / document / cartoon) disable it via the
``hair_refine`` strategy flag.

Performance: the colour / structure analysis maps are computed once on a
capped work copy (never overtly larger than ``analysis_cap`` per side); only
the per-pixel alpha / colour decisions run at full resolution, so the stage
adds a few seconds even on 2K+ photos.
"""

import numpy as np
from PIL import Image

from ..utils import box_blur, to_pil
from ..matting.edge_detection import edge_zone, fg_similarity_lab
from .alpha_refine import _bg_field_robust, _reach, _texture_map
from ..color_ops import lab_distance

ANALYSIS_CAP = 2048  # Increased from 1280 for better high-res hair detail


def refine_edges(image_rgb, alpha, strategy):
    """Return ``(decontaminated_rgb, refined_alpha)``.

    ``image_rgb``: HxWx3 float32 RGB (0..255, ideally already decontaminated),
    ``alpha``: float32 0..1 at the same size. Gated by
    ``strategy['hair_refine']`` (default True — off for razor-edge
    categories). Interior pixels (a >= 0.98) are bit-identical on output.
    """
    cfg = strategy or {}
    rgb = np.asarray(image_rgb, dtype=np.float32)
    a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)

    if not cfg.get('hair_refine', True):
        return np.clip(rgb, 0.0, 255.0), a
    if cfg.get('color_mode'):
        return np.clip(rgb, 0.0, 255.0), a
    if not (a > 0.02).any() or not (a < 0.97).any():
        return np.clip(rgb, 0.0, 255.0), a

    m = _analysis(rgb, a, int(cfg.get('analysis_cap', ANALYSIS_CAP)), cfg)

    # 1 — plug background-coloured holes between fine structures.
    a = _plug_gaps(a, m['bg_sim'], m['fg_sim'], m['bg_reach'], m['fine'],
                   m['interior_flat'], cfg)

    # 2 — rim spill re-projected from the colour evidence (contour band only).
    a = _rim_pull(rgb, a, m, cfg)

    # 3 — deeper strand colour recovery, once the background field is clean.
    rgb = _strand_decontam(rgb, a, m, cfg)

    return np.clip(rgb, 0.0, 255.0), np.clip(a, 0.0, 1.0)


# ---------------------------------------------------------------------- #
# analysis maps (computed on a capped work copy, applied at full res)
# ---------------------------------------------------------------------- #
def _analysis(rgb, a, cap, cfg):
    """Return the per-pixel maps needed by the refinement stages.

    All heavy colour / structure analysis runs at ``cap`` resolution; the
    maps are then upscaled to the full size so every *decision* lands on the
    true pixels. Boolean structure maps are upscaled NEAREST (hard structure
    flags), continuous colour-similarity maps with LANCZOS.
    """
    rgb_w, a_w, s = _to_work(rgb, a, cap)
    full = (h, w) = a.shape
    work = s != 1.0

    lum_w = _lum(rgb_w)
    tex_w = np.clip(_texture_map(lum_w), 0.0, 1.0)

    # Flat *interior* surfaces (a garment, sheer fabric, a smooth product
    # face). The window statistic says "flat" both for the interior of a
    # shirt and for the exterior background right next to the silhouette, so
    # it is AND-ed with "not near any confident-background pixel": a garment
    # interior sits a few px away from the transparent region, while a
    # genuine background-gap *between* strands is itself background-coloured
    # and background-adjacent. This is the anti-regression gate for "white
    # clothes on a bright backdrop": the shirt core is protected, the wall
    # holes and the rim keep their fix.
    fr = int(cfg.get('flat_radius', 8) * s) if work else int(cfg.get('flat_radius', 8))
    fc = int(cfg.get('flat_clearance', 6) * s) if work else int(cfg.get('flat_clearance', 6))
    flat_w = _flat_surface(tex_w, max(1, fr))
    near_bg_w = _reach((a_w < 0.35), max(1, fc))
    interior_flat_w = flat_w & ~near_bg_w

    fg_sim_w = np.clip(fg_similarity_lab(rgb_w, a_w, radius=7), 0.0, 1.0)
    B_w, reach_w = _bg_field_robust(rgb_w, a_w, radius=7)
    bg_sim_w = _bg_similarity(rgb_w, B_w)

    gap_reach = int(cfg.get('gap_reach', 7) * s) if work else int(cfg.get('gap_reach', 7))
    near_solid_w = _reach((a_w >= 0.5), max(1, gap_reach))

    # Robust foreground colour (multi-erosion average) + a smooth background
    # field, both for the rim spill re-projection at full resolution.
    F_w = _robust_fg(rgb_w, a_w, s if work else 1.0)
    try:
        lab_ev_w = np.clip((lab_distance(rgb_w, F_w) - 40.0) / 140.0, 0.0, 1.0)
    except Exception:
        lab_ev_w = np.zeros_like(bg_sim_w)

    if not work:
        return {
            'tex': tex_w, 'fine': tex_w > float(cfg.get('fine_thr', 0.5)),
            'interior_flat': interior_flat_w, 'fg_sim': fg_sim_w,
            'bg_sim': bg_sim_w, 'bg_reach': reach_w, 'near_solid': near_solid_w,
            'B': B_w, 'F': F_w, 'lab_ev': lab_ev_w,
        }

    return {
        'tex': _up_f(tex_w, full),
        'fine': (_up_b(tex_w > float(cfg.get('fine_thr', 0.5)), full)),
        'interior_flat': _up_b(interior_flat_w, full),
        'fg_sim': _up_f(fg_sim_w, full),
        'bg_sim': _up_f(bg_sim_w, full),
        'bg_reach': _up_b(reach_w, full),
        'near_solid': _up_b(near_solid_w, full),
        'B': _up3(B_w, full),
        'F': _up3(F_w, full),
        'lab_ev': _up_f(lab_ev_w, full),
    }


# ---------------------------------------------------------------------- #
# stage 1 — background-gap plug
# ---------------------------------------------------------------------- #
def _plug_gaps(a, bg_sim, fg_sim, bg_reach, fine, interior_flat, cfg):
    """Pull background-coloured semi-transparent *holes* toward alpha 0.

    A candidate hole must hug the foreground (``near_solid``), have a real
    background sample within reach, and be LAB-similar to the local background
    while clearly *disagreeing* with the local foreground. The foreground
    disagreement is what keeps real structure safe: a hair strand or a
    garment edge matches the local foreground estimate (high ``fg_sim``) and
    is never a candidate, no matter how pale. Deep *interior* flat surfaces
    (garments, sheer fabric, smooth objects) are excluded outright — the
    anti-regression gate for white clothing on a bright backdrop. Pixels on
    strong fine structure need a much stronger background signal before they
    are touched, so a fly-away pale strand survives. Confident holes are
    removed outright; ambiguous ones are pulled continuously.
    """
    min_keep = float(cfg.get('gap_min_a', 0.02))  # Lowered from 0.03
    cap = float(cfg.get('gap_alpha_cap', 0.75))  # Lowered from 0.85
    guard = (_reach(a >= 0.5, int(cfg.get('gap_reach', 10))) & bg_reach  # Increased reach from 7 to 10
             & (a > min_keep) & (a < cap) & ~interior_flat)

    cand_fine = (fine & (bg_sim > float(cfg.get('gap_bg_hi', 0.70)))  # Increased from 0.66
                 & (fg_sim < float(cfg.get('gap_fg_hi_fine', 0.25))))  # Lowered from 0.30
    cand_flat = (~fine & (bg_sim > float(cfg.get('gap_bg_lo', 0.60)))  # Increased from 0.55
                 & (fg_sim < float(cfg.get('gap_fg_hi', 0.48))))  # Lowered from 0.52
    cand = guard & (cand_fine | cand_flat)
    if not cand.any():
        return a

    s = (np.clip((bg_sim - 0.45) / 0.30, 0.0, 1.0)  # Adjusted
         * np.clip((0.58 - fg_sim) / 0.40, 0.0, 1.0))  # Adjusted
    out = a * (1.0 - 0.8 * (s * cand))  # Reduced from 0.9

    # Real fine structure (a physical strand, not a flat gap) must survive:
    # where the pixel sits on strong high-frequency texture the pull is
    # nearly disabled, otherwise sparse strands whose local fg average is
    # dominated by the background would be wiped as "background gaps".
    fine_protect = np.clip(np.asarray(fine, dtype=np.float32), 0.0, 1.0) > 0.5
    s = s * (1.0 - 0.85 * fine_protect)  # Increased protection from 0.8

    hard = cand & (bg_sim > 0.78) & (fg_sim < 0.18) & (a < 0.70)  # Stricter
    if hard.any():
        out = np.where(hard, 0.0, out)
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------- #
# stage 2 — rim spill re-projection
# ---------------------------------------------------------------------- #
def _rim_pull(rgb, a, m, cfg):
    """Lower the alpha of the background-dominated outer shell.

    Colour evidence is re-projected against the robust foreground field ``F``
    and the smooth background field ``B`` (both precomputed): a pixel whose
    implied alpha sits far below its prediction is background-dominated and
    pulled down. The F/B separation is judged in RGB *and* Lab so the gate is
    perceptually reliable; where the two barely separate (pale hair on a pale
    wall, white seams on white fabric) the evidence is ~0 and nothing moves.
    Fine strands, whose soft alpha is the cleaner signal, are exempted; their
    colour is recovered by the decontamination stages.
    """
    a = np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0)
    rgb = np.asarray(rgb, dtype=np.float32)
    F = np.asarray(m['F'], dtype=np.float32)
    Bgg = np.asarray(m['B'], dtype=np.float32)
    fine = np.asarray(m['fine'], dtype=bool)

    zone = edge_zone(a, thresh=0.03,
                     dilation=int(cfg.get('rim_dilation', 3)))
    if not zone.any():
        return a

    fb = F - Bgg
    denom = (fb * fb).sum(axis=2) + 1e-4
    a_imp = np.clip(((rgb - Bgg) * fb).sum(axis=2) / denom, 0.0, 1.0)

    ev_n = np.clip((np.sqrt((fb * fb).sum(axis=2)) - 30.0) / 90.0, 0.0, 1.0)
    ev_n = np.maximum(ev_n, np.asarray(m['lab_ev'], dtype=np.float32))

    radius = int(cfg.get('spill_radius', 7) or 7)
    support = box_blur((a < 0.35).astype(np.float32), radius) > 0.02
    cond = zone & (a > float(cfg.get('rim_min_a', 0.08))) & support \
        & (a_imp < a - 0.08)
    if not cond.any():
        return a

    pull = np.clip((a - a_imp) / 0.25, 0.0, 1.0)
    w = float(cfg.get('rim_strength', 0.9)) * ev_n * pull * 0.8
    out_c = a * (1.0 - w) + a_imp * w
    out = np.where(cond, np.minimum(out_c, a), a)
    out = np.where(fine, a, out)
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------- #
# stage 3 — strand colour recovery
# ---------------------------------------------------------------------- #
def _strand_decontam(rgb, a, m, cfg):
    """Wider-radius un-mix confined to fine structures.

    The pipeline's main decontamination already recovered the rim colours;
    inside a thick hair / fur mass its edge-aware (guided) background field
    is still partly strand-colour. Re-un-mixing every fine-structure pixel
    against the clean precomputed background field gives dark strands back
    their real hue instead of a grey wall wash. Only pixels with a real
    background sample within reach are touched.
    """
    a = np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0)
    rgb = np.asarray(rgb, dtype=np.float32)
    band = (a > float(cfg.get('strand_min_a', 0.05))) & (a < 0.98)
    focus = (band & (np.asarray(m['tex'], dtype=np.float32)
                     > float(cfg.get('strand_tex', 0.25)))
             & np.asarray(m['bg_reach'], dtype=bool))
    if not focus.any():
        return rgb

    B = np.asarray(m['B'], dtype=np.float32)
    a_safe = np.maximum(a, 0.06)[..., None]
    F = np.clip((rgb - (1.0 - a)[..., None] * B) / a_safe, 0.0, 255.0)
    w = (float(cfg.get('strand_strength', 0.65))
         * focus.astype(np.float32))[..., None]
    out = rgb * (1.0 - w) + F * w
    body = (a >= 0.98)[..., None]
    out = np.where(body, rgb, out)
    return np.clip(out, 0.0, 255.0)


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def _robust_fg(rgb, a, s):
    """Foreground colour estimate robust to rim contamination: averaged over
    several erosion radii so one bad boundary sample cannot dominate."""
    from ..utils import erode
    from .spill_correction import fg_estimate
    base = (a > 0.9).astype(np.float32)
    masks = []
    if base.mean() > 0.01:
        for k in (3, 5, 7):
            kk = max(3, int(k * s) // 2 * 2 + 1)
            m = (erode(base * 255.0, kk) > 0).astype(np.float32)
            if m.mean() > 0.0:
                masks.append(m)
    if not masks:
        masks.append(base if base.mean() > 0.0 else (a > 0.9).astype(np.float32))
    acc = None
    for mk in masks:
        f = fg_estimate(rgb, mk)
        acc = f if acc is None else acc + f
    return acc / len(masks)


def _bg_similarity(rgb, B):
    """0..1 how close each pixel's colour is to the local background field."""
    try:
        d = lab_distance(rgb, B)
    except Exception:
        d = np.sqrt(((rgb - B) ** 2).sum(axis=2))
    return 1.0 - np.clip((d - 10.0) / 50.0, 0.0, 1.0)


def _flat_surface(tex, radius=8, low_thr=0.4, cover=0.72):
    """True where a *wide* local window is almost entirely low-texture.

    A flat surface (garment, fabric, smooth object) is low-frequency over the
    whole window; a wall hole *inside* hair is surrounded by strand edges and
    therefore reads textured at this window size.
    """
    r = max(1, int(radius))
    low = (np.asarray(tex, dtype=np.float32) < low_thr).astype(np.float32)
    return box_blur(low, r) > cover


def _to_work(rgb, a, cap):
    """Downscale the analysis copies (maps only, never the decisions)."""
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


def _up_f(v, size):
    """Upscale a float 0..1 map to ``size`` (BILINEAR — maps are smooth)."""
    h, w = size
    small = to_pil(np.clip(v, 0.0, 1.0) * 255.0)
    return np.asarray(small.resize((w, h), Image.BILINEAR),
                      dtype=np.float32) / 255.0


def _up_b(v, size):
    """Upscale a boolean map to ``size`` with NEAREST (structure flag)."""
    h, w = size
    small = to_pil(np.asarray(v, dtype=np.uint8) * 255)
    return np.asarray(small.resize((w, h), Image.NEAREST)) > 0


def _up3(v, size):
    """Upscale an HxWx3 colour field to ``size`` (BILINEAR — smooth fields)."""
    h, w = size
    small = Image.fromarray(np.clip(v, 0, 255).astype(np.uint8), 'RGB')
    return np.asarray(small.resize((w, h), Image.BILINEAR), dtype=np.float32)


def _lum(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


__all__ = ['refine_edges']