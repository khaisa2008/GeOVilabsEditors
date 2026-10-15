"""High-quality hair/fur alpha matting + color defringing using PyMatting / OpenCV.

This module provides a post-processing step that specifically addresses:
1. **Alpha Matting Refinement** — using PyMatting's Closed-Form (CF) or
   K-Nearest-Neighbour (KN) matting with adaptive trimap and fine-hair threshold
   to recover thin, curly/wavy strands without halo bleeding.
2. **Color Defringing / Decontamination** — removes residual white/grey background
   colour from semi-transparent edge pixels (the "white halo" problem) via
   Lab-space chroma correction and foreground colour recovery.

If PyMatting/OpenCV are not installed, it falls back to the existing numpy/PIL
guided-filter implementation (see `alpha_matting.py`, `decontamination.py`).

Usage:
    from background_remover.postprocess.hair_matting import refine_hair_matte

    # image_rgb: HxWx3 uint8 or float32 (0..255)
    # alpha: HxW float32 (0..1) or uint8 (0..255)
    # strategy: dict with optional keys below
    rgb_out, alpha_out = refine_hair_matte(image_rgb, alpha, strategy)

Strategy keys (all optional):
    - use_pymatting: bool (default True) — try PyMatting first
    - matting_method: 'cf' | 'kn' (default 'cf') — PyMatting method
    - hair_threshold: float (default 0.15) — alpha threshold below which
      pixels are considered "fine hair" and get wider trimap band
    - trimap_dilate: int (default 15) — base dilation for unknown band
    - trimap_erode: int (default 15) — base erosion for known fg/bg
    - defringe_strength: float (default 0.8) — color decontamination strength
    - defringe_radius: int (default 8) — background estimation radius
    - halo_removal: bool (default True) — enable white/grey halo suppression
    - feather_px: float (default 0.5) — light feather on transition band
    - max_work_px: int (default 2048) — downscale cap for matting speed
"""

import numpy as np
from PIL import Image, ImageFilter
from typing import Optional, Tuple, Dict, Any

# --- Optional dependencies ---
_HAS_PYMATTING = False
_HAS_CV2 = False

try:
    import pymatting
    _HAS_PYMATTING = True
except Exception:
    pass

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    pass

# --- Local imports ---
from ..utils import box_blur, box_blur_3d, erode, dilate, guided_filter, to_pil
from ..matting.trimap import generate_trimap
from ..matting.edge_detection import edge_zone, fg_bg_contrast
from ..color_ops import rgb_to_lab, lab_to_rgb, lab_distance


# =============================================================================
# Public API
# =============================================================================

def refine_hair_matte(
    image_rgb: np.ndarray,
    alpha: np.ndarray,
    strategy: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Refine alpha matte for hair/fur and remove color fringes.

    Args:
        image_rgb: HxWx3 RGB image (uint8 0..255 or float32 0..255).
        alpha: HxW alpha mask (float32 0..1 or uint8 0..255).
        strategy: Optional dict with tuning parameters (see module docstring).

    Returns:
        (decontaminated_rgb, refined_alpha) both float32, rgb 0..255, alpha 0..1.
    """
    cfg = strategy or {}
    rgb = _ensure_float_rgb(image_rgb)
    a = _ensure_float_alpha(alpha)

    # Optional original RGB for Lab chroma correction (pulls toward true source colors)
    original_rgb = cfg.get('original_rgb')
    if original_rgb is not None:
        original_rgb = _ensure_float_rgb(original_rgb)

    # Early exit if no semi-transparent pixels
    if not ((a > 0.01) & (a < 0.99)).any():
        return np.clip(rgb, 0.0, 255.0), np.clip(a, 0.0, 1.0)

    # 1. Alpha matting refinement (PyMatting or fallback)
    a = _refine_alpha_matting(rgb, a, cfg)

    # 2. Color defringing / decontamination (white halo removal)
    # Pass original_rgb for accurate Lab chroma correction toward true source colors
    rgb, a = _defringe_color(rgb, a, cfg, original_rgb=original_rgb)

    # 3. Optional light feather on transition band
    a = _feather_transition(a, cfg)

    return np.clip(rgb, 0.0, 255.0), np.clip(a, 0.0, 1.0)


# =============================================================================
# Stage 1: Alpha Matting Refinement (PyMatting + adaptive trimap)
# =============================================================================

def _refine_alpha_matting(
    rgb: np.ndarray,
    alpha: np.ndarray,
    cfg: Dict[str, Any],
) -> np.ndarray:
    """Refine alpha using PyMatting (CF/KN) with hair-aware adaptive trimap."""
    use_pymatting = cfg.get('use_pymatting', True) and _HAS_PYMATTING
    method = cfg.get('matting_method', 'cf')  # 'cf' or 'kn'
    max_work = int(cfg.get('max_work_px', 2048))

    h, w = alpha.shape
    scale = 1.0
    if max(h, w) > max_work:
        scale = max_work / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        rgb_small = np.asarray(
            Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), 'RGB')
            .resize((nw, nh), Image.BILINEAR), dtype=np.float32)
        a_small = np.asarray(
            to_pil(alpha * 255.0).resize((nw, nh), Image.BILINEAR),
            dtype=np.float32) / 255.0
    else:
        rgb_small, a_small = rgb, alpha

    # Generate hair-aware trimap
    trimap = _build_hair_trimap(a_small, rgb_small, cfg)

    if use_pymatting:
        try:
            a_refined = _pymatting_refine(rgb_small, a_small, trimap, method, cfg)
            if scale != 1.0:
                a_refined = _upscale_alpha(a_refined, h, w)
            return a_refined
        except Exception:
            # Fall through to numpy fallback
            pass

    # Fallback: local guided-filter matting (existing implementation)
    from ..matting.alpha_matting import refine_alpha as _local_refine
    return _local_refine(
        rgb, alpha,
        trimap=trimap if scale == 1.0 else None,
        radius=int(cfg.get('matting_radius', 6)),
        downscale=max_work,
        iterations=int(cfg.get('matting_iterations', 2)),
        adaptive=True,
        cfg=cfg,
    )


def _build_hair_trimap(
    alpha: np.ndarray,
    rgb: np.ndarray,
    cfg: Dict[str, Any],
) -> np.ndarray:
    """Build adaptive trimap with precise unknown band for fine hair.

    Strategy:
    1. Start with adaptive trimap from generate_trimap (contrast-aware)
    2. Refine unknown band using fine-hair detection (low alpha + high gradient)
    3. Constrain unknown band to regions with real background support
    """
    hair_thr = float(cfg.get('hair_threshold', 0.15))  # Increased from 0.10 to capture more semi-transparent hair
    base_dilate = int(cfg.get('trimap_dilate', 12))    # Increased from 8 for wider unknown band
    base_erode = int(cfg.get('trimap_erode', 8))       # Reduced from 10 to preserve more fg detail
    max_radius = int(cfg.get('trimap_max_radius', 24)) # Increased from 20

    # Standard adaptive trimap (contrast-aware erosion/dilation)
    trimap, _, _, _ = generate_trimap(
        alpha,
        fg_erode=base_erode,
        bg_dilate=base_dilate,
        adaptive=True,
        rgb=rgb,
        cfg=cfg,
        max_radius=max_radius,
    )

    # Fine-hair mask: low alpha, high gradient, AND has background support
    if hair_thr > 0:
        grad = _alpha_gradient(alpha)

        # Background support: confident background (a < 0.3) within reach
        # Use larger radius (12 vs 8) to catch background between tight curls
        bg_support = box_blur((alpha < 0.3).astype(np.float32), 12) > 0.03  # Lowered threshold

        # Also detect hair by texture (Laplacian) in addition to alpha gradient
        # box_blur is already imported at module level from ..utils
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        from scipy.ndimage import laplace
        tex = np.abs(laplace(gray))
        tex_norm = tex / (tex.max() + 1e-6)
        high_tex = tex_norm > 0.15  # Fine structure texture

        hair_mask = (
            (alpha > 0.01) & (alpha < hair_thr) &  # semi-transparent (lowered from 0.02)
            ((grad > 0.03) | high_tex) &           # high gradient OR high texture (lowered from 0.04)
            bg_support                              # real background nearby
        )

        if hair_mask.any():
            # Expand unknown band ONLY into hair regions with bg support
            hair_dilate_k = max(5, int(base_dilate * 1.5) | 1)  # Increased dilation
            hair_unknown = dilate((hair_mask.astype(np.uint8)) * 255, hair_dilate_k) > 0

            # Only convert bg (0) or fg (255) to unknown (128) where hair_mask exists
            trimap[hair_unknown & (trimap == 0)] = 128
            trimap[hair_unknown & (trimap == 255)] = 128

    # Safety: ensure we have both known fg and bg
    if not (trimap == 255).any():
        trimap[alpha > 0.95] = 255
    if not (trimap == 0).any():
        trimap[alpha < 0.05] = 0

    return trimap


def _pymatting_refine(
    rgb: np.ndarray,
    alpha: np.ndarray,
    trimap: np.ndarray,
    method: str,
    cfg: Dict[str, Any],
) -> np.ndarray:
    """Run PyMatting Closed-Form or KNN matting."""
    # PyMatting expects trimap as 0/128/255 uint8, image as float32 0..1
    img = np.clip(rgb / 255.0, 0.0, 1.0).astype(np.float32)
    tri = trimap.astype(np.uint8)

    if method == 'kn':
        # KNN matting - faster, good for hair
        from pymatting import estimate_alpha_knn
        alpha_refined = estimate_alpha_knn(
            img, tri,
            n_neighbors=int(cfg.get('knn_neighbors', 10)),
            laplacian_radius=int(cfg.get('knn_laplacian_radius', 1)),
            epsilon=float(cfg.get('knn_epsilon', 1e-5)),
        )
    else:
        # Closed-Form matting - higher quality, slower
        from pymatting import estimate_alpha_cf
        alpha_refined = estimate_alpha_cf(
            img, tri,
            epsilon=float(cfg.get('cf_epsilon', 1e-6)),
        )

    # Blend with original alpha based on confidence (trimap unknown region)
    unk = (trimap == 128)
    if unk.any():
        # Use local contrast as confidence - trust matting MORE where F/B separate
        from ..matting.edge_detection import fg_bg_contrast
        conf = fg_bg_contrast(rgb, alpha)
        conf = np.clip(conf, 0.0, 1.0)
        # Stronger trust in matting: 0.1 + 0.8 * conf (was 0.3 + 0.6 * conf)
        blend = 0.1 + 0.8 * conf
        alpha_refined = np.where(unk, blend * alpha_refined + (1 - blend) * alpha, alpha_refined)

    return np.clip(alpha_refined, 0.0, 1.0)


def _upscale_alpha(alpha_small: np.ndarray, h: int, w: int) -> np.ndarray:
    """Upscale alpha with LANCZOS for smooth edges."""
    a_img = to_pil(np.clip(alpha_small, 0.0, 1.0) * 255.0)
    a_img = a_img.resize((w, h), Image.LANCZOS)
    return np.asarray(a_img, dtype=np.float32) / 255.0


def _alpha_gradient(alpha: np.ndarray) -> np.ndarray:
    """Normalized alpha gradient magnitude (0..1)."""
    a = np.asarray(alpha, dtype=np.float32)
    gy, gx = np.gradient(a)
    mag = np.sqrt(gx * gx + gy * gy)
    hi = mag.max()
    return mag / hi if hi > 1e-6 else np.zeros_like(mag)


# =============================================================================
# Stage 2: Color Defringing / Decontamination (White Halo Removal)
# =============================================================================

def _defringe_color(
    rgb: np.ndarray,
    alpha: np.ndarray,
    cfg: Dict[str, Any],
    original_rgb: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove white/grey background color from semi-transparent edge pixels.

    This is the main "color defringing" step that eliminates the white halo:
    1. Estimate local background color field (edge-aware, from confident bg)
    2. Multi-radius support check: only decontaminate where real bg is nearby
    3. Un-mix foreground color: F = (I - (1-a)*B) / a
    4. Background-color match: delete pixels still bg-colored after un-mixing
    5. Lab-space chroma correction to remove residual color spill
    6. White/grey luminance halo suppression (alpha pull)
    """
    strength = float(cfg.get('defringe_strength', 0.9))      # Increased from 0.8
    radius = int(cfg.get('defringe_radius', 10))             # Increased from 8
    iterations = int(cfg.get('defringe_iterations', 4))      # Increased from 2
    halo_removal = cfg.get('halo_removal', True)

    a = np.clip(alpha, 0.0, 1.0)
    rgb_work = rgb.astype(np.float32).copy()

    # Base band: semi-transparent edge pixels
    min_a = float(cfg.get('defringe_min_a', 0.03))           # Lowered from 0.05
    band = (a >= min_a) & (a < 0.98)

    # For hair, expand band to include gaps between strands
    if cfg.get('hair_refine', True):
        wide_dilate = int(cfg.get('decontam_zone', 12) or 12)  # Increased from 8
        band = band | (edge_zone(a, thresh=0.02, dilation=wide_dilate) & (a > 0.005))  # Lowered thresh

    if not band.any():
        return rgb_work

    # Multi-radius background support check - MORE PERMISSIVE for hair
    # Only trust background estimate where confident bg (a < 0.4) is within reach
    support = np.zeros_like(a, dtype=np.float32)
    for r in (max(2, radius // 2), radius, min(radius * 2, 18)):  # Increased max radius
        support = np.maximum(
            support, (box_blur((a < 0.35).astype(np.float32), r) > 0.03)  # Lowered from 0.05
        )
    # Band restricted to regions with real background support
    supported_band = band & (support > 0)

    # Iterative decontamination
    for _ in range(max(1, iterations)):
        # Edge-aware background field estimation (uses existing decontamination module)
        from ..postprocess.decontamination import estimate_bg_field
        B = estimate_bg_field(rgb_work, a, radius, use_guided=True)

        # Un-mix: F = (I - (1-a)*B) / a
        # Use adaptive min_a - lower for hair regions
        adaptive_min_a = np.where(
            (a > 0.01) & (a < 0.2) & high_texture_mask(rgb_work, a),
            0.02, min_a
        )
        a_safe = np.maximum(a, adaptive_min_a)[..., None]
        F = (rgb_work - (1.0 - a)[..., None] * B) / a_safe
        F = np.clip(F, 0.0, 255.0)

        # Blend only where we have background support
        w = strength * supported_band.astype(np.float32)
        w = _edge_smooth(w, rgb_work, radius=3)
        w = w[..., None]

        rgb_work = rgb_work * (1.0 - w) + F * w
        # Protect solid interior (a > 0.98)
        body = (a > 0.98)[..., None]
        rgb_work = np.where(body, rgb, rgb_work)

    # Background-color match: after un-mixing, if pixel is still ~background color,
    # it has no foreground content (detached fringe, gaps) -> REMOVE it (alpha = 0)
    B_final = estimate_bg_field(rgb_work, a, radius, use_guided=True)
    diff_bg = np.sqrt(((rgb_work - B_final) ** 2).mean(axis=2))
    bg_like = supported_band & (diff_bg < 15.0) & (a > 0.01) & (a < 0.85)  # Tighter threshold
    if bg_like.any():
        # These pixels are essentially background color -> zero alpha
        a = np.where(bg_like, 0.0, a)
        # Also zero RGB in those regions (will be transparent)
        rgb_work = np.where(bg_like[..., None], 0.0, rgb_work)

    # Lab-space chroma correction (removes residual color spill)
    # Use updated alpha (with bg_like zeroed)
    # Use original_rgb as reference for true colors if available
    chroma_ref = original_rgb if original_rgb is not None else rgb
    rgb_work = _lab_chroma_correction(rgb_work, chroma_ref, a, supported_band, strength * 1.2, cfg)  # Boosted

    # White/grey luminance halo suppression (modifies alpha) - MORE AGGRESSIVE
    if halo_removal:
        # Use configurable halo_strength, default 1.2 (was 1.0)
        halo_strength = float(cfg.get('halo_strength', 1.2))
        # Pass the updated alpha with bg_like already zeroed
        rgb_work, a = _suppress_luminance_halo(
            rgb_work, a, supported_band,
            strength * 1.5 * halo_strength,  # Increased from 1.2
            radius, cfg
        )

    return np.clip(rgb_work, 0.0, 255.0), np.clip(a, 0.0, 1.0)


def high_texture_mask(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Detect high-texture regions (fine hair strands) for adaptive min_a."""
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    from scipy.ndimage import laplace
    tex = np.abs(laplace(gray))
    tex_norm = tex / (tex.max() + 1e-6)
    # High texture AND semi-transparent
    return (tex_norm > 0.1) & (alpha > 0.01) & (alpha < 0.3)


def _estimate_bg_field_guided(
    rgb: np.ndarray,
    alpha: np.ndarray,
    radius: int,
) -> np.ndarray:
    """Edge-aware background color field using guided filter (OpenCV or numpy)."""
    a = np.clip(alpha, 0.0, 1.0)
    w = (1.0 - a)  # background weight

    if _HAS_CV2:
        return _estimate_bg_field_cv2(rgb, a, w, radius)

    # Numpy fallback (from decontamination.py)
    bg_w = box_blur(w, radius) + 1e-4
    B = box_blur_3d(rgb * w[..., None], radius) / bg_w[..., None]
    B = np.clip(B, 0.0, 255.0)

    # Guided filter for edge-awareness
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    lum_g = lum / 255.0
    gr = max(2, radius)
    out = np.empty_like(B, dtype=np.float32)
    for c in range(3):
        out[..., c] = guided_filter(B[..., c], lum_g, radius=gr, eps=0.02)
    return np.clip(out, 0.0, 255.0)


def _estimate_bg_field_cv2(
    rgb: np.ndarray,
    alpha: np.ndarray,
    bg_weight: np.ndarray,
    radius: int,
) -> np.ndarray:
    """OpenCV-accelerated background field estimation."""
    # Use cv2.boxFilter for fast box blur (separable, O(1) per pixel)
    ksize = 2 * radius + 1
    bg_w = cv2.boxFilter(bg_weight.astype(np.float32), -1, (ksize, ksize),
                          normalize=True, borderType=cv2.BORDER_REFLECT) + 1e-4

    B = np.empty_like(rgb, dtype=np.float32)
    for c in range(3):
        ch = rgb[..., c] * bg_weight
        B[..., c] = cv2.boxFilter(ch.astype(np.float32), -1, (ksize, ksize),
                                   normalize=True, borderType=cv2.BORDER_REFLECT) / bg_w

    B = np.clip(B, 0.0, 255.0)

    # Guided filter using OpenCV (ximgproc) if available, else numpy
    try:
        import cv2.ximgproc as ximgproc
        guide = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        guide = guide.astype(np.float32) / 255.0
        for c in range(3):
            B[..., c] = ximgproc.guidedFilter(guide, B[..., c], radius, 0.02)
    except Exception:
        # Fallback to numpy guided_filter
        lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        lum_g = lum / 255.0
        for c in range(3):
            B[..., c] = guided_filter(B[..., c], lum_g, radius=max(2, radius), eps=0.02)

    return np.clip(B, 0.0, 255.0)


def _lab_chroma_correction(
    rgb_decont: np.ndarray,
    rgb_orig: np.ndarray,
    alpha: np.ndarray,
    band: np.ndarray,
    strength: float,
    cfg: Dict[str, Any],
) -> np.ndarray:
    """Lab-space chroma pull: move a*,b* toward original while keeping L*."""
    try:
        lab_cur = rgb_to_lab(np.clip(rgb_decont, 0, 255).astype(np.uint8))
        lab_src = rgb_to_lab(np.clip(rgb_orig, 0, 255).astype(np.uint8))
    except Exception:
        return rgb_decont

    La, Aa, Ba = lab_cur[..., 0], lab_cur[..., 1], lab_cur[..., 2]
    Ls, As, Bs = lab_src[..., 0], lab_src[..., 1], lab_src[..., 2]

    # Weight: only in band, scaled by strength - INCREASED from 0.6 to 0.8
    w = strength * 0.8 * band.astype(np.float32)
    w = _edge_smooth(w, rgb_decont, radius=3)

    # Pull chroma (a*, b*) toward source; keep luminance (L*) from decontaminated
    A_corr = Aa + (As - Aa) * w
    B_corr = Ba + (Bs - Ba) * w
    lab2 = np.stack([La, A_corr, B_corr], axis=-1)

    try:
        rgb_corr = lab_to_rgb(lab2)
        out = rgb_decont * (1.0 - w[..., None]) + rgb_corr * w[..., None]
        # Protect solid body
        body = (alpha > 0.98)[..., None]
        out = np.where(body, rgb_decont, out)
        return np.clip(out, 0.0, 255.0)
    except Exception:
        return rgb_decont


def _suppress_luminance_halo(
    rgb: np.ndarray,
    alpha: np.ndarray,
    band: np.ndarray,
    strength: float,
    radius: int,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Suppress bright neutral/white fringe (the classic white halo).

    Detects pixels that are:
    - Significantly brighter than local solid foreground
    - Low chroma (neutral/grey)
    - In the transition band
    """
    a = np.clip(alpha, 0.0, 1.0)
    rgb = np.asarray(rgb, dtype=np.float32)

    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    # Solid interior reference (eroded so halo doesn't pollute it)
    solid = (a >= 0.985).astype(np.float32)
    interior = box_blur(solid, max(1, radius // 2)) > 0.999
    k_erode = max(3, 2 * (radius // 2) + 1)
    if _HAS_CV2:
        interior = cv2.erode((interior.astype(np.uint8)) * 255,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_erode, k_erode))) > 0
    else:
        interior = erode((interior.astype(np.uint8)) * 255, k_erode) > 0

    if not interior.any():
        interior = a > 0.9

    interior_f = interior.astype(np.float32)
    fw = box_blur(interior_f, radius) + 1e-4
    fg_lum = box_blur(lum * interior_f, radius) / fw

    # Brightness lift relative to local foreground - MORE SENSITIVE
    lift = lum - fg_lum
    fg_lum_lo = float(np.percentile(fg_lum[interior], 20)) if interior.any() else 40.0
    base = np.clip(fg_lum_lo * 0.08, 4.0, 15.0)  # Reduced from 0.12/6-20 to 0.08/4-15
    lift_n = np.clip((lift - base) / (base * 4.0), 0.0, 1.0)  # Increased sensitivity

    # Chroma gate: only suppress neutral/white fringes - WIDER GATE
    mx = np.maximum(np.abs(rgb[..., 0] - rgb[..., 1]),
                     np.abs(rgb[..., 1] - rgb[..., 2]))
    grey = 1.0 - np.clip(mx / 50.0, 0.0, 1.0)  # Increased from 70 to 50
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    chroma_w = np.clip((30.0 - chroma) / 20.0, 0.0, 1.0)  # Tighter from 40/25 to 30/20

    # Also detect DARK halos (background darker than foreground)
    dark_lift = fg_lum - lum
    dark_base = np.clip(fg_lum_lo * 0.08, 4.0, 15.0)
    dark_lift_n = np.clip((dark_lift - dark_base) / (dark_base * 4.0), 0.0, 1.0)
    dark_halo_w = dark_lift_n * grey * chroma_w * band.astype(np.float32)

    halo_weight = (lift_n + dark_lift_n) * grey * chroma_w * band.astype(np.float32)
    halo_weight = _edge_smooth(halo_weight, rgb, radius=3)

    # Reduce alpha on confirmed white/dark halo pixels
    a_out = a.copy()
    scale = 1.0 - strength * 0.95 * halo_weight  # Increased from 0.9
    a_out = np.clip(a_out * scale, 0.0, 1.0)

    # Hard removal for extreme halo
    hard = halo_weight > 0.7  # Lowered from 0.8
    if hard.any():
        a_out[hard] = 0.0

    return rgb, a_out


def _edge_smooth(weight: np.ndarray, guide_rgb: np.ndarray, radius: int = 3) -> np.ndarray:
    """Edge-aware smoothing of a weight map using guided filter."""
    try:
        lum = 0.299 * guide_rgb[..., 0] + 0.587 * guide_rgb[..., 1] + 0.114 * guide_rgb[..., 2]
        lum = np.clip(lum, 0.0, 255.0) / 255.0
        if _HAS_CV2:
            try:
                import cv2.ximgproc as ximgproc
                return ximgproc.guidedFilter(lum, weight.astype(np.float32), radius, 0.06)
            except Exception:
                pass
        return guided_filter(weight.astype(np.float32), lum, radius=radius, eps=0.06)
    except Exception:
        return weight


def _feather_transition(alpha: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """Light feather on the alpha transition band only."""
    feather = float(cfg.get('feather_px', 0.5) or 0.0)
    if feather <= 0:
        return alpha

    a = np.clip(alpha, 0.0, 1.0)
    band = (a > 0.02) & (a < 0.98)
    if not band.any():
        return a

    # Gaussian blur only the transition band
    if _HAS_CV2:
        blurred = cv2.GaussianBlur((a * 255.0).astype(np.float32), (0, 0), feather)
        blurred = blurred / 255.0
    else:
        from PIL import ImageFilter
        blurred = np.asarray(
            to_pil(a * 255.0).filter(ImageFilter.GaussianBlur(feather)),
            dtype=np.float32) / 255.0

    # Blend only in band
    w = band.astype(np.float32)
    w = _edge_smooth(w, np.stack([a]*3, axis=-1) * 255.0, radius=3)
    out = a * (1.0 - w) + blurred * w
    return np.clip(out, 0.0, 1.0)


# =============================================================================
# Helpers
# =============================================================================

def _ensure_float_rgb(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32)
    return np.clip(arr.astype(np.float32), 0.0, 255.0)


def _ensure_float_alpha(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return np.clip(arr.astype(np.float32) / 255.0, 0.0, 1.0)
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


# =============================================================================
# Convenience: standalone function for direct use on files
# =============================================================================

def process_image_file(
    input_path: str,
    output_path: str,
    strategy: Optional[Dict[str, Any]] = None,
    background: str = 'transparent',
) -> Tuple[int, int]:
    """Convenience: load image, run hair matting refinement, save RGBA PNG.

    This can be used as a standalone post-process on an already-removed background.
    """
    from background_remover import get_pipeline
    from background_remover.compositing import parse_background, render_background

    img = Image.open(input_path)
    img = Image.ImageOps.exif_transpose(img) if hasattr(Image, 'ImageOps') else img
    img_rgb = img.convert('RGB')

    # If input already has alpha, use it; else run full pipeline first
    if img.mode == 'RGBA':
        arr = np.asarray(img, dtype=np.float32)
        rgb = arr[..., :3]
        alpha = arr[..., 3] / 255.0
    else:
        # Run the full pipeline to get initial alpha
        from background_remover.pipeline import BackgroundPipeline
        pipe = BackgroundPipeline()
        # We need a temp file for the pipeline... simpler: use get_pipeline().remove()
        # but that writes to disk. Instead, let's just return an error.
        raise ValueError("Input must be RGBA (already background-removed). "
                         "Use the main pipeline for end-to-end removal.")

    rgb_out, alpha_out = refine_hair_matte(rgb, alpha, strategy)

    # Composite
    cut_src = Image.fromarray(np.clip(rgb_out, 0, 255).astype(np.uint8), 'RGB')
    cutout = cut_src.convert('RGBA')
    cutout.putalpha(to_pil(alpha_out * 255.0))

    mode, param = parse_background(background)
    from background_remover.compositing import render_background
    final = render_background(cutout, mode, param, original=img_rgb)
    final.save(output_path, 'PNG')
    return final.size


__all__ = ['refine_hair_matte', 'process_image_file']