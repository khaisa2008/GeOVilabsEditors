"""The adaptive background-removal pipeline.

Orchestrates the whole engine for one image:

    Image
    → Segmentation
    → Stable Alpha Calibration
    → Boundary/Edge Detection
    → Adaptive Trimap
    → Alpha Matting / Edge Refinement
    → Foreground Decontamination
    → Halo Removal            (quality-gated, may run extra rounds)
    → Final RGBA PNG

The decontaminated RGB is carried all the way to compositing (previously it
was computed and then dropped — the main reason white/grey fringes survived):
the cutout pixels on the edge are the *recovered* foreground colours, so the
final PNG contains no residual background colour.

Everything accepts a ``progress_cb(pct, msg)`` callback so the Flask layer
can stream progress; the callback is also our cancellation hook (the
caller raises on cancel, which unwinds naturally).
"""

import os
import time

import numpy as np
from PIL import Image, ImageOps

from .categories import (
    HUMAN, ANIMAL, PLANT, PRODUCT, LOGO, CARTOON, DOCUMENT,
    FOOD, VEHICLE, MIXED, CATEGORY_LABELS, strategy_for,
)
from .classifier.image_classifier import Classification

# --- resolution / tiling constants ---------------------------------------
SINGLE_CAP = 2560          # single-shot segmentation up to this long side
PASS2_CAP = 3200           # a second pass may afford a higher resolution
TILE_SIZE = 1600           # tiled path tile size
TILE_OVERLAP = 192         # overlap between tiles (feathered seams)
QUALITY_RETRY = 0.55       # below this → attempt PASS 2
HALO_GATE = 0.35           # halo severity above this → shown in the result message


class BackgroundPipeline:
    """Process-wide background-removal orchestrator (stateless workers)."""

    def __init__(self):
        self._birefnet = None
        self._birefnet_portrait = None
        self._classifier = None

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def remove(self, in_path, out_path, background='transparent',
               method='ai', color='#ffffff', tolerance=30,
               category=None, max_passes=2, add_shadow=False,
               custom_image=None, progress_cb=None):
        """Process one image end-to-end. Returns a ``PipelineResult``."""
        from . import PipelineResult

        t0 = time.time()
        self._call(progress_cb, 4, 'Loading image...')
        img = self._load(in_path)

        if category:
            cls = Classification(category, 1.0)
            self._call(progress_cb, 8, f'Using {CATEGORY_LABELS.get(category, category).lower()} mode')
        else:
            cls = self.classify_image(img, progress_cb)
        strategy = strategy_for(cls.category)
        if method == 'color':
            # By-colour removal must stay faithful to the distance-based
            # selection: no alpha calibration, no matting, no halo passes.
            # Those AI-edge tools smudge clean flat-colour cutouts.
            strategy = dict(strategy, color_mode=True)

        passes = 0
        quality, details = 0.0, {}
        alpha = None
        decont_rgb = None
        used_method = 'color' if method == 'color' else 'ai'
        cap = SINGLE_CAP

        while passes < max(1, max_passes):
            passes += 1
            self._call(progress_cb, 25 + (passes - 1) * 12,
                       f'Segmenting (pass {passes})...')
            alpha = self._segment_pass(
                img, cls.category, strategy, method, color, tolerance,
                cap=cap, progress_cb=progress_cb)

            self._call(progress_cb, 60 + (passes - 1) * 10,
                       'Refining edges...')
            decont_rgb, alpha = self._refine_alpha(
                img, alpha, cls.category, strategy,
                progress_cb, pass_number=passes)

            self._call(progress_cb, 78 + (passes - 1) * 8, 'Scoring quality...')
            img_rgb = np.asarray(img, dtype=np.float32)
            work = _decimate_rgb(img_rgb)
            work_a = _decimate_alpha(alpha)
            quality, details = _score(work_a, work)

            if (passes >= max_passes
                    or quality >= QUALITY_RETRY
                    or method == 'color'
                    or max(img.size) > SINGLE_CAP):
                break
            cap = PASS2_CAP
            self._call(progress_cb, 20,
                       f'Quality {quality:.2f} — running pass {passes + 1}...')

        out = self._compose(img, alpha, background, add_shadow, custom_image,
                            progress_cb, rgb=decont_rgb)
        out.save(out_path, 'PNG')

        elapsed = time.time() - t0
        halo = float(details.get('halo', 0.0))
        message = (f'{CATEGORY_LABELS.get(cls.category, cls.category)} - '
                   f'{_fmt_time(elapsed)} - quality {quality:.2f}'
                   + (f' - halo {halo:.2f}' if halo >= HALO_GATE else '')
                   + (f' (retried)' if passes > 1 else ''))
        return PipelineResult(
            category=cls.category,
            confidence=cls.confidence,
            quality_score=quality,
            passes=passes,
            method=used_method,
            width=img.size[0],
            height=img.size[1],
            processing_time=elapsed,
            message=message,
        )

    def classify_image(self, image, progress_cb=None):
        """Run the heuristic classifier, returning a Classification."""
        self._call(progress_cb, 9, 'Analyzing content...')
        try:
            from .classifier.image_classifier import classify as _classify
            return _classify(image)
        except Exception:
            return Classification(MIXED, 0.0)

    def classify_file(self, path):
        """Classify an image on disk (used by the /api/classify-image route)."""
        img = self._load(path)
        return self.classify_image(img).as_dict()

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _load(self, path):
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        return img.convert('RGB')

    def _segment_pass(self, image, category, strategy, method, color,
                      tolerance, cap, progress_cb):
        """Segmentation for one pass → float32 alpha 0..1 at image size."""
        w, h = image.size
        if method == 'color':
            from .segmentation.color import ColorSegmenter
            seg = ColorSegmenter(_hex(color), tolerance)
            return np.asarray(seg.segment(image), dtype=np.float32)

        # AI path — choose adaptive segmenter by category + backdrop.
        seg = self._select_segmenter(image, category, strategy)
        long_side = max(w, h)
        if long_side > TILE_SIZE * 2:
            # Very large images: tile, then refine with a 2048px pass to
            # recover the fine-detail structures tiling can soften.
            return self._segment_tiled(image, seg, progress_cb)
        if long_side > SINGLE_CAP:
            # Large image: two-scale segmentation blends global context with
            # fine detail, far better than tiling for hair / fur.
            return self._segment_multiscale(image, seg, progress_cb)

        work = image if max(w, h) <= cap else _resize_long(image, cap)
        if work.size != image.size:
            from .utils import to_pil
            mask = np.asarray(seg.segment(work), dtype=np.float32)
            mask_img = to_pil(mask * 255.0).resize(image.size, Image.LANCZOS)
            return np.asarray(mask_img, dtype=np.float32) / 255.0
        return np.asarray(seg.segment(image), dtype=np.float32)

    def _select_segmenter(self, image, category, strategy):
        if strategy.get('seg') == 'color':
            from .segmentation.color import border_color_stats, auto_tolerance
            try:
                bg, frac = border_color_stats(image, band_ratio=0.04)
                if frac >= 0.5:
                    from .segmentation.color import ColorSegmenter
                    tol = auto_tolerance(image, bg)
                    return ColorSegmenter(bg, tol)
            except Exception:
                pass
        # Human / animal subjects get the portrait-optimised model — the
        # single biggest quality win for hair, fur and skin edges. On GPU the
        # portrait BiRefNet weights are used; on CPU-only hardware the fast
        # ISNet model is selected automatically (both models share the same
        # 1024x1024 + normalisation interface).
        if category in ('human', 'animal'):
            return self._portrait_segmenter()
        return self._birefnet_segmenter()

    def _birefnet_segmenter(self):
        from .segmentation.birefnet import make_segmenter
        if self._birefnet is None:
            self._birefnet = make_segmenter('general')
        return self._birefnet

    def _portrait_segmenter(self):
        from .segmentation.birefnet import make_segmenter
        if self._birefnet_portrait is None:
            self._birefnet_portrait = make_segmenter('portrait')
        return self._birefnet_portrait

    def _segment_tiled(self, image, seg, progress_cb):
        """Segmentation on overlapping tiles with feathered seams."""
        w, h = image.size
        alpha = np.zeros((h, w), dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)
        step = TILE_SIZE - TILE_OVERLAP

        xs = list(range(0, max(1, w - step), step))
        ys = list(range(0, max(1, h - step), step))
        if not xs or xs[-1] + TILE_SIZE < w:
            xs.append(max(0, w - TILE_SIZE))
        if not ys or ys[-1] + TILE_SIZE < h:
            ys.append(max(0, h - TILE_SIZE))

        total = len(xs) * len(ys)
        idx = 0
        for y0 in ys:
            for x0 in xs:
                idx += 1
                box = (x0, y0, min(w, x0 + TILE_SIZE), min(h, y0 + TILE_SIZE))
                tile = image.crop(box)
                mask = np.asarray(seg.segment(tile), dtype=np.float32)
                wgt = _feather_weights(mask.shape[0], mask.shape[1],
                                       TILE_OVERLAP)
                alpha[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]] += mask * wgt
                weight[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]] += wgt
                self._call(progress_cb, 35 + int(idx / total * 25),
                           f'Segmenting tile {idx}/{total}...')

        div = np.maximum(weight, 1e-6)
        return alpha / div

    def _segment_multiscale(self, image, seg, progress_cb, scales=(1024, 2048)):
        """Segment at two resolutions and blend by edge strength.

        The low-res pass captures global context; the high-res pass sees the
        fine strands. In the transition band (where hair / fur / fine detail
        live) the high-res result dominates; the flat interior is filled by
        the low-res pass so distant regions stay consistent.
        """
        from .matting.edge_detection import edge_zone
        from .utils import gaussian, to_pil

        w, h = image.size
        alphas = []
        n_scales = len(scales)
        for i, scale in enumerate(scales):
            if max(w, h) <= scale:
                mask = np.asarray(seg.segment(image), dtype=np.float32)
            else:
                work = _resize_long(image, scale)
                mask = np.asarray(seg.segment(work), dtype=np.float32)
                mask_img = to_pil(mask * 255.0).resize((w, h), Image.LANCZOS)
                mask = np.asarray(mask_img, dtype=np.float32) / 255.0
            alphas.append(mask)
            self._call(progress_cb,
                       35 + int((i + 1) / n_scales * 12),
                       f'Segmenting scale {scale // 1024}K ({i + 1}/{n_scales})...')

        alpha_lo, alpha_hi = alphas[0], alphas[-1]
        # Edge zone of the low-res pass → where we prefer fine detail.
        edge_mask = edge_zone(alpha_lo, thresh=0.02, dilation=10)
        blend = np.clip(gaussian(edge_mask.astype(np.float32), 5.0), 0.0, 1.0)

        # Also keep high-res where the two disagree strongly about
        # transparency (a thin strand the low-res pass erased).
        diff = np.abs(alpha_lo - alpha_hi)
        diff_n = np.clip((diff - 0.15) / 0.4, 0.0, 1.0)
        blend = np.maximum(blend, diff_n)

        out = blend * alpha_hi + (1.0 - blend) * alpha_lo
        return np.clip(out, 0.0, 1.0)

    def _refine_alpha(self, image, alpha, category, strategy, progress_cb,
                      pass_number=1):
        """Optimized refinement chain for BiRefNet matte.

        Key improvements:
        - Avoids triple decontamination (was: decontaminate -> hair_refine -> hair_matting)
        - Passes original_rgb to hair_matting for accurate Lab chroma correction
        - Uses hair_refine for alpha+color refinement, hair_matting for PyMatting+defringing
        - Order: cleanup -> hair_refine (alpha refinement + decontam + halo) 
                 -> hair_matting (PyMatting refinement + final defringing) -> feather
        """
        cfg = dict(strategy or {})
        a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
        img_rgb = np.asarray(image, dtype=np.float32)

        # By-colour mode is a plain distance-based cut — no AI-edge tooling.
        if cfg.get('color_mode'):
            from .postprocess import cleanup
            a255 = cleanup(a * 255.0, category, cfg)
            a = np.clip(np.asarray(a255, dtype=np.float32) / 255.0, 0.0, 1.0)
            self._call(progress_cb, 76, 'Cleaning up edges...')
            return img_rgb, a

        # 1 — geometric mask cleanup (speckles + holes), category-aware.
        from .postprocess import cleanup
        a255 = cleanup(a * 255.0, category, cfg)
        a = np.clip(np.asarray(a255, dtype=np.float32) / 255.0, 0.0, 1.0)
        self._call(progress_cb, 66, 'Cleaning mask...')

        # 2 — Hair-aware edge refinement (alpha + color + halo).
        # This replaces the old standalone decontaminate() call which caused
        # triple-decontamination when combined with hair_refine + hair_matting.
        # hair_refine does: alpha re-projection, spill reduction, halo removal, strand decontamination
        rgb_out = img_rgb
        if cfg.get('hair_refine', True) and (a > 0.02).any() and (a < 0.97).any():
            from .postprocess.hair_refine import refine_hair
            rgb_out, a = refine_hair(rgb_out, a, cfg)
            self._call(progress_cb, 72, 'Refining hair edges...')
        elif (cfg.get('decontaminate', True) and (a > 0.05).any() and (a < 0.98).any()):
            # Fallback: lightweight RGB-only decontamination for non-hair categories
            from .postprocess.decontamination import decontaminate
            dct_radius = 8 if cfg.get('hair_refine', True) else 6
            rgb_out = decontaminate(img_rgb, a, strength=0.7, radius=dct_radius,  # Reduced from 0.9
                                    iterations=1, min_a=0.08)
            self._call(progress_cb, 72, 'Decontaminating edges...')

        # 3 — Optional high-quality hair matting + color defringing (PyMatting/OpenCV).
        # Enabled via strategy['hair_matting'] = True. Uses PyMatting KNN matting
        # with fine-hair threshold and Lab-space color defringing for white halo removal.
        # Runs AFTER hair_refine so PyMatting gets a cleaner alpha, then final defringing.
        if cfg.get('hair_matting', False) and (a > 0.01).any() and (a < 0.99).any():
            from .postprocess.hair_matting import refine_hair_matte
            # Merge strategy with hair_matting specific config
            hm_cfg = dict(cfg)
            hm_cfg.update(cfg.get('hair_matting_cfg', {}))
            # Pass original image for Lab chroma correction reference (critical for color accuracy)
            hm_cfg['original_rgb'] = img_rgb
            rgb_out, a = refine_hair_matte(rgb_out, a, hm_cfg)
            self._call(progress_cb, 75, 'Hair matting + defringing...')

        # 4 — Light category feather on the alpha transition band only.
        f = float(cfg.get('feather', 0.0) or 0.0)
        if 0.0 < f <= 2.0 and (a > 0.0).any() and (a < 1.0).any():
            import PIL.ImageFilter as _IF
            from .utils import to_pil
            blurred = np.asarray(
                to_pil(a * 255.0).filter(_IF.GaussianBlur(f)),
                dtype=np.float32) / 255.0
            band = (a > 0.02) & (a < 0.98)
            a = np.where(band, blurred, a)

        return np.clip(rgb_out, 0.0, 255.0), np.clip(a, 0.0, 1.0)

    def _compose(self, image, alpha, background, add_shadow, custom_image,
                 progress_cb, rgb=None):
        self._call(progress_cb, 88, 'Compositing...')
        from .compositing import parse_background, render_background, soft_shadow
        mode, param = parse_background(background)
        custom = None
        if mode == 'custom':
            if custom_image:
                custom = Image.open(custom_image).convert('RGB')
            else:
                raise ValueError(
                    'A custom background image is required for this mode')

        from .utils import to_pil
        if rgb is not None:
            cut_src = Image.fromarray(
                np.clip(rgb, 0, 255).astype(np.uint8), mode='RGB')
        else:
            cut_src = image.convert('RGB')
        cutout = cut_src.convert('RGBA')
        cutout.putalpha(to_pil(alpha * 255.0))
        if add_shadow and mode == 'transparent':
            cutout = soft_shadow(cutout)
        return render_background(cutout, mode, param, original=image,
                                 custom_image=custom)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _call(progress_cb, pct, msg):
        if progress_cb:
            progress_cb(pct, msg)


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def _score(alpha, rgb):
    from .quality import score
    return score(alpha, rgb, category=None)


def _hex(value):
    value = value.lstrip('#')
    if len(value) != 6:
        raise ValueError(f'Invalid hex colour: {value!r}')
    return [int(value[i:i + 2], 16) for i in (0, 2, 4)]


def _resize_long(image, long_side):
    w, h = image.size
    r = long_side / max(w, h)
    return image.resize((max(1, int(w * r)), max(1, int(h * r))), Image.LANCZOS)


def _decimate_rgb(rgb, long_side=320):
    h, w, _ = rgb.shape
    if max(h, w) <= long_side:
        return rgb
    r = long_side / max(h, w)
    nw, nh = max(1, int(w * r)), max(1, int(h * r))
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), 'RGB')
    return np.asarray(img.resize((nw, nh), Image.LANCZOS), dtype=np.float32)


def _decimate_alpha(alpha, long_side=320):
    from .utils import to_pil
    h, w = alpha.shape
    if max(h, w) <= long_side:
        return alpha.astype(np.float32)
    r = long_side / max(h, w)
    nw, nh = max(1, int(w * r)), max(1, int(h * r))
    return np.asarray(to_pil(alpha * 255.0).resize((nw, nh), Image.BILINEAR),
                      dtype=np.float32) / 255.0


def _feather_weights(h, w, overlap):
    """Raised-cosine weight ramp: 0 at tile edges → 1 inside, so adjacent
    tiles blend smoothly across their overlap."""
    def ramp(n, over):
        if over <= 1 or n <= 2 * over:
            return np.ones(n, dtype=np.float32)
        x = np.ones(n, dtype=np.float32)
        x[:over] = 0.5 - 0.5 * np.cos(np.pi * np.arange(over) / over)
        x[n - over:] = 0.5 + 0.5 * np.cos(np.pi * np.arange(over) / over)
        return x

    return ramp(h, overlap)[:, None] * ramp(w, overlap)[None, :]


def _fmt_time(sec):
    if sec >= 60:
        return f'{int(sec // 60)}m{int(sec % 60)}s'
    if sec >= 1:
        return f'{sec:.1f}s'
    return f'{int(sec * 1000)}ms'