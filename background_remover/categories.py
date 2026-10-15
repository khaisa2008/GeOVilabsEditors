"""Image-content categories and their adaptive processing strategies.

Each strategy tunes the segmentation, calibration, matting, refinement,
decontamination and cleanup so different kinds of subjects get the correct
treatment — a logo needs sharp binary edges (no halo feathering), a portrait
needs hair-friendly soft edges with colour-gated dilation and heavier
decontamination so white fringes disappear.
"""

HUMAN = 'human'
ANIMAL = 'animal'
PLANT = 'plant'
PRODUCT = 'product'
LOGO = 'logo'
CARTOON = 'cartoon'
DOCUMENT = 'document'
FOOD = 'food'
VEHICLE = 'vehicle'
MIXED = 'mixed'

CATEGORY_LABELS = {
    HUMAN: 'Human / Portrait',
    ANIMAL: 'Animal',
    PLANT: 'Plant / Nature',
    PRODUCT: 'Product / Object',
    LOGO: 'Logo / Graphic',
    CARTOON: 'Cartoon / Illustration',
    DOCUMENT: 'Document / Text',
    FOOD: 'Food',
    VEHICLE: 'Vehicle',
    MIXED: 'Mixed / Unknown',
}

# Default strategy (safe, balanced) — used when classification confidence
# is low (see requirement: don't force a category when unsure).
# The post-model chain is deliberately minimal and model-appropriate: BiRefNet
# already outputs a soft, well-formed alpha, so we only clean the mask
# geometrically and de-contaminate edge colours — never pull alpha heuristically.
DEFAULTS = dict(
    seg='birefnet',
    cleanup='normal',
    decontaminate=True,     # RGB-only fringe un-mix (halo removal), alpha untouched
    hair_refine=True,       # hair-aware edge refinement (spill / gaps / halo / decon)
    feather=0.8,            # light transition feather (px) on the 1-2px band

    # --- hair-aware refinement tuning (see postprocess.hair_refine) ----------
    edge_band_width=6.0,            # narrow transition band half-width (px)
    spill_strength=0.85,            # bg-dominated rim alpha pull (0..1)
    decontamination_strength=0.80,  # strand colour un-mix strength (0..1)
    strand_preserve_threshold=0.50, # min Lab fg-match protecting thin strands
    alpha_refine_strength=0.85,     # colour-implied alpha blend weight (0..1)
)

STRATEGIES = {
    HUMAN: dict(
        seg='birefnet_portrait',   # portrait-optimised BiRefNet (hair/skin)
        cleanup='light',           # never erode thin strands
        feather=1.0,
        # Improved hair refinement parameters
        edge_band_width=12.0,              # Increased from 6.0 for curly hair
        spill_strength=0.90,               # Increased from 0.85
        decontamination_strength=0.90,     # Increased from 0.80
        strand_preserve_threshold=0.45,    # Lowered from 0.50 to protect more strands
        alpha_refine_strength=0.90,        # Increased from 0.85
        hair_matting=True,                 # enable PyMatting/OpenCV hair matting + defringing
        hair_matting_cfg=dict(
            use_pymatting=True,
            matting_method='kn',           # KNN faster and often better for hair texture
            hair_threshold=0.15,           # Increased from 0.10 to capture more semi-transparent hair
            trimap_dilate=12,              # Increased from 8 for wider unknown band
            trimap_erode=8,                # Reduced from 10 to preserve more fg detail
            defringe_strength=0.95,        # Slightly reduced from 1.0 (more natural)
            defringe_radius=12,            # Reduced from 24 (was too large)
            defringe_iterations=4,         # Reduced from 8 (faster, still effective)
            halo_removal=True,
            halo_strength=1.3,             # Increased from 1.2
            feather_px=0.3,
            max_work_px=2048,
            # New params for improved decontamination
            decontam_zone=12,
            defringe_min_a=0.03,
        ),
    ),
    ANIMAL: dict(
        seg='birefnet_portrait',
        cleanup='light',
        feather=0.9,
        edge_band_width=10.0,
        spill_strength=0.88,
        decontamination_strength=0.88,
        strand_preserve_threshold=0.45,
        alpha_refine_strength=0.88,
        hair_matting=True,
        hair_matting_cfg=dict(
            use_pymatting=True,
            matting_method='kn',
            hair_threshold=0.12,
            trimap_dilate=12,
            trimap_erode=8,
            defringe_strength=0.95,
            defringe_radius=12,
            defringe_iterations=4,
            halo_removal=True,
            halo_strength=1.2,
            feather_px=0.3,
            max_work_px=2048,
            decontam_zone=12,
            defringe_min_a=0.03,
        ),
    ),
    PLANT: dict(
        seg='birefnet',
        cleanup='light',
        feather=0.7,
    ),
    PRODUCT: dict(
        seg='birefnet',
        cleanup='normal',
        feather=0.4,
    ),
    LOGO: dict(
        seg='color',               # fall back to BiRefNet if not flat enough
        cleanup='sharp',
        decontaminate=False,       # hard edges must stay razor sharp
        hair_refine=False,         # no halo stage on razor-sharp silhouettes
        feather=0.0,
    ),
    CARTOON: dict(
        seg='birefnet',
        cleanup='normal',
        decontaminate=False,       # keep outline colours crisp
        hair_refine=False,
        feather=0.0,
    ),
    DOCUMENT: dict(
        seg='color',
        cleanup='strong',          # speckle removal on flat scans
        decontaminate=False,       # text edges must stay razor sharp
        hair_refine=False,
        feather=0.0,
    ),
    FOOD: dict(
        seg='birefnet',
        cleanup='normal',
        feather=0.6,
    ),
    VEHICLE: dict(
        seg='birefnet',
        cleanup='normal',
        feather=0.4,
    ),
    MIXED: dict(**DEFAULTS),
}


def strategy_for(category):
    """Strategy dict for a category, merged over defaults."""
    cfg = dict(DEFAULTS)
    cfg.update(STRATEGIES.get(category, STRATEGIES[MIXED]))
    return cfg