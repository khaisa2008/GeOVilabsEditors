"""Matting subpackage — edge detection, calibration, trimap + alpha matting."""

from .trimap import generate_trimap              # noqa: F401
from .alpha_matting import refine_alpha          # noqa: F401
from .edge_detection import (                    # noqa: F401
    alpha_gradient, color_edge_map, boundary_map,
    fg_bg_contrast, fg_similarity,
)
from .calibration import estimate_modes, calibrate_alpha  # noqa: F401

__all__ = [
    'generate_trimap', 'refine_alpha',
    'alpha_gradient', 'color_edge_map', 'boundary_map',
    'fg_bg_contrast', 'fg_similarity',
    'estimate_modes', 'calibrate_alpha',
]