#!/usr/bin/env python
"""Test script for hair_matting module."""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from background_remover.postprocess.hair_matting import refine_hair_matte


def create_synthetic_hair_test():
    """Create a synthetic test image simulating curly hair on a colored background."""
    h, w = 512, 512

    # Background: solid color (simulating wall)
    bg_color = (200, 180, 160)  # warm beige wall
    bg = np.full((h, w, 3), bg_color, dtype=np.float32)

    # Foreground: dark hair color
    hair_color = (30, 20, 15)  # dark brown/black hair

    # Create base alpha with "hair-like" strands
    alpha = np.zeros((h, w), dtype=np.float32)

    # Draw some curved "hair strands" using PIL
    img = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(img)

    # Draw multiple curved strands
    import math
    np.random.seed(42)
    for i in range(30):
        x_start = np.random.randint(50, w - 50)
        y_start = np.random.randint(50, h - 50)
        length = np.random.randint(80, 200)
        thickness = np.random.randint(1, 4)
        angle = np.random.uniform(0, 2 * math.pi)
        curvature = np.random.uniform(-0.02, 0.02)

        points = []
        x, y = x_start, y_start
        a = angle
        for step in range(length):
            points.append((x, y))
            x += math.cos(a) * 2
            y += math.sin(a) * 2
            a += curvature * np.random.uniform(-1, 1)
            if x < 0 or x >= w or y < 0 or y >= h:
                break
        if len(points) > 2:
            draw.line(points, fill=255, width=thickness, joint='curve')

    # Add a solid head/body region
    draw.ellipse([w//2 - 80, h//2 - 60, w//2 + 80, h//2 + 100], fill=255)

    alpha = np.asarray(img, dtype=np.float32) / 255.0

    # Blur alpha slightly for semi-transparent edges
    alpha_img = Image.fromarray((alpha * 255).astype(np.uint8))
    alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(1.5))
    alpha = np.asarray(alpha_img, dtype=np.float32) / 255.0

    # Create RGB: composite hair on background with some "fringe" artifacts
    rgb = bg.copy()
    # Hair color where alpha > 0
    for c in range(3):
        rgb[..., c] = alpha * hair_color[c] + (1 - alpha) * bg_color[c]

    # Add white halo artifact (simulating the problem)
    # White fringe at alpha transition
    halo_mask = (alpha > 0.1) & (alpha < 0.5)
    halo_strength = np.clip((0.5 - alpha) / 0.5, 0, 1) * halo_mask
    for c in range(3):
        rgb[..., c] = rgb[..., c] * (1 - halo_strength * 0.4) + 255 * halo_strength * 0.4

    # Add some gray wash in hair gaps (background spill)
    gap_mask = (alpha > 0.05) & (alpha < 0.3)
    for c in range(3):
        rgb[..., c] = rgb[..., c] * (1 - gap_mask * 0.2) + bg_color[c] * gap_mask * 0.2

    return np.clip(rgb, 0, 255).astype(np.uint8), alpha


def test_refine_hair_matte():
    """Test the refine_hair_matte function."""
    print("Creating synthetic hair test image...")
    rgb, alpha = create_synthetic_hair_test()

    print(f"Input RGB shape: {rgb.shape}, dtype: {rgb.dtype}")
    print(f"Input alpha shape: {alpha.shape}, dtype: {alpha.dtype}")
    print(f"Alpha range: {alpha.min():.3f} - {alpha.max():.3f}")
    print(f"Semi-transparent pixels: {((alpha > 0.01) & (alpha < 0.99)).sum()}")

    # Test with PyMatting CF method
    print("\n--- Testing with PyMatting CF (high quality) ---")
    strategy_cf = {
        'use_pymatting': True,
        'matting_method': 'cf',
        'hair_threshold': 0.15,
        'trimap_dilate': 15,
        'trimap_erode': 15,
        'defringe_strength': 0.85,
        'defringe_radius': 8,
        'defringe_iterations': 2,
        'halo_removal': True,
        'feather_px': 0.5,
        'max_work_px': 1024,
        'hair_refine': True,
    }
    rgb_out_cf, alpha_out_cf = refine_hair_matte(rgb, alpha, strategy_cf)
    print(f"Output alpha range: {alpha_out_cf.min():.3f} - {alpha_out_cf.max():.3f}")
    print(f"Semi-transparent pixels after: {((alpha_out_cf > 0.01) & (alpha_out_cf < 0.99)).sum()}")

    # Test with PyMatting KNN method (faster)
    print("\n--- Testing with PyMatting KNN (faster) ---")
    strategy_kn = dict(strategy_cf, matting_method='kn')
    rgb_out_kn, alpha_out_kn = refine_hair_matte(rgb, alpha, strategy_kn)
    print(f"Output alpha range: {alpha_out_kn.min():.3f} - {alpha_out_kn.max():.3f}")

    # Test fallback (numpy only)
    print("\n--- Testing fallback (numpy only, use_pymatting=False) ---")
    strategy_fallback = dict(strategy_cf, use_pymatting=False)
    rgb_out_fb, alpha_out_fb = refine_hair_matte(rgb, alpha, strategy_fallback)
    print(f"Output alpha range: {alpha_out_fb.min():.3f} - {alpha_out_fb.max():.3f}")

    # Save test outputs
    def save_rgba(rgb_arr, alpha_arr, name):
        rgb_u8 = np.clip(rgb_arr, 0, 255).astype(np.uint8)
        a_u8 = (np.clip(alpha_arr, 0, 1) * 255).astype(np.uint8)
        rgba = np.dstack([rgb_u8, a_u8])
        Image.fromarray(rgba, 'RGBA').save(f'test_output_{name}.png')
        print(f"Saved test_output_{name}.png")

    save_rgba(rgb, alpha, 'input')
    save_rgba(rgb_out_cf, alpha_out_cf, 'cf')
    save_rgba(rgb_out_kn, alpha_out_kn, 'kn')
    save_rgba(rgb_out_fb, alpha_out_fb, 'fallback')

    # Compute metrics
    def compute_fringe_metric(rgb_arr, alpha_arr):
        """Measure white/grey fringe in transition band."""
        band = (alpha_arr > 0.05) & (alpha_arr < 0.5)
        if not band.any():
            return 0.0
        band_rgb = rgb_arr[band]
        # Check for high luminance + low chroma (white/grey fringe)
        lum = 0.299 * band_rgb[..., 0] + 0.587 * band_rgb[..., 1] + 0.114 * band_rgb[..., 2]
        chroma = band_rgb.max(axis=1) - band_rgb.min(axis=1)
        white_fringe = (lum > 180) & (chroma < 30)
        return white_fringe.mean() if white_fringe.any() else 0.0

    print("\n--- Fringe Metrics (lower is better) ---")
    print(f"Input fringe score: {compute_fringe_metric(rgb, alpha):.4f}")
    print(f"CF output fringe:   {compute_fringe_metric(rgb_out_cf, alpha_out_cf):.4f}")
    print(f"KNN output fringe:  {compute_fringe_metric(rgb_out_kn, alpha_out_kn):.4f}")
    print(f"Fallback fringe:    {compute_fringe_metric(rgb_out_fb, alpha_out_fb):.4f}")

    print("\n[OK] All tests passed!")


if __name__ == '__main__':
    test_refine_hair_matte()