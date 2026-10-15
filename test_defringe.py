#!/usr/bin/env python
"""More focused test for color defringing (white halo removal)."""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from background_remover.postprocess.hair_matting import _defringe_color, _estimate_bg_field_guided


def create_white_fringe_test():
    """Create a test case with obvious white fringe on dark hair."""
    h, w = 256, 256

    # Dark hair color
    hair_color = np.array([30, 20, 15], dtype=np.float32)
    # Bright background (white wall)
    bg_color = np.array([240, 235, 230], dtype=np.float32)

    # Simple alpha: solid circle with soft edge
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    radius = 80

    # Smooth alpha transition
    alpha = np.clip(1.0 - (dist - radius + 10) / 20.0, 0.0, 1.0)

    # Create clean composite (no fringe)
    rgb_clean = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        rgb_clean[..., c] = alpha * hair_color[c] + (1 - alpha) * bg_color[c]

    # Create version WITH white fringe (the problem case)
    # Add white fringe in transition zone (alpha 0.2-0.6)
    fringe_zone = (alpha > 0.2) & (alpha < 0.6)
    fringe_strength = np.clip((0.6 - alpha) / 0.4, 0, 1) * fringe_zone
    rgb_fringed = rgb_clean.copy()
    for c in range(3):
        # Mix in white (255) in the fringe zone
        rgb_fringed[..., c] = rgb_fringed[..., c] * (1 - fringe_strength * 0.6) + 255 * fringe_strength * 0.6

    return rgb_clean, rgb_fringed, alpha, hair_color, bg_color


def test_defringe():
    """Test the _defringe_color function directly."""
    print("Creating white fringe test case...")
    rgb_clean, rgb_fringed, alpha, hair_color, bg_color = create_white_fringe_test()

    # Verify input has white fringe
    band = (alpha > 0.2) & (alpha < 0.6)
    fringe_lum = 0.299 * rgb_fringed[band, 0] + 0.587 * rgb_fringed[band, 1] + 0.114 * rgb_fringed[band, 2]
    print(f"Input fringe zone luminance: mean={fringe_lum.mean():.1f}, max={fringe_lum.max():.1f}")

    # Test defringing
    cfg = {
        'defringe_strength': 0.9,
        'defringe_radius': 14,
        'defringe_iterations': 4,
        'defringe_min_a': 0.05,
        'halo_removal': True,
        'hair_refine': True,
    }

    print("\nRunning defringing...")
    rgb_defringed, alpha_defringed = _defringe_color(rgb_fringed, alpha, cfg)

    # Check output
    defringed_lum = 0.299 * rgb_defringed[band, 0] + 0.587 * rgb_defringed[band, 1] + 0.114 * rgb_defringed[band, 2]
    defringed_chroma = rgb_defringed[band].max(axis=1) - rgb_defringed[band].min(axis=1)
    print(f"Output fringe zone luminance: mean={defringed_lum.mean():.1f}, max={defringed_lum.max():.1f}")
    print(f"Output fringe zone chroma: mean={defringed_chroma.mean():.1f}")

    # Check if hair color is recovered (should be closer to hair_color)
    hair_zone = alpha > 0.7
    hair_rgb = rgb_defringed[hair_zone].mean(axis=0)
    print(f"Recovered hair color (solid): {hair_rgb}")
    print(f"Expected hair color: {hair_color}")

    # Better metric: measure "whiteness" in fringe zone
    # White fringe = high luminance + low chroma
    clean_lum = 0.299 * rgb_clean[band, 0] + 0.587 * rgb_clean[band, 1] + 0.114 * rgb_clean[band, 2]
    clean_chroma = rgb_clean[band].max(axis=1) - rgb_clean[band].min(axis=1)

    def whiteness(rgb_arr, alpha_arr):
        b = (alpha_arr > 0.2) & (alpha_arr < 0.6)
        if not b.any():
            return 0.0
        lum = 0.299 * rgb_arr[b, 0] + 0.587 * rgb_arr[b, 1] + 0.114 * rgb_arr[b, 2]
        chroma = rgb_arr[b].max(axis=1) - rgb_arr[b].min(axis=1)
        white = (lum > 180) & (chroma < 30)
        return white.mean() if white.any() else 0.0

    white_in = whiteness(rgb_fringed, alpha)
    white_out = whiteness(rgb_defringed, alpha)
    white_clean = whiteness(rgb_clean, alpha)

    print(f"\nWhiteness metric (lower=less white fringe):")
    print(f"  Clean reference:  {white_clean:.4f}")
    print(f"  Fringed input:    {white_in:.4f}")
    print(f"  Defringed output: {white_out:.4f}")
    print(f"  Reduction:        {white_in - white_out:.4f} ({(white_in - white_out)/white_in*100:.1f}%)")

    # Also check color difference in solid hair region
    hair_diff_in = np.abs(rgb_fringed[hair_zone] - rgb_clean[hair_zone]).mean()
    hair_diff_out = np.abs(rgb_defringed[hair_zone] - rgb_clean[hair_zone]).mean()
    print(f"\nSolid hair color diff from clean:")
    print(f"  Fringed input:  {hair_diff_in:.2f}")
    print(f"  Defringed out:  {hair_diff_out:.2f}")

    # Save for visual inspection
    def save_rgba(rgb_arr, alpha_arr, name):
        rgb_u8 = np.clip(rgb_arr, 0, 255).astype(np.uint8)
        a_u8 = (np.clip(alpha_arr, 0, 1) * 255).astype(np.uint8)
        rgba = np.dstack([rgb_u8, a_u8])
        Image.fromarray(rgba, 'RGBA').save(f'test_fringe_{name}.png')

    save_rgba(rgb_clean, alpha, 'clean')
    save_rgba(rgb_fringed, alpha, 'fringed_input')
    save_rgba(rgb_defringed, alpha_defringed, 'defringed_output')

    print("\nSaved test images.")
    return white_out < white_in  # Should be True if defringing works


def test_bg_field():
    """Test background field estimation."""
    print("\n--- Testing background field estimation ---")
    rgb_clean, rgb_fringed, alpha, hair_color, bg_color = create_white_fringe_test()

    B = _estimate_bg_field_guided(rgb_fringed, alpha, radius=8)
    print(f"True background color: {bg_color}")
    print(f"Estimated background (center): {B[128, 128]}")
    print(f"Estimated background (edge): {B[80, 80]}")

    # The background field should be close to true bg in background regions
    bg_region = alpha < 0.1
    if bg_region.any():
        B_bg = B[bg_region].mean(axis=0)
        print(f"Background field in bg region: {B_bg}")
        print(f"Error: {np.abs(B_bg - bg_color).mean():.2f}")


if __name__ == '__main__':
    success = test_defringe()
    test_bg_field()
    if success:
        print("\n[PASS] Defringing reduced color difference from clean reference!")
    else:
        print("\n[WARN] Defringing did not improve - check implementation")