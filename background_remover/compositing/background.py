"""Background rendering.

Supported backgrounds:

- ``transparent``   → keep RGBA, save as PNG
- ``'#rrggbb'``     → solid colour fill
- ``'blur'``        → the original photo blurred behind the cutout
- ``CUSTOM_PATH``   → any image file pasted behind the cutout (scaled to fit)

We never invent model files here: a custom background is always a real
file supplied by the client (uploaded via the existing upload flow).
"""

import os
from PIL import Image, ImageFilter

# Sentinel used by the API to say 'use the uploaded custom image'
CUSTOM_KEY = 'custom'


def parse_background(value):
    """Normalise a client-provided background token.

    Returns one of:
        ('transparent', None)
        ('color', '#rrggbb')
        ('blur', None)
        ('custom', <file path>)
    """
    if not value or value == 'transparent':
        return 'transparent', None
    if value == 'blur':
        return 'blur', None
    if value == CUSTOM_KEY:
        return 'custom', None
    return 'color', value


def render_background(cutout, mode, param=None, original=None,
                      custom_image=None):
    """Render ``cutout`` (RGBA PIL) onto the requested background.

    ``original`` is the source photo (for 'blur'). ``custom_image`` is a
    PIL image (for 'custom'). Returns a PIL image (RGBA for transparent,
    RGB otherwise).
    """
    if mode == 'transparent':
        return cutout
    if mode == 'color':
        rgb = _hex_rgb(param or '#ffffff')
        bg = Image.new('RGBA', cutout.size, rgb + (255,))
        bg.alpha_composite(cutout)
        return bg.convert('RGB')
    if mode == 'blur':
        img = (original or cutout).convert('RGB')
        long_side = max(img.size)
        radius = max(2.0, long_side / 300.0)
        blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
        bg = blurred.convert('RGBA')
        bg.alpha_composite(cutout)
        return bg.convert('RGB')
    if mode == 'custom':
        src = custom_image or original
        if src is None:
            raise ValueError('Custom background requires an image')
        bg = src.convert('RGB').resize(cutout.size, Image.LANCZOS).convert('RGBA')
        bg.alpha_composite(cutout)
        return bg.convert('RGB')
    raise ValueError(f'Unsupported background mode: {mode!r}')


def _hex_rgb(value):
    value = value.lstrip('#')
    if len(value) != 6:
        raise ValueError(f'Invalid hex colour: {value!r}')
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))