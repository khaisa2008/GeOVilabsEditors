"""Compositing — put the cutout onto any chosen background."""

from .background import render_background, CUSTOM_KEY, parse_background   # noqa: F401
from .shadow import soft_shadow                                           # noqa: F401
from .lighting import harmonize                                           # noqa: F401

__all__ = ['render_background', 'soft_shadow', 'harmonize',
           'CUSTOM_KEY', 'parse_background']