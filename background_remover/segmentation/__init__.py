"""Adaptive segmentation subpackage.

Provides a common ``MaskSegmenter`` interface so the pipeline can swap
strategies without touching orchestration logic:

    segmenter.segment(image) -> float32 alpha mask (0..1) at image size

BiRefNet wraps the downloaded model via model_store; ``rmbg`` and ``u2net``
are drop-in modular implementations (point them at a user-provided onnx
file — the pipeline falls back automatically if they are missing).
"""

from .base import MaskSegmenter, ONNXSegmenter  # noqa: F401
from .birefnet import BiRefNetSegmenter          # noqa: F401
from .color import ColorSegmenter               # noqa: F401

__all__ = ['MaskSegmenter', 'ONNXSegmenter', 'BiRefNetSegmenter',
           'ColorSegmenter']