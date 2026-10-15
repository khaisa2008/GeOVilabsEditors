"""Adaptive Background Removal & Image Segmentation Engine for GeOViLabs.

A modular, category-aware background remover that:

1. Classifies image content (human / animal / plant / product / logo /
   cartoon / document / food / vehicle / mixed).
2. Picks an adaptive segmentation + refinement strategy per category.
3. Refines edges with trimap + alpha matting only where needed.
4. Removes halos / background color contamination.
5. Cleans the mask without destroying hair, fur or thin structures.
6. Scores output quality and re-runs with a stronger strategy if low.
7. Composites onto transparent, color, blurred or custom backgrounds.

Only numpy + Pillow + onnxruntime are required at runtime. Modules are
imported lazily so a missing heavy model never crashes the app.
"""

from .categories import (
    HUMAN, ANIMAL, PLANT, PRODUCT, LOGO, CARTOON,
    DOCUMENT, FOOD, VEHICLE, MIXED,
    CATEGORY_LABELS, STRATEGIES,
)

__all__ = [
    'HUMAN', 'ANIMAL', 'PLANT', 'PRODUCT', 'LOGO', 'CARTOON',
    'DOCUMENT', 'FOOD', 'VEHICLE', 'MIXED',
    'CATEGORY_LABELS', 'STRATEGIES',
    'PipelineResult', 'BackgroundPipeline',
]


class PipelineResult:
    """Metadata returned by :meth:`BackgroundPipeline.remove`."""

    def __init__(self, category='mixed', confidence=0.0, quality_score=0.0,
                 passes=1, method='ai', width=0, height=0,
                 processing_time=0.0, message=''):
        self.category = category
        self.confidence = confidence
        self.quality_score = quality_score
        self.passes = passes
        self.method = method
        self.width = width
        self.height = height
        self.processing_time = processing_time
        self.message = message

    def as_dict(self):
        return {
            'success': True,
            'image_type': self.category,
            'confidence': round(self.confidence, 3),
            'quality_score': round(self.quality_score, 3),
            'passes': self.passes,
            'method': self.method,
            'processing_time': round(self.processing_time, 2),
            'width': self.width,
            'height': self.height,
            'message': self.message,
        }


def get_pipeline():
    """Return the process-wide :class:`BackgroundPipeline` singleton."""
    from .pipeline import BackgroundPipeline
    instance = globals().get('_pipeline')
    if instance is None:
        instance = BackgroundPipeline()
        globals()['_pipeline'] = instance
    return instance


def remove_background(*args, **kwargs):
    """Convenience entry point used by ``bg_remove.py``."""
    return get_pipeline().remove(*args, **kwargs)