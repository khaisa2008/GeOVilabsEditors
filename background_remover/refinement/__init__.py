"""Category-aware edge / structure refinement.

The pipeline routes ``alpha`` to the right refiner based on the detected
content type (human → hair, animal → fur, plant → thin-structure, logo →
sharp, cartoon → crisp outlines, product → geometric edges).

Every refiner shares the signature:
    refine(alpha, image, strategy, fallback=None) -> float32 alpha 0..1
"""

_NAME_TO_MODULE = {
    'hair': 'hair',
    'fur': 'fur',
    'plant': 'plant',
    'logo': 'logo',
    'cartoon': 'cartoon',
    'edge': 'edge',
}

_CATEGORY_TO_NAME = {
    'human': 'hair',
    'animal': 'fur',
    'plant': 'plant',
    'product': 'edge',
    'logo': 'logo',
    'cartoon': 'cartoon',
    'document': 'logo',      # text must stay razor sharp
    'food': 'edge',
    'vehicle': 'edge',
}


def refine_for(category, alpha, image, strategy):
    """Dispatch to the category's refiner. Falls back to generic edge."""
    from .edge import edge_refine

    name = _CATEGORY_TO_NAME.get(category, strategy.get('refine', 'edge'))
    module = _NAME_TO_MODULE.get(name, 'edge')
    if module == 'edge':
        return edge_refine(alpha, image, strategy)

    if module == 'hair':
        from .hair import refine as fn
    elif module == 'fur':
        from .fur import refine as fn
    elif module == 'plant':
        from .plant import refine as fn
    elif module == 'logo':
        from .logo import refine as fn
    else:  # cartoon
        from .cartoon import refine as fn
    return fn(alpha, image, strategy, fallback=edge_refine)


__all__ = ['refine_for']