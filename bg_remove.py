"""Background removal — public facade for the adaptive engine.

Keeps the exact function contracts that ``app.py`` relies on
(``available``, ``download_status``, ``start_download``,
``remove_background`` → ``(w, h)``) while delegating the heavy lifting to
the :mod:`background_remover` package: content classification, adaptive
segmentation, trimap + alpha matting, category refinement, halo removal,
mask cleanup, quality scoring and compositing.
"""

import model_store

from background_remover import get_pipeline


def model_path():
    return model_store.model_path('bg')


def available():
    return model_store.available('bg')


def download_status():
    return model_store.status('bg')


def start_download():
    model_store.start_download('bg')


def remove_background(in_path, out_path, background='transparent',
                      progress_cb=None, custom_image=None, add_shadow=False):
    """Remove the background via the adaptive AI pipeline.

    ``background`` may be ``'transparent'``, a ``'#rrggbb'`` colour,
    ``'blur'`` (original photo blurred) or ``'custom'`` (an image file
    provided via ``custom_image``). Returns ``(width, height)``.
    """
    result = get_pipeline().remove(
        in_path, out_path, background=background, method='ai',
        progress_cb=progress_cb, custom_image=custom_image,
        add_shadow=add_shadow)
    return result.width, result.height


def remove_by_color(in_path, out_path, color='#ffffff', tolerance=30,
                    background='transparent', progress_cb=None,
                    custom_image=None, add_shadow=False):
    """Remove a solid-colour background (no AI model required).

    For logos, documents and product shots on clean backgrounds. Still runs
    through the adaptive pipeline so labels get the sharp-logo treatment.
    Returns ``(width, height)``.
    """
    result = get_pipeline().remove(
        in_path, out_path, background=background, method='color',
        color=color, tolerance=tolerance, progress_cb=progress_cb,
        custom_image=custom_image, add_shadow=add_shadow)
    return result.width, result.height