"""BiRefNet / ISNet segmentation engines (ONNX Runtime).

Wraps the background-removal model family located through the app's
``paths.find_model`` registry (MIT licence):

- ``birefnet-portrait-fp16.onnx`` — optimal for humans / animals (trained on
  P3M-10k portrait data: hair, fur and skin edges are dramatically cleaner).
- ``birefnet-general-fp16.onnx``  — fast general-purpose variant.
- ``birefnet-general.onnx``      — original general purpose (fp32).
- ``birefnet-lite.onnx``         — smallest / fastest BiRefNet fallback.
- ``isnet-general-use.onnx``     — the CPU workhorse (~178MB, seconds per
  image instead of minutes).

**Automatic hardware-aware selection** (requirement: fast on CPU, maximal
quality on GPU):

- GPU (CUDA / DirectML) → the BiRefNet priority chain, best quality wins.
- CPU-only               → ``isnet-general-use`` first: a single inference is
  ~2s instead of the 5+ minutes the BiRefNet-SwinL fp32 takes at 1024×1024.
  The pipeline then refines with matting / decontamination as usual.

``make_segmenter(kind)`` is the single entry point the pipeline uses.
"""

import os

from .base import ONNXSegmenter

_GPU_GENERAL = (
    'birefnet-general-fp16.onnx',
    'birefnet-general.onnx',
    'birefnet-lite.onnx',
)

_GPU_PORTRAIT = (
    'birefnet-portrait-fp16.onnx',
    'birefnet-general-fp16.onnx',
    'birefnet-general.onnx',
    'birefnet-lite.onnx',
)

# CPU-only: ISNet first (seconds), BiRefNet fp16/fp32 only as a fallback when
# the fast model has not been downloaded (still slow, but functional).
_CPU_GENERAL = (
    'isnet-general-use.onnx',
    'birefnet-general-fp16.onnx',
    'birefnet-general.onnx',
    'birefnet-lite.onnx',
)

_CPU_PORTRAIT = (
    'isnet-general-use.onnx',
    'birefnet-portrait-fp16.onnx',
    'birefnet-general-fp16.onnx',
    'birefnet-general.onnx',
    'birefnet-lite.onnx',
)

_ISNET = 'isnet-general-use.onnx'


def _find_first(names):
    try:
        import paths
    except Exception:
        return None
    for name in names:
        path = paths.find_model(name)
        if path:
            return path
    return None


def _priority_for(portrait):
    try:
        from ..models import has_gpu
        gpu = has_gpu()
    except Exception:
        gpu = False
    if gpu:
        return _GPU_PORTRAIT if portrait else _GPU_GENERAL
    return _CPU_PORTRAIT if portrait else _CPU_GENERAL


def _resolve_model(portrait):
    """Return (model_path, is_fast) for the strongest usable model."""
    for name in _priority_for(portrait):
        p = _find_first((name,))
        if p:
            return p, name == _ISNET
    return None, False


def make_segmenter(kind='general'):
    """Instantiate the best segmenter for this machine, given ``kind``
    ('general' | 'portrait'). Portrait is used for human/animal subjects."""
    portrait = kind in ('portrait', 'human', 'animal')
    path, fast = _resolve_model(portrait)
    if not path:
        raise FileNotFoundError(
            'No background-removal model installed — run the one-time '
            'setup download first.')
    if fast:
        return ISNetSegmenter(path)
    return BiRefNetSegmenter(path, portrait=portrait)


class ISNetSegmenter(ONNXSegmenter):
    """ISNet-general-use — the fast CPU model (~2s / 1024x1024).

    Input 1024x1024, centre-normalised ((x/255) − 0.5), sigmoid output.
    """

    name = 'isnet'

    def __init__(self, model_path=None, input_size=1024, norm='half'):
        path = model_path or _find_first(_CPU_GENERAL)
        if not path or not os.path.exists(path):
            raise FileNotFoundError('ISNet model not installed — run the '
                                    'one-time setup download first.')
        super().__init__(path, input_size=input_size, norm=norm)

    @property
    def available(self):
        return bool(self.model_path and os.path.exists(self.model_path))


class BiRefNetSegmenter(ONNXSegmenter):
    name = 'birefnet'

    def __init__(self, model_path=None, portrait=False, input_size=1024,
                 norm='imagenet'):
        if model_path is None:
            path, _ = _resolve_model(portrait)
        else:
            path = model_path
        if not path or not os.path.exists(path):
            raise FileNotFoundError('BiRefNet model not installed — run '
                                    'the one-time setup download first.')
        super().__init__(path, input_size=input_size, norm=norm)

    @property
    def available(self):
        return bool(self.model_path and os.path.exists(self.model_path))


class BiRefNetPortraitSegmenter(BiRefNetSegmenter):
    """Portrait-optimised BiRefNet — best hair / fur / skin result.

    Falls back to the strongest installed general model when the portrait
    weights are not present so the engine never hard-fails.
    """

    name = 'birefnet_portrait'

    def __init__(self, model_path=None, input_size=1024, norm='imagenet'):
        super().__init__(model_path=model_path, portrait=True,
                         input_size=input_size, norm=norm)